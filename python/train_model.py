"""
UFC Oracle v4.1 — ML Model Training
Trains winner + method models, comparing Logistic Regression vs XGBoost.
Recency weighting: recent fights count more (exponential decay by year).
Walk-forward CV selects the better model; both are exported for inference.

Exports:
  model/winner_model.json       — LR coefficients (TypeScript fallback)
  model/winner_model_xgb.json   — XGBoost model (used if better CV score)
  model/method_model.json       — Multinomial LR for method of victory
  model/calibration.json        — Platt scaling for LR
  model/model_meta.json         — Which model won + CV scores for logging
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    log.warning('xgboost not installed — will train LR only. Run: pip install xgboost')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning('shap not installed — feature importance will use gain only. Run: pip install shap')

DATASET_PATH = Path('data/training_dataset.csv')
MODEL_DIR = Path('model')
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    'elo_diff', 'striking_elo_diff', 'grappling_elo_diff',
    'sig_strikes_landed_pm_diff', 'sig_strike_accuracy_diff',
    'sig_strikes_absorbed_pm_diff', 'strike_defense_pct_diff',
    'takedown_avg_diff', 'takedown_accuracy_diff', 'takedown_defense_diff',
    'submission_avg_diff', 'control_time_pct_diff',
    'knockdown_rate_diff',
    'reach_diff', 'height_diff',
    'age_diff', 'age_fighter_a', 'age_fighter_b',
    'win_pct_diff', 'ufc_win_pct_diff', 'finish_rate_diff', 'decision_rate_diff', 'avg_fight_time_diff',
    'days_since_last_fight_diff', 'fighter_a_layoff', 'fighter_b_layoff',
    'win_streak_diff', 'recent_3_win_pct_diff', 'recent_3_sig_strikes_diff',
    'weight_class_encoded', 'title_fight_flag', 'main_event_flag', 'rounds_scheduled',
    'stance_matchup', 'style_matchup_encoded',
    'camp_quality_diff', 'elevation_flag', 'prior_opponent_quality_diff',
    # Trajectory: recent vs career trend (positive = improving)
    'sig_strikes_trend_a', 'sig_strikes_trend_b', 'win_trend_diff',
]

# ─── Recency weights ──────────────────────────────────────────────────────────
# Exponential decay: fights from N years ago get weight exp(-DECAY * N).
# DECAY=0.15 → 5yr-old fights get ~47% weight, 10yr-old fights get ~22%.
RECENCY_DECAY = 0.15

def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    current_year = date.today().year
    years_ago = df['event_date'].str[:4].astype(int).apply(lambda y: current_year - y)
    weights = np.exp(-RECENCY_DECAY * years_ago)
    # Normalize so sum == len(df) (preserves effective sample size interpretation)
    return (weights / weights.mean()).values

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATASET_PATH}. Run build_dataset.py first.')
    df = pd.read_csv(DATASET_PATH)
    log.info(f'Loaded dataset: {len(df)} rows, label balance: {df["label"].value_counts().to_dict()}')

    for col in FEATURE_COLS:
        if col not in df.columns:
            log.warning(f'Missing column {col} — filling with 0')
            df[col] = 0.0

    df = df.dropna(subset=['label'])
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df

# ─── Walk-forward CV (recency-weighted) ──────────────────────────────────────

def walk_forward_cv_lr(df: pd.DataFrame, C: float, weights: np.ndarray) -> float:
    df = df.copy()
    df['_weight'] = weights
    df = df.sort_values('event_date').reset_index(drop=True)
    years = sorted(df['event_date'].str[:4].unique())
    if len(years) < 3:
        return 0.0

    all_preds, all_labels = [], []
    for i, test_year in enumerate(years[2:], start=2):
        train_mask = df['event_date'].str[:4].isin(years[:i]).values
        test_mask  = (df['event_date'].str[:4] == test_year).values
        if not test_mask.any():
            continue

        X_tr, y_tr, w_tr = df.loc[train_mask, FEATURE_COLS].values, df.loc[train_mask, 'label'].values, df.loc[train_mask, '_weight'].values
        X_te, y_te        = df.loc[test_mask,  FEATURE_COLS].values, df.loc[test_mask,  'label'].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        m = LogisticRegression(C=C, penalty='l2', max_iter=1000, solver='lbfgs')
        m.fit(X_tr_s, y_tr, sample_weight=w_tr)
        all_preds.extend(m.predict(X_te_s))
        all_labels.extend(y_te)

    return accuracy_score(all_labels, all_preds) if all_labels else 0.0

def walk_forward_cv_xgb(df: pd.DataFrame, params: dict, weights: np.ndarray) -> float:
    df = df.copy()
    df['_weight'] = weights
    df = df.sort_values('event_date').reset_index(drop=True)
    years = sorted(df['event_date'].str[:4].unique())
    if len(years) < 3:
        return 0.0

    all_preds, all_labels = [], []
    for i, test_year in enumerate(years[2:], start=2):
        train_mask = df['event_date'].str[:4].isin(years[:i]).values
        test_mask  = (df['event_date'].str[:4] == test_year).values
        if not test_mask.any():
            continue

        X_tr, y_tr, w_tr = df.loc[train_mask, FEATURE_COLS].values, df.loc[train_mask, 'label'].values, df.loc[train_mask, '_weight'].values
        X_te, y_te        = df.loc[test_mask,  FEATURE_COLS].values, df.loc[test_mask,  'label'].values

        # XGBoost doesn't need scaling
        m = xgb.XGBClassifier(**params, eval_metric='logloss', verbosity=0)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        all_preds.extend(m.predict(X_te))
        all_labels.extend(y_te)

    return accuracy_score(all_labels, all_preds) if all_labels else 0.0

# ─── Train LR (recency-weighted) ──────────────────────────────────────────────

def train_lr(df: pd.DataFrame, weights: np.ndarray) -> tuple:
    log.info('Training Logistic Regression (recency-weighted)...')
    X = df[FEATURE_COLS].values
    y = df['label'].values

    # Tune C
    best_c, best_acc = 1.0, 0.0
    for c in [0.3, 0.5, 0.75, 1.0]:
        acc = walk_forward_cv_lr(df, c, weights)
        log.info(f'  LR C={c}: walk-forward CV = {acc:.4f}')
        if acc > best_acc:
            best_acc, best_c = acc, c

    log.info(f'Best LR: C={best_c}, CV={best_acc:.4f}')

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, solver='lbfgs')
    model.fit(X_s, y, sample_weight=weights)

    # Platt scaling
    cal = CalibratedClassifierCV(
        LogisticRegression(C=best_c, penalty='l2', max_iter=1000, solver='lbfgs'),
        method='sigmoid', cv=5
    )
    cal.fit(X_s, y, sample_weight=weights)
    probs = cal.predict_proba(X_s)[:, 1]
    brier = brier_score_loss(y, probs)

    cal_params = cal.calibrated_classifiers_[0].calibrators[0]

    winner_model_json = {
        'modelType': 'logistic_regression',
        'intercept': float(model.intercept_[0]),
        'coefficients': dict(zip(FEATURE_COLS, model.coef_[0].tolist())),
        'featureNames': FEATURE_COLS,
        'scalerMean': scaler.mean_.tolist(),
        'scalerStd': scaler.scale_.tolist(),
        'trainedOn': str(date.today()),
        'cvAccuracy': round(best_acc, 4),
        'brierScore': round(brier, 4),
        'recencyDecay': RECENCY_DECAY,
        'version': '4.1.0',
    }
    calibration_json = {
        'a': float(cal_params.a_),
        'b': float(cal_params.b_),
        'trainedOn': str(date.today()),
    }

    (MODEL_DIR / 'winner_model.json').write_text(json.dumps(winner_model_json, indent=2))
    (MODEL_DIR / 'calibration.json').write_text(json.dumps(calibration_json, indent=2))
    log.info(f'LR saved → model/winner_model.json (CV={best_acc:.4f}, Brier={brier:.4f})')

    return best_acc, model, scaler, probs

# ─── Train XGBoost (recency-weighted) ────────────────────────────────────────

def train_xgb(df: pd.DataFrame, weights: np.ndarray) -> tuple:
    log.info('Training XGBoost (recency-weighted)...')
    X = df[FEATURE_COLS].values
    y = df['label'].values

    # Tune key hyperparameters
    best_params = None
    best_acc = 0.0
    search_space = [
        {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
        {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1,  'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.0, 'reg_lambda': 1.0},
    ]

    for params in search_space:
        acc = walk_forward_cv_xgb(df, {**params, 'random_state': 42}, weights)
        log.info(f'  XGB depth={params["max_depth"]} lr={params["learning_rate"]} n={params["n_estimators"]}: CV={acc:.4f}')
        if acc > best_acc:
            best_acc, best_params = acc, params

    log.info(f'Best XGB: CV={best_acc:.4f} params={best_params}')

    # Train final model on full data
    final_xgb = xgb.XGBClassifier(
        **best_params,
        eval_metric='logloss',
        verbosity=0,
        random_state=42,
    )
    final_xgb.fit(X, y, sample_weight=weights)

    # Feature importances — gain-based (always available)
    importances = dict(zip(FEATURE_COLS, final_xgb.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    log.info('Top 10 XGB (gain): ' + ', '.join(f'{k}={v:.3f}' for k, v in top_features))

    # SHAP values — more reliable than gain for correlated features
    shap_importance = importances  # fallback to gain if shap unavailable
    if SHAP_AVAILABLE:
        try:
            log.info('Computing SHAP values...')
            explainer = shap.TreeExplainer(final_xgb)
            shap_vals = explainer.shap_values(X)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_importance = dict(zip(FEATURE_COLS, mean_abs_shap.tolist()))
            top_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
            log.info('Top 10 SHAP: ' + ', '.join(f'{k}={v:.4f}' for k, v in top_shap[:10]))
            low_impact = [k for k, v in top_shap if v < 0.001]
            if low_impact:
                log.info(f'Low-SHAP features (consider pruning): {low_impact}')
            (MODEL_DIR / 'feature_importance_shap.json').write_text(
                json.dumps(shap_importance, indent=2)
            )
            log.info('SHAP importances saved → model/feature_importance_shap.json')
        except Exception as e:
            log.warning(f'SHAP computation failed ({e}) — using gain importances')

    # Save XGBoost model (native JSON format)
    xgb_path = str(MODEL_DIR / 'winner_model_xgb.json')
    final_xgb.save_model(xgb_path)

    # Save metadata
    meta = {
        'modelType': 'xgboost',
        'params': best_params,
        'featureNames': FEATURE_COLS,
        'featureImportancesGain': importances,
        'featureImportancesShap': shap_importance,
        'trainedOn': str(date.today()),
        'cvAccuracy': round(best_acc, 4),
        'recencyDecay': RECENCY_DECAY,
        'version': '4.1.0',
    }
    (MODEL_DIR / 'winner_model_xgb_meta.json').write_text(json.dumps(meta, indent=2))
    log.info(f'XGB saved → model/winner_model_xgb.json (CV={best_acc:.4f})')

    probs = final_xgb.predict_proba(X)[:, 1]
    return best_acc, final_xgb, probs

# ─── Train method model ───────────────────────────────────────────────────────

def train_method_model(df: pd.DataFrame, winner_probs: np.ndarray, weights: np.ndarray):
    log.info('Training method of victory model...')
    method_df = df[df['method'].isin(['KO/TKO', 'Submission', 'Decision'])].copy()
    if len(method_df) == 0:
        log.warning('No method labels — skipping method model')
        return

    method_weights = weights[:len(method_df)]
    X = method_df[FEATURE_COLS].values
    X_with_winner = np.column_stack([X, winner_probs[:len(method_df)]])
    y = method_df['method'].values
    all_features = FEATURE_COLS + ['predicted_winner_prob']

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_with_winner)

    model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, solver='lbfgs', multi_class='multinomial')
    model.fit(X_s, y, sample_weight=method_weights)

    # Simple walk-forward accuracy estimate
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_s, y, cv=5, scoring='accuracy')
    log.info(f'Method model CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    method_model = {
        'modelType': 'logistic_regression',
        'intercept': model.intercept_.tolist(),
        'coefficients': {feat: coef.tolist() for feat, coef in zip(all_features, model.coef_.T)},
        'classes': model.classes_.tolist(),
        'featureNames': all_features,
        'scalerMean': scaler.mean_.tolist(),
        'scalerStd': scaler.scale_.tolist(),
        'trainedOn': str(date.today()),
        'cvAccuracy': round(float(cv_scores.mean()), 4),
        'recencyDecay': RECENCY_DECAY,
        'version': '4.1.0',
    }
    (MODEL_DIR / 'method_model.json').write_text(json.dumps(method_model, indent=2))
    log.info('Method model saved → model/method_model.json')

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    weights = compute_sample_weights(df)

    year_dist = df['event_date'].str[:4].value_counts().sort_index()
    log.info(f'Fights per year:\n{year_dist.to_string()}')
    log.info(f'Sample weight range: {weights.min():.3f} – {weights.max():.3f} (mean=1.0)')

    # Train LR (always)
    lr_acc, lr_model, lr_scaler, lr_probs = train_lr(df, weights)

    # Train XGBoost (if available)
    xgb_acc = 0.0
    winner_probs = lr_probs  # default to LR probs for method model

    if XGB_AVAILABLE:
        xgb_acc, xgb_model, xgb_probs = train_xgb(df, weights)
        winner_probs = xgb_probs if xgb_acc >= lr_acc else lr_probs
    else:
        log.info('Skipping XGBoost (not installed)')

    # Determine which model wins
    winner_model_type = 'xgboost' if XGB_AVAILABLE and xgb_acc > lr_acc else 'logistic_regression'
    winner_cv = max(lr_acc, xgb_acc)

    meta = {
        'activeModel': winner_model_type,
        'lrCvAccuracy': round(lr_acc, 4),
        'xgbCvAccuracy': round(xgb_acc, 4) if XGB_AVAILABLE else None,
        'winnerCvAccuracy': round(winner_cv, 4),
        'trainedOn': str(date.today()),
        'recencyDecay': RECENCY_DECAY,
    }
    (MODEL_DIR / 'model_meta.json').write_text(json.dumps(meta, indent=2))

    log.info('─' * 50)
    log.info(f'LR accuracy:  {lr_acc:.4f} ({lr_acc:.1%})')
    if XGB_AVAILABLE:
        log.info(f'XGB accuracy: {xgb_acc:.4f} ({xgb_acc:.1%})')
    log.info(f'Active model: {winner_model_type.upper()} ({winner_cv:.1%})')
    log.info(f'Target: >62% overall | >68% high-conviction | >74% locks')
    log.info('─' * 50)

    train_method_model(df, winner_probs, weights)
    log.info('All models saved.')

if __name__ == '__main__':
    main()
