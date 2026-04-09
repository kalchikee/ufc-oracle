"""
UFC Oracle v4.1 — ML Model Training
Trains two models from the historical dataset (data/training_dataset.csv):
  1. Binary logistic regression for winner prediction (L2, C=0.5–1.0)
  2. Multinomial LR for method of victory (KO/TKO, Submission, Decision)
Then applies Platt scaling for calibration.
Exports: model/winner_model.json, model/method_model.json, model/calibration.json
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

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
]

def load_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATASET_PATH}. Run build_dataset.py first.')
    df = pd.read_csv(DATASET_PATH)
    log.info(f'Loaded dataset: {len(df)} rows, {df["label"].value_counts().to_dict()} label balance')

    # Drop rows with too many missing features
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_cols)
    if missing:
        log.warning(f'Missing columns (will use 0): {missing}')
        for col in missing:
            df[col] = 0.0

    df = df.dropna(subset=['label'])
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df

def walk_forward_cv(df: pd.DataFrame, model: LogisticRegression) -> dict:
    """Walk-forward cross-validation: train on 2015–N, test on N+1."""
    df = df.sort_values('event_date').reset_index(drop=True)
    years = sorted(df['event_date'].str[:4].unique())

    if len(years) < 3:
        log.warning('Not enough years for walk-forward CV, falling back to k-fold')
        X = df[FEATURE_COLS].values
        y = df['label'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        return {'accuracy_mean': float(scores.mean()), 'accuracy_std': float(scores.std())}

    all_preds = []
    all_labels = []

    for i, test_year in enumerate(years[2:], start=2):
        train_years = years[:i]
        train_mask = df['event_date'].str[:4].isin(train_years)
        test_mask = df['event_date'].str[:4] == test_year

        X_train = df.loc[train_mask, FEATURE_COLS].values
        y_train = df.loc[train_mask, 'label'].values
        X_test = df.loc[test_mask, FEATURE_COLS].values
        y_test = df.loc[test_mask, 'label'].values

        if len(X_test) == 0:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        m = LogisticRegression(C=model.C, penalty='l2', max_iter=1000, solver='lbfgs')
        m.fit(X_train_s, y_train)
        preds = m.predict(X_test_s)
        all_preds.extend(preds)
        all_labels.extend(y_test)

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    log.info(f'Walk-forward CV accuracy: {acc:.4f} ({len(all_labels)} samples)')
    return {'accuracy_mean': acc, 'accuracy_std': 0.0}

def train_winner_model(df: pd.DataFrame):
    log.info('Training winner prediction model...')
    X = df[FEATURE_COLS].values
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tune C via walk-forward CV
    best_c, best_acc = 1.0, 0.0
    for c in [0.3, 0.5, 0.75, 1.0]:
        m = LogisticRegression(C=c, penalty='l2', max_iter=1000, solver='lbfgs')
        cv_results = walk_forward_cv(df, m)
        log.info(f'  C={c}: {cv_results["accuracy_mean"]:.4f}')
        if cv_results['accuracy_mean'] > best_acc:
            best_acc = cv_results['accuracy_mean']
            best_c = c

    log.info(f'Best C={best_c} (CV accuracy: {best_acc:.4f})')

    # Train final model on full dataset
    model = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, solver='lbfgs')
    model.fit(X_scaled, y)

    # Platt scaling calibration
    cal_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    cal_model.fit(X_scaled, y)

    # Brier score on full dataset (overfitted — use CV score for real eval)
    probs = cal_model.predict_proba(X_scaled)[:, 1]
    brier = brier_score_loss(y, probs)
    log.info(f'Brier score (train, overfitted): {brier:.4f}')

    # Extract calibration parameters (Platt: sigmoid(a*x + b))
    calibrator = cal_model.calibrated_classifiers_[0].calibrators[0]
    cal_a = float(calibrator.a_)
    cal_b = float(calibrator.b_)

    # Export winner model
    winner_model = {
        'intercept': float(model.intercept_[0]),
        'coefficients': dict(zip(FEATURE_COLS, model.coef_[0].tolist())),
        'featureNames': FEATURE_COLS,
        'trainedOn': str(date.today()),
        'cvAccuracy': round(best_acc, 4),
        'brierScore': round(brier, 4),
        'version': '4.1.0',
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
    }

    calibration = {
        'a': cal_a,
        'b': cal_b,
        'trainedOn': str(date.today()),
    }

    (MODEL_DIR / 'winner_model.json').write_text(json.dumps(winner_model, indent=2))
    (MODEL_DIR / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    log.info('Winner model saved → model/winner_model.json + model/calibration.json')

    return winner_model, X_scaled, y, scaler

def train_method_model(df: pd.DataFrame, winner_probs: np.ndarray):
    log.info('Training method of victory model...')

    method_df = df[df['method'].isin(['KO/TKO', 'Submission', 'Decision'])].copy()
    if len(method_df) == 0:
        log.warning('No method labels found in dataset — skipping method model')
        return

    X = method_df[FEATURE_COLS].values
    # Add predicted winner probability as a feature
    X_with_winner = np.column_stack([X, winner_probs[:len(method_df)]])
    y = method_df['method'].values

    all_features = FEATURE_COLS + ['predicted_winner_prob']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_winner)

    model = LogisticRegression(
        C=1.0, penalty='l2', max_iter=1000, solver='lbfgs',
        multi_class='multinomial'
    )
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    log.info(f'Method model CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    method_model = {
        'intercept': model.intercept_.tolist(),
        'coefficients': {feat: coef.tolist() for feat, coef in zip(all_features, model.coef_.T)},
        'classes': model.classes_.tolist(),
        'featureNames': all_features,
        'trainedOn': str(date.today()),
        'cvAccuracy': round(float(cv_scores.mean()), 4),
        'version': '4.1.0',
    }

    (MODEL_DIR / 'method_model.json').write_text(json.dumps(method_model, indent=2))
    log.info('Method model saved → model/method_model.json')

def main():
    df = load_data()
    winner_model, X_scaled, y, scaler = train_winner_model(df)

    # Get winner probabilities for method model
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.preprocessing import StandardScaler as SS
    final_lr = LR(C=winner_model.get('C', 1.0), penalty='l2', max_iter=1000, solver='lbfgs')
    X = df[FEATURE_COLS].values
    sc = SS()
    Xs = sc.fit_transform(X)
    final_lr.fit(Xs, y)
    winner_probs = final_lr.predict_proba(Xs)[:, 1]

    train_method_model(df, winner_probs)

    log.info('Training complete!')
    log.info(f'  Winner CV accuracy: {winner_model["cvAccuracy"]:.1%}')
    log.info(f'  Target: >62% (high conviction >68%, lock >74%)')

if __name__ == '__main__':
    main()
