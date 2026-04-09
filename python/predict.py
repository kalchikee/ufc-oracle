"""
UFC Oracle v4.1 — Python inference script
Called by predictionRunner.ts for each fight card.

Reads:  JSON array of feature vectors from stdin
Writes: JSON array of predictions to stdout

Each input item: { featureVector: Record<string, number> }
Each output item: { winnerProb: number, koProb: number, subProb: number, decProb: number }
"""

import json
import sys
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / 'model'

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
    # Trajectory: recent vs career trend
    'sig_strikes_trend_a', 'sig_strikes_trend_b', 'win_trend_diff',
]

# ─── Load models ──────────────────────────────────────────────────────────────

def load_meta() -> dict:
    meta_path = MODEL_DIR / 'model_meta.json'
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {'activeModel': 'logistic_regression'}

def load_winner_model(meta: dict):
    active = meta.get('activeModel', 'logistic_regression')

    if active == 'xgboost':
        xgb_path = MODEL_DIR / 'winner_model_xgb.json'
        if xgb_path.exists():
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(str(xgb_path))
                log.warning(f'Loaded XGBoost model (CV={meta.get("xgbCvAccuracy", "?")})')
                return ('xgboost', model, None, None)
            except Exception as e:
                log.warning(f'XGBoost load failed ({e}), falling back to LR')

    # Logistic Regression fallback
    lr_path = MODEL_DIR / 'winner_model.json'
    if not lr_path.exists():
        raise FileNotFoundError('No trained model found. Run python/train_model.py first.')

    lr_data = json.loads(lr_path.read_text())
    cal_data = json.loads((MODEL_DIR / 'calibration.json').read_text())

    intercept = lr_data['intercept']
    coefs = np.array([lr_data['coefficients'].get(f, 0.0) for f in FEATURE_COLS])
    scaler_mean = np.array(lr_data.get('scalerMean', [0.0] * len(FEATURE_COLS)))
    scaler_std  = np.array(lr_data.get('scalerStd',  [1.0] * len(FEATURE_COLS)))
    cal_a = cal_data['a']
    cal_b = cal_data['b']

    log.warning(f'Loaded LR model (CV={lr_data.get("cvAccuracy", "?")})')
    return ('logistic_regression', (intercept, coefs), (scaler_mean, scaler_std), (cal_a, cal_b))

def load_method_model() -> dict | None:
    path = MODEL_DIR / 'method_model.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())

# ─── Inference ────────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def softmax(xs: np.ndarray) -> np.ndarray:
    xs = xs - xs.max()
    e = np.exp(xs)
    return e / e.sum()

def predict_winner_lr(fv: np.ndarray, model_data: tuple, scaler: tuple, cal: tuple) -> float:
    intercept, coefs = model_data
    mean, std = scaler
    cal_a, cal_b = cal
    std = np.where(std == 0, 1.0, std)
    fv_scaled = (fv - mean) / std
    raw = intercept + np.dot(coefs, fv_scaled)
    return float(sigmoid(cal_a * raw + cal_b))

def predict_winner_xgb(fv: np.ndarray, model) -> float:
    prob = model.predict_proba(fv.reshape(1, -1))[0, 1]
    return float(prob)

def predict_method(fv: np.ndarray, winner_prob: float, method_data: dict) -> dict:
    features = method_data['featureNames']
    classes  = method_data['classes']
    s_mean   = np.array(method_data.get('scalerMean', [0.0] * len(features)))
    s_std    = np.array(method_data.get('scalerStd',  [1.0] * len(features)))

    # Build extended feature vector (include winner_prob as last feature)
    base_fv = fv.tolist() + [winner_prob]
    x = np.array([base_fv[i] if i < len(base_fv) else 0.0 for i in range(len(features))])
    s_std = np.where(s_std == 0, 1.0, s_std)
    x_s = (x - s_mean) / s_std

    # Compute logits per class
    intercepts = method_data['intercept']
    coefs_by_feat = method_data['coefficients']
    logits = np.array([
        intercepts[i] + sum(coefs_by_feat.get(f, [0.0] * len(classes))[i] * x_s[j]
                            for j, f in enumerate(features))
        for i in range(len(classes))
    ])
    probs = softmax(logits)

    result = dict(zip(classes, probs.tolist()))
    return {
        'koProb':  result.get('KO/TKO', 0.28),
        'subProb': result.get('Submission', 0.12),
        'decProb': result.get('Decision', 0.55),
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    raw = sys.stdin.read().strip()
    if not raw:
        sys.stdout.write(json.dumps([]))
        return

    items = json.loads(raw)
    if not items:
        sys.stdout.write(json.dumps([]))
        return

    meta = load_meta()
    model_type, model_data, scaler_data, cal_data = load_winner_model(meta)
    method_data = load_method_model()

    results = []
    for item in items:
        fv_dict = item.get('featureVector', {})
        fv = np.array([fv_dict.get(f, 0.0) for f in FEATURE_COLS], dtype=float)

        if model_type == 'xgboost':
            winner_prob = predict_winner_xgb(fv, model_data)
        else:
            winner_prob = predict_winner_lr(fv, model_data, scaler_data, cal_data)

        method = predict_method(fv, winner_prob, method_data) if method_data else {
            'koProb': 0.28, 'subProb': 0.12, 'decProb': 0.55
        }

        results.append({
            'winnerProb': winner_prob,
            'koProb':     method['koProb'],
            'subProb':    method['subProb'],
            'decProb':    method['decProb'],
        })

    sys.stdout.write(json.dumps(results))

if __name__ == '__main__':
    main()
