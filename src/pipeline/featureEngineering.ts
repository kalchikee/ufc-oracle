// UFC Oracle v4.1 — Feature Engineering
// Computes the full 40+ feature vector as Fighter A – Fighter B differentials.
// All striking stats are per-minute rates; wrestling stats are per-15-min.

import type { Fighter, FightCardBout, FeatureVector, WeightClass } from '../types.js';
import { eloFeatures } from '../elo/eloSystem.js';
import { encodeStyleMatchup, encodeStanceMatchup } from '../style-model/styleClassifier.js';
import campData from '../../config/camps.json' assert { type: 'json' };
import ageCurves from '../../config/age_curves.json' assert { type: 'json' };

// ─── Weight class encoding ────────────────────────────────────────────────────

const WEIGHT_CLASS_ORDER: WeightClass[] = [
  'WomenStrawweight', 'Strawweight', 'WomenFlyweight', 'Flyweight',
  'WomenBantamweight', 'Bantamweight', 'WomenFeatherweight', 'Featherweight',
  'Lightweight', 'Welterweight', 'Middleweight', 'LightHeavyweight', 'Heavyweight',
];

function encodeWeightClass(wc: WeightClass): number {
  return WEIGHT_CLASS_ORDER.indexOf(wc);
}

// ─── Age penalty (non-linear) ─────────────────────────────────────────────────

function agePerformanceModifier(ageYears: number, weightClass: WeightClass): number {
  // Peak range per weight class
  const heavy = ['Heavyweight', 'LightHeavyweight'];
  const light = ['Flyweight', 'Bantamweight', 'WomenFlyweight', 'WomenStrawweight', 'WomenBantamweight', 'Strawweight'];

  let peakStart: number, peakEnd: number, declineRate: number;

  if (heavy.includes(weightClass)) {
    peakStart = 28; peakEnd = 35; declineRate = 0.015;
  } else if (light.includes(weightClass)) {
    peakStart = 27; peakEnd = 30; declineRate = 0.018;
  } else {
    peakStart = 28; peakEnd = 32; declineRate = 0.016;
  }

  if (ageYears < peakStart) {
    // Under peak: slight inexperience penalty (diminishing as they approach peak)
    return -Math.max(0, (peakStart - ageYears) * 0.005);
  } else if (ageYears <= peakEnd) {
    return 0; // in prime
  } else {
    // Past prime: accelerating decline
    const yearsDecline = ageYears - peakEnd;
    return -(yearsDecline * yearsDecline * declineRate);
  }
}

// ─── Layoff penalty ───────────────────────────────────────────────────────────

function layoffPenalty(daysSinceFight?: number): number {
  if (!daysSinceFight) return 0;
  const months = daysSinceFight / 30.44;
  if (months <= 6) return 0;
  if (months <= 12) return -0.02;
  if (months <= 18) return -0.05;
  return -0.08;
}

// ─── Camp quality ─────────────────────────────────────────────────────────────

function campQuality(campName?: string): number {
  if (!campName) return 0;
  const lower = campName.toLowerCase();
  const camps = campData as Record<string, number>;
  for (const [name, tier] of Object.entries(camps)) {
    if (lower.includes(name.toLowerCase())) return tier;
  }
  return 0; // unknown camp
}

// ─── Prior opponent quality ───────────────────────────────────────────────────
// Simplified: use win percentage as a proxy for opponent quality.
// Full implementation would compute avg Elo of last 5 opponents from fight history.

function priorOpponentQuality(fighter: Fighter): number {
  // Proxy: UFC win% (higher UFC win% implies fought tougher competition)
  const ufcFights = fighter.ufcWins + fighter.ufcLosses;
  if (ufcFights === 0) return 0;
  return fighter.ufcWins / ufcFights;
}

// ─── Age (years) from date of birth ──────────────────────────────────────────

function ageFromDOB(dob?: string): number {
  if (!dob) return 29; // league-average UFC fighter age
  const birth = new Date(dob);
  const now = new Date();
  return (now.getTime() - birth.getTime()) / (1000 * 60 * 60 * 24 * 365.25);
}

// ─── Elevation flag ───────────────────────────────────────────────────────────
// Approximate: detect high-altitude venues by location string
const HIGH_ALTITUDE_VENUES = ['mexico city', 'denver', 'bogota', 'quito', 'la paz', 'albuquerque'];

export function isHighAltitude(location: string): number {
  const lower = location.toLowerCase();
  return HIGH_ALTITUDE_VENUES.some(v => lower.includes(v)) ? 1 : 0;
}

// ─── Main feature engineering ─────────────────────────────────────────────────

export function buildFeatureVector(
  fighterA: Fighter,
  fighterB: Fighter,
  bout: FightCardBout,
  eventLocation: string,
): FeatureVector {
  const elo = eloFeatures(fighterA, fighterB);

  const ageA = ageFromDOB(fighterA.dateOfBirth);
  const ageB = ageFromDOB(fighterB.dateOfBirth);

  const layoffA = fighterA.daysSinceLastFight;
  const layoffB = fighterB.daysSinceLastFight;

  const campA = campQuality(fighterA.camp);
  const campB = campQuality(fighterB.camp);

  const oppQualA = priorOpponentQuality(fighterA);
  const oppQualB = priorOpponentQuality(fighterB);

  const recentWinPctA = (fighterA.recentWins + fighterA.recentLosses) > 0
    ? fighterA.recentWins / (fighterA.recentWins + fighterA.recentLosses)
    : 0.5;
  const recentWinPctB = (fighterB.recentWins + fighterB.recentLosses) > 0
    ? fighterB.recentWins / (fighterB.recentWins + fighterB.recentLosses)
    : 0.5;

  return {
    // Elo
    elo_diff: elo.elo_diff,
    striking_elo_diff: elo.striking_elo_diff,
    grappling_elo_diff: elo.grappling_elo_diff,

    // Striking offense/defense
    sig_strikes_landed_pm_diff: fighterA.sigStrikesLandedPM - fighterB.sigStrikesLandedPM,
    sig_strike_accuracy_diff: fighterA.sigStrikeAccuracy - fighterB.sigStrikeAccuracy,
    sig_strikes_absorbed_pm_diff: fighterA.sigStrikesAbsorbedPM - fighterB.sigStrikesAbsorbedPM,
    strike_defense_pct_diff: fighterA.sigStrikeDefense - fighterB.sigStrikeDefense,

    // Wrestling
    takedown_avg_diff: fighterA.takedownAvgPer15 - fighterB.takedownAvgPer15,
    takedown_accuracy_diff: fighterA.takedownAccuracy - fighterB.takedownAccuracy,
    takedown_defense_diff: fighterA.takedownDefense - fighterB.takedownDefense,

    // Grappling
    submission_avg_diff: fighterA.submissionAvgPer15 - fighterB.submissionAvgPer15,
    control_time_pct_diff: fighterA.controlTimePct - fighterB.controlTimePct,

    // Power
    knockdown_rate_diff: fighterA.knockdownRate - fighterB.knockdownRate,

    // Physical
    reach_diff: (fighterA.reach ?? 72) - (fighterB.reach ?? 72),
    height_diff: (fighterA.height ?? 70) - (fighterB.height ?? 70),

    // Age
    age_diff: ageA - ageB,
    age_fighter_a: ageA,
    age_fighter_b: ageB,

    // Career record
    win_pct_diff: fighterA.winPct - fighterB.winPct,
    ufc_win_pct_diff: (fighterA.ufcWins / Math.max(1, fighterA.ufcWins + fighterA.ufcLosses))
                    - (fighterB.ufcWins / Math.max(1, fighterB.ufcWins + fighterB.ufcLosses)),
    finish_rate_diff: fighterA.finishRate - fighterB.finishRate,
    decision_rate_diff: fighterA.decisionRate - fighterB.decisionRate,
    avg_fight_time_diff: fighterA.avgFightTime - fighterB.avgFightTime,

    // Activity / layoff
    days_since_last_fight_diff: (layoffA ?? 90) - (layoffB ?? 90),
    fighter_a_layoff: layoffA ?? 90,
    fighter_b_layoff: layoffB ?? 90,

    // Momentum
    win_streak_diff: fighterA.winStreak - fighterB.winStreak,
    recent_3_win_pct_diff: recentWinPctA - recentWinPctB,
    recent_3_sig_strikes_diff: fighterA.recentSigStrikesPM - fighterB.recentSigStrikesPM,

    // Fight context
    weight_class_encoded: encodeWeightClass(bout.weightClass),
    title_fight_flag: bout.isTitleFight ? 1 : 0,
    main_event_flag: bout.isMainEvent ? 1 : 0,
    rounds_scheduled: bout.scheduledRounds,

    // Stylistic
    stance_matchup: encodeStanceMatchup(fighterA.stance, fighterB.stance),
    style_matchup_encoded: encodeStyleMatchup(fighterA.style, fighterB.style),

    // Camp
    camp_quality_diff: campA - campB,

    // Environment
    elevation_flag: isHighAltitude(eventLocation),

    // Quality of competition
    prior_opponent_quality_diff: oppQualA - oppQualB,
  };
}

// ─── Feature normalization (z-score) ─────────────────────────────────────────

// Mean and std from historical 2015–2025 dataset (approximate)
const FEATURE_STATS: Record<string, { mean: number; std: number }> = {
  elo_diff: { mean: 0, std: 200 },
  striking_elo_diff: { mean: 0, std: 200 },
  grappling_elo_diff: { mean: 0, std: 200 },
  sig_strikes_landed_pm_diff: { mean: 0, std: 2.0 },
  sig_strike_accuracy_diff: { mean: 0, std: 0.1 },
  sig_strikes_absorbed_pm_diff: { mean: 0, std: 2.0 },
  strike_defense_pct_diff: { mean: 0, std: 0.1 },
  takedown_avg_diff: { mean: 0, std: 2.0 },
  takedown_accuracy_diff: { mean: 0, std: 0.15 },
  takedown_defense_diff: { mean: 0, std: 0.15 },
  submission_avg_diff: { mean: 0, std: 0.8 },
  control_time_pct_diff: { mean: 0, std: 0.15 },
  knockdown_rate_diff: { mean: 0, std: 0.5 },
  reach_diff: { mean: 0, std: 4 },
  height_diff: { mean: 0, std: 3 },
  age_diff: { mean: 0, std: 5 },
  age_fighter_a: { mean: 29, std: 4 },
  age_fighter_b: { mean: 29, std: 4 },
  win_pct_diff: { mean: 0, std: 0.15 },
  ufc_win_pct_diff: { mean: 0, std: 0.2 },
  finish_rate_diff: { mean: 0, std: 0.2 },
  decision_rate_diff: { mean: 0, std: 0.2 },
  avg_fight_time_diff: { mean: 0, std: 3 },
  days_since_last_fight_diff: { mean: 0, std: 180 },
  fighter_a_layoff: { mean: 120, std: 120 },
  fighter_b_layoff: { mean: 120, std: 120 },
  win_streak_diff: { mean: 0, std: 3 },
  recent_3_win_pct_diff: { mean: 0, std: 0.3 },
  recent_3_sig_strikes_diff: { mean: 0, std: 1.5 },
  weight_class_encoded: { mean: 6, std: 3 },
  title_fight_flag: { mean: 0.1, std: 0.3 },
  main_event_flag: { mean: 0.1, std: 0.3 },
  rounds_scheduled: { mean: 3.2, std: 0.8 },
  stance_matchup: { mean: 0, std: 0.03 },
  style_matchup_encoded: { mean: 0, std: 0.05 },
  camp_quality_diff: { mean: 0, std: 1.5 },
  elevation_flag: { mean: 0.05, std: 0.22 },
  prior_opponent_quality_diff: { mean: 0, std: 0.2 },
};

export function normalizeFeatures(fv: FeatureVector): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, val] of Object.entries(fv)) {
    const stats = FEATURE_STATS[key];
    if (stats && stats.std > 0) {
      out[key] = (val - stats.mean) / stats.std;
    } else {
      out[key] = val;
    }
  }
  return out;
}
