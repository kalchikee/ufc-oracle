// UFC Oracle v4.1 — Core Type Definitions

// ─── Fighter types ────────────────────────────────────────────────────────────

export type FightStyle = 'Striker' | 'Wrestler' | 'SubmissionArtist' | 'PressureFighter' | 'CounterStriker' | 'WellRounded';

export type WeightClass =
  | 'Strawweight' | 'Flyweight' | 'Bantamweight' | 'Featherweight'
  | 'Lightweight' | 'Welterweight' | 'Middleweight' | 'LightHeavyweight'
  | 'Heavyweight' | 'WomenStrawweight' | 'WomenFlyweight' | 'WomenBantamweight' | 'WomenFeatherweight';

export type FightMethod = 'KO/TKO' | 'Submission' | 'Decision' | 'Other';

export interface Fighter {
  fighterId: string;          // UFCStats fighter URL slug
  name: string;
  nickname?: string;
  weightClass: WeightClass;
  height?: number;            // inches
  reach?: number;             // inches
  stance?: string;            // Orthodox | Southpaw | Switch
  dateOfBirth?: string;       // YYYY-MM-DD
  camp?: string;
  // Career stats (per-minute / per-15-min rates)
  sigStrikesLandedPM: number;     // significant strikes landed per minute
  sigStrikesAbsorbedPM: number;   // significant strikes absorbed per minute
  sigStrikeAccuracy: number;      // 0–1
  sigStrikeDefense: number;       // 0–1
  takedownAvgPer15: number;       // takedowns per 15 min
  takedownAccuracy: number;       // 0–1
  takedownDefense: number;        // 0–1
  submissionAvgPer15: number;     // submission attempts per 15 min
  knockdownRate: number;          // knockdowns per 15 min
  controlTimePct: number;         // 0–1 of ground-control time
  avgFightTime: number;           // minutes
  // Career record
  wins: number;
  losses: number;
  draws: number;
  winPct: number;
  ufcWins: number;
  ufcLosses: number;
  finishRate: number;             // % of wins by KO or Sub
  decisionRate: number;           // % of fights going to decision
  // Derived style
  style: FightStyle;
  // Elo ratings
  eloOverall: number;
  eloStriking: number;
  eloGrappling: number;
  // Activity
  lastFightDate?: string;         // YYYY-MM-DD
  daysSinceLastFight?: number;
  // Recent form (last 3 fights)
  recentWins: number;
  recentLosses: number;
  recentSigStrikesPM: number;
  winStreak: number;
  updatedAt: string;
}

export interface Fight {
  fightId: string;
  eventId: string;
  eventName: string;
  eventDate: string;          // YYYY-MM-DD
  fighterAId: string;
  fighterBId: string;
  weightClass: WeightClass;
  isMainEvent: boolean;
  isTitleFight: boolean;
  scheduledRounds: number;    // 3 or 5
  actualRounds?: number;
  winnerId?: string;
  method?: FightMethod;
  round?: number;
  time?: string;
}

export interface FightCard {
  eventId: string;
  eventName: string;
  eventDate: string;          // YYYY-MM-DD
  location: string;
  venue: string;
  fights: FightCardBout[];
}

export interface FightCardBout {
  fightId: string;
  fighterA: string;           // name
  fighterB: string;           // name
  fighterAId?: string;
  fighterBId?: string;
  weightClass: WeightClass;
  isMainEvent: boolean;
  isTitleFight: boolean;
  scheduledRounds: number;
  cardPosition: 'main_event' | 'co_main' | 'main_card' | 'prelim' | 'early_prelim';
  // Optional odds (moneyline American format)
  fighterAMoneyline?: number;
  fighterBMoneyline?: number;
}

// ─── Feature vector (all features are Fighter A – Fighter B diffs) ────────────

export interface FeatureVector {
  // Elo differentials
  elo_diff: number;
  striking_elo_diff: number;
  grappling_elo_diff: number;

  // Striking offense/defense
  sig_strikes_landed_pm_diff: number;
  sig_strike_accuracy_diff: number;
  sig_strikes_absorbed_pm_diff: number;
  strike_defense_pct_diff: number;

  // Wrestling
  takedown_avg_diff: number;
  takedown_accuracy_diff: number;
  takedown_defense_diff: number;

  // Grappling
  submission_avg_diff: number;
  control_time_pct_diff: number;

  // Power
  knockdown_rate_diff: number;

  // Physical
  reach_diff: number;
  height_diff: number;

  // Age
  age_diff: number;
  age_fighter_a: number;
  age_fighter_b: number;

  // Career record
  win_pct_diff: number;
  ufc_win_pct_diff: number;
  finish_rate_diff: number;
  decision_rate_diff: number;
  avg_fight_time_diff: number;

  // Activity / layoff
  days_since_last_fight_diff: number;
  fighter_a_layoff: number;
  fighter_b_layoff: number;

  // Momentum / recent form
  win_streak_diff: number;
  recent_3_win_pct_diff: number;
  recent_3_sig_strikes_diff: number;

  // Fight context
  weight_class_encoded: number;
  title_fight_flag: number;
  main_event_flag: number;
  rounds_scheduled: number;

  // Stylistic
  stance_matchup: number;
  style_matchup_encoded: number;

  // Camp quality
  camp_quality_diff: number;

  // Environment
  elevation_flag: number;

  // Quality of competition
  prior_opponent_quality_diff: number;
}

// ─── Model outputs ────────────────────────────────────────────────────────────

export interface Prediction {
  predictionId: string;
  fightId: string;
  eventId: string;
  eventName: string;
  eventDate: string;
  fighterAId: string;
  fighterBId: string;
  fighterAName: string;
  fighterBName: string;
  weightClass: WeightClass;
  cardPosition: string;
  isMainEvent: boolean;
  isTitleFight: boolean;
  scheduledRounds: number;
  featureVector: FeatureVector;
  // Winner model
  fighterAWinProb: number;         // calibrated
  fighterBWinProb: number;
  predictedWinnerId: string;
  predictedWinnerName: string;
  confidenceTier: ConfidenceTier;
  // Method model
  koProb: number;
  submissionProb: number;
  decisionProb: number;
  otherProb: number;
  predictedMethod: FightMethod;
  // Market edge
  fighterAMoneyline?: number;
  fighterBMoneyline?: number;
  vegasWinnerProb?: number;       // vig-removed
  edge?: number;                  // model – vegas (for predicted winner)
  edgeCategory?: EdgeCategory;
  // Outcome (filled after event)
  actualWinnerId?: string;
  actualMethod?: FightMethod;
  correct?: boolean;
  methodCorrect?: boolean;
  // Meta
  modelVersion: string;
  createdAt: string;
}

export type ConfidenceTier = 'coin_flip' | 'lean' | 'strong' | 'high_conviction' | 'lock';
export type EdgeCategory = 'none' | 'value' | 'priority' | 'extreme';

// ─── Elo types ────────────────────────────────────────────────────────────────

export interface EloRatings {
  fighterId: string;
  eloOverall: number;
  eloStriking: number;
  eloGrappling: number;
  gamesPlayed: number;
  updatedAt: string;
}

// ─── Accuracy tracking ────────────────────────────────────────────────────────

export interface AccuracyStats {
  ytdCorrect: number;
  ytdTotal: number;
  ytdAccuracy: number;
  highConvCorrect: number;
  highConvTotal: number;
  highConvAccuracy: number;
  methodCorrect: number;
  methodTotal: number;
  methodAccuracy: number;
  valueBetROI: number;
  underdogCorrect: number;
  underdogTotal: number;
  mainEventCorrect: number;
  mainEventTotal: number;
  brierScore: number;
  eventRecord: EventRecord[];
}

export interface EventRecord {
  eventId: string;
  eventName: string;
  eventDate: string;
  correct: number;
  total: number;
  accuracy: number;
}

// ─── Model weights (JSON format) ─────────────────────────────────────────────

export interface ModelWeights {
  intercept: number;
  coefficients: Record<string, number>;
  featureNames: string[];
  trainedOn: string;      // YYYY-MM-DD
  cvAccuracy: number;
  brierScore: number;
  version: string;
}

export interface MethodModelWeights {
  intercept: number[];                           // one per class
  coefficients: Record<string, number[]>;        // feature -> [ko_coef, sub_coef, dec_coef]
  classes: string[];                             // ['KO/TKO', 'Submission', 'Decision']
  featureNames: string[];
  trainedOn: string;
  cvAccuracy: number;
  version: string;
}

export interface CalibrationWeights {
  a: number;    // Platt scaling: sigmoid(a * raw_score + b)
  b: number;
  trainedOn: string;
}
