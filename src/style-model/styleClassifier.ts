// UFC Oracle v4.1 — Stylistic Classification & Matchup Model
// Styles: Striker | Wrestler | SubmissionArtist | PressureFighter | CounterStriker | WellRounded
// Matchup win rates derived from historical UFC data (2015–2025)

import type { FightStyle, Fighter } from '../types.js';

// ─── Style classification ─────────────────────────────────────────────────────

interface StyleInputs {
  sigStrikesLandedPM: number;
  takedownAvgPer15: number;
  submissionAvgPer15: number;
  sigStrikeDefense: number;   // 0–1
  sigStrikeAccuracy: number;  // 0–1
}

export function classifyStyle(s: StyleInputs): FightStyle {
  const { sigStrikesLandedPM, takedownAvgPer15, submissionAvgPer15, sigStrikeDefense, sigStrikeAccuracy } = s;

  // Submission Artist: sub attempts > 1.5/15min, meaningful grappling
  if (submissionAvgPer15 >= 1.5 && takedownAvgPer15 >= 1.5) return 'SubmissionArtist';

  // Wrestler: TD avg > 2.5, TD accuracy meaningful (encoded in takedownAvgPer15)
  if (takedownAvgPer15 >= 2.5) return 'Wrestler';

  // Pressure Fighter: high output (>6 sig strikes/min), not primarily a wrestler
  if (sigStrikesLandedPM >= 6.0 && takedownAvgPer15 < 2.0) return 'PressureFighter';

  // Striker: >5 sig strikes/min, low takedown activity
  if (sigStrikesLandedPM >= 5.0 && takedownAvgPer15 < 1.0) return 'Striker';

  // Counter Striker: high defense, high accuracy, modest volume
  if (sigStrikeDefense >= 0.65 && sigStrikeAccuracy >= 0.55) return 'CounterStriker';

  // Well-Rounded: no extreme in any dimension
  return 'WellRounded';
}

// ─── Matchup encoding ─────────────────────────────────────────────────────────

// Encode style matchup as a numeric feature the model can learn from.
// Positive = Fighter A has stylistic advantage, Negative = Fighter B advantage.
// Values from historical UFC win rates (approx).

const STYLE_WIN_RATES: Record<string, Record<string, number>> = {
  Striker: {
    Striker: 0.50,
    Wrestler: 0.42,          // Wrestler beats Striker ~58%
    SubmissionArtist: 0.55,  // Striker can KO sub artist
    PressureFighter: 0.48,
    CounterStriker: 0.45,    // Counter strikers frustrate strikers
    WellRounded: 0.48,
  },
  Wrestler: {
    Striker: 0.58,
    Wrestler: 0.50,
    SubmissionArtist: 0.44,  // Sub artists catch wrestlers from back
    PressureFighter: 0.54,
    CounterStriker: 0.56,
    WellRounded: 0.52,
  },
  SubmissionArtist: {
    Striker: 0.45,
    Wrestler: 0.56,
    SubmissionArtist: 0.50,
    PressureFighter: 0.48,
    CounterStriker: 0.50,
    WellRounded: 0.50,
  },
  PressureFighter: {
    Striker: 0.52,
    Wrestler: 0.46,
    SubmissionArtist: 0.52,
    PressureFighter: 0.50,
    CounterStriker: 0.42,    // Counter strikers neutralize pressure
    WellRounded: 0.49,
  },
  CounterStriker: {
    Striker: 0.55,
    Wrestler: 0.44,
    SubmissionArtist: 0.50,
    PressureFighter: 0.58,
    CounterStriker: 0.50,
    WellRounded: 0.51,
  },
  WellRounded: {
    Striker: 0.52,
    Wrestler: 0.48,
    SubmissionArtist: 0.50,
    PressureFighter: 0.51,
    CounterStriker: 0.49,
    WellRounded: 0.50,
  },
};

export function styleMatchupScore(styleA: FightStyle, styleB: FightStyle): number {
  // Returns Fighter A historical win rate vs Fighter B style, centered at 0
  const winRate = STYLE_WIN_RATES[styleA]?.[styleB] ?? 0.50;
  return winRate - 0.50; // centered: positive = A advantage, negative = B advantage
}

// Encode as numeric for model feature
export function encodeStyleMatchup(styleA: FightStyle, styleB: FightStyle): number {
  return styleMatchupScore(styleA, styleB);
}

// ─── Stance matchup encoding ─────────────────────────────────────────────────

// Orthodox vs Southpaw creates mirror-image striking angles that benefit the Southpaw slightly
// historically (~53% for Southpaw vs Orthodox in striking exchanges).

export function encodeStanceMatchup(stanceA?: string, stanceB?: string): number {
  const a = (stanceA || '').toLowerCase();
  const b = (stanceB || '').toLowerCase();

  if (a === 'orthodox' && b === 'southpaw') return -0.03; // slight disadvantage for A
  if (a === 'southpaw' && b === 'orthodox') return 0.03;  // slight advantage for A
  if (a === 'switch' || b === 'switch') return 0.01;      // switch stance slight advantage
  return 0;
}

// ─── Style label for Discord embeds ──────────────────────────────────────────

export function styleLabel(style: FightStyle): string {
  const labels: Record<FightStyle, string> = {
    Striker: '🥊 Striker',
    Wrestler: '🤼 Wrestler',
    SubmissionArtist: '🩹 Sub Artist',
    PressureFighter: '💨 Pressure',
    CounterStriker: '🛡️ Counter',
    WellRounded: '⚖️ Well-Rounded',
  };
  return labels[style] ?? style;
}

export function matchupDescription(styleA: FightStyle, styleB: FightStyle): string {
  const score = styleMatchupScore(styleA, styleB);
  if (Math.abs(score) < 0.02) return 'Even stylistic matchup';
  const favored = score > 0 ? 'Fighter A' : 'Fighter B';
  const adv = Math.abs(score);
  if (adv >= 0.07) return `${favored} has significant stylistic edge (${styleA} vs ${styleB})`;
  if (adv >= 0.04) return `${favored} has moderate stylistic edge (${styleA} vs ${styleB})`;
  return `${favored} slight stylistic edge (${styleA} vs ${styleB})`;
}
