// UFC Oracle v4.1 — Multi-Dimensional Elo System
// Three parallel Elo ratings per fighter: Overall, Striking, Grappling
// Overall Elo: K=32 | Striking Elo: K=40 on KO/TKO, K=15 dec, K=10 sub
// Grappling Elo: K=40 on Submission, K=15 dec, K=10 KO/TKO
// New fighters start at 1200 (below average 1500).
// Inactivity decay: >18 months → regress 10% toward mean per year absent.
// Weight class transfer: regress 5% toward mean on division change.

import type { Fighter, FightMethod } from '../types.js';
import { upsertFighter, getFighter, persistDb } from '../db/database.js';
import { logger } from '../logger.js';

const ELO_MEAN = 1500;
const K_OVERALL = 32;
const K_STRIKING_KO = 40;
const K_STRIKING_DEC = 15;
const K_STRIKING_SUB = 10;
const K_GRAPPLING_SUB = 40;
const K_GRAPPLING_DEC = 15;
const K_GRAPPLING_KO = 10;
const NEW_FIGHTER_ELO = 1200;

// ─── Core Elo math ────────────────────────────────────────────────────────────

export function expectedScore(ratingA: number, ratingB: number): number {
  return 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
}

export function eloUpdate(rating: number, expected: number, actual: number, k: number): number {
  return rating + k * (actual - expected);
}

// ─── K-factor selection ───────────────────────────────────────────────────────

function strikingK(method: FightMethod): number {
  if (method === 'KO/TKO') return K_STRIKING_KO;
  if (method === 'Decision') return K_STRIKING_DEC;
  return K_STRIKING_SUB;
}

function grapplingK(method: FightMethod): number {
  if (method === 'Submission') return K_GRAPPLING_SUB;
  if (method === 'Decision') return K_GRAPPLING_DEC;
  return K_GRAPPLING_KO;
}

// ─── Inactivity decay ─────────────────────────────────────────────────────────

export function applyInactivityDecay(fighter: Fighter, asOfDate: Date): Fighter {
  if (!fighter.lastFightDate) return fighter;
  const lastFight = new Date(fighter.lastFightDate);
  const monthsInactive = (asOfDate.getTime() - lastFight.getTime()) / (1000 * 60 * 60 * 24 * 30.44);
  if (monthsInactive <= 18) return fighter;

  const yearsAbsent = (monthsInactive - 18) / 12;
  const decayFactor = Math.pow(0.90, yearsAbsent); // 10% toward mean per year

  return {
    ...fighter,
    eloOverall: ELO_MEAN + (fighter.eloOverall - ELO_MEAN) * decayFactor,
    eloStriking: ELO_MEAN + (fighter.eloStriking - ELO_MEAN) * decayFactor,
    eloGrappling: ELO_MEAN + (fighter.eloGrappling - ELO_MEAN) * decayFactor,
  };
}

// ─── Weight class transfer regression ────────────────────────────────────────

export function applyWeightClassTransfer(fighter: Fighter): Fighter {
  const REGRESS = 0.05;
  return {
    ...fighter,
    eloOverall: ELO_MEAN + (fighter.eloOverall - ELO_MEAN) * (1 - REGRESS),
    eloStriking: ELO_MEAN + (fighter.eloStriking - ELO_MEAN) * (1 - REGRESS),
    eloGrappling: ELO_MEAN + (fighter.eloGrappling - ELO_MEAN) * (1 - REGRESS),
  };
}

// ─── Post-fight Elo update ────────────────────────────────────────────────────

export interface EloUpdateResult {
  winnerNewOverall: number;
  winnerNewStriking: number;
  winnerNewGrappling: number;
  loserNewOverall: number;
  loserNewStriking: number;
  loserNewGrappling: number;
}

export function computeEloUpdate(
  winner: Fighter,
  loser: Fighter,
  method: FightMethod,
): EloUpdateResult {
  const expWinner = expectedScore(winner.eloOverall, loser.eloOverall);
  const expLoser = 1 - expWinner;

  const expWinnerStr = expectedScore(winner.eloStriking, loser.eloStriking);
  const expWinnerGrp = expectedScore(winner.eloGrappling, loser.eloGrappling);

  const kStr = strikingK(method);
  const kGrp = grapplingK(method);

  return {
    winnerNewOverall: eloUpdate(winner.eloOverall, expWinner, 1, K_OVERALL),
    winnerNewStriking: eloUpdate(winner.eloStriking, expWinnerStr, 1, kStr),
    winnerNewGrappling: eloUpdate(winner.eloGrappling, expWinnerGrp, 1, kGrp),
    loserNewOverall: eloUpdate(loser.eloOverall, expLoser, 0, K_OVERALL),
    loserNewStriking: eloUpdate(loser.eloStriking, 1 - expWinnerStr, 0, kStr),
    loserNewGrappling: eloUpdate(loser.eloGrappling, 1 - expWinnerGrp, 0, kGrp),
  };
}

export async function applyEloUpdate(
  winnerId: string,
  loserId: string,
  method: FightMethod,
  fightDate: string,
): Promise<void> {
  const winner = getFighter(winnerId);
  const loser = getFighter(loserId);

  if (!winner || !loser) {
    logger.warn({ winnerId, loserId }, 'Fighter not found for Elo update');
    return;
  }

  const result = computeEloUpdate(winner, loser, method);

  upsertFighter({
    ...winner,
    eloOverall: result.winnerNewOverall,
    eloStriking: result.winnerNewStriking,
    eloGrappling: result.winnerNewGrappling,
    lastFightDate: fightDate,
    daysSinceLastFight: 0,
  });

  upsertFighter({
    ...loser,
    eloOverall: result.loserNewOverall,
    eloStriking: result.loserNewStriking,
    eloGrappling: result.loserNewGrappling,
    lastFightDate: fightDate,
    daysSinceLastFight: 0,
  });

  logger.info(
    { winnerId, loserId, method, winnerElo: result.winnerNewOverall.toFixed(1), loserElo: result.loserNewOverall.toFixed(1) },
    'Elo updated'
  );
}

// ─── Initialization helpers ───────────────────────────────────────────────────

export function defaultElo(): { eloOverall: number; eloStriking: number; eloGrappling: number } {
  return { eloOverall: NEW_FIGHTER_ELO, eloStriking: NEW_FIGHTER_ELO, eloGrappling: NEW_FIGHTER_ELO };
}

// ─── Elo diff for feature engineering ────────────────────────────────────────

export function eloFeatures(fighterA: Fighter, fighterB: Fighter, asOfDate: Date = new Date()) {
  const a = applyInactivityDecay(fighterA, asOfDate);
  const b = applyInactivityDecay(fighterB, asOfDate);
  return {
    elo_diff: a.eloOverall - b.eloOverall,
    striking_elo_diff: a.eloStriking - b.eloStriking,
    grappling_elo_diff: a.eloGrappling - b.eloGrappling,
  };
}
