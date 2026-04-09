// UFC Oracle v4.1 — Prediction Runner
// Loads ML model weights and generates predictions for an upcoming fight card.

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type {
  Fighter, FightCard, FightCardBout, Prediction,
  ModelWeights, MethodModelWeights, CalibrationWeights,
  ConfidenceTier, EdgeCategory, FightMethod,
} from '../types.js';
import { buildFeatureVector, normalizeFeatures } from './featureEngineering.js';
import { getFighterByName, getFighter } from '../db/database.js';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const MODEL_DIR = resolve(__dirname, '../../model');
const MODEL_VERSION = '4.1.0';

// ─── Load models ──────────────────────────────────────────────────────────────

function loadModel<T>(filename: string): T | null {
  const path = resolve(MODEL_DIR, filename);
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, 'utf-8')) as T;
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function softmax(xs: number[]): number[] {
  const maxX = Math.max(...xs);
  const exps = xs.map(x => Math.exp(x - maxX));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function dotProduct(features: Record<string, number>, coefficients: Record<string, number>): number {
  return Object.entries(coefficients).reduce((sum, [k, v]) => sum + (features[k] ?? 0) * v, 0);
}

// ─── Winner prediction ────────────────────────────────────────────────────────

function predictWinner(
  features: Record<string, number>,
  model: ModelWeights,
  calibration: CalibrationWeights | null,
): number {
  const rawScore = model.intercept + dotProduct(features, model.coefficients);
  const rawProb = sigmoid(rawScore);

  if (calibration) {
    // Platt scaling: sigmoid(a * raw_score + b)
    return sigmoid(calibration.a * rawScore + calibration.b);
  }

  return rawProb;
}

// ─── Method prediction ────────────────────────────────────────────────────────

function predictMethod(
  features: Record<string, number>,
  winnerProb: number,
  model: MethodModelWeights,
): { koProb: number; submissionProb: number; decisionProb: number; otherProb: number } {
  const featuresWithWinner = { ...features, predicted_winner_prob: winnerProb };
  const classes = model.classes; // ['KO/TKO', 'Submission', 'Decision']

  const rawScores = classes.map((_, i) => {
    const intercept = model.intercept[i] ?? 0;
    const coefs = Object.entries(model.coefficients).reduce((sum, [k, v]) => {
      return sum + (featuresWithWinner[k] ?? 0) * (Array.isArray(v) ? v[i] ?? 0 : v);
    }, 0);
    return intercept + coefs;
  });

  const probs = softmax(rawScores);

  const koIdx = classes.indexOf('KO/TKO');
  const subIdx = classes.indexOf('Submission');
  const decIdx = classes.indexOf('Decision');

  return {
    koProb: koIdx >= 0 ? probs[koIdx] : 0.28,
    submissionProb: subIdx >= 0 ? probs[subIdx] : 0.12,
    decisionProb: decIdx >= 0 ? probs[decIdx] : 0.55,
    otherProb: 0.05,
  };
}

// ─── Fallback (heuristic) predictions ────────────────────────────────────────

function heuristicWinnerProb(fv: Record<string, number>): number {
  // Weighted combination of key features as a simple heuristic
  const score = (
    fv.elo_diff * 0.003 +
    fv.striking_elo_diff * 0.001 +
    fv.grappling_elo_diff * 0.001 +
    fv.sig_strikes_landed_pm_diff * 0.04 +
    fv.takedown_avg_diff * 0.03 +
    fv.win_pct_diff * 0.5 +
    fv.ufc_win_pct_diff * 0.3 +
    fv.win_streak_diff * 0.05
  );
  return Math.min(0.95, Math.max(0.05, sigmoid(score)));
}

function heuristicMethodProbs(
  fighterA: Fighter,
  fighterB: Fighter,
  winnerProb: number,
): { koProb: number; submissionProb: number; decisionProb: number; otherProb: number } {
  // Base rates adjusted by fighter profiles
  const winnerIsA = winnerProb >= 0.5;
  const winner = winnerIsA ? fighterA : fighterB;
  const loser = winnerIsA ? fighterB : fighterA;

  let koBase = 0.28;
  let subBase = 0.12;
  const decBase = 0.55;

  // Adjust by finish rates
  koBase += winner.finishRate * 0.1;
  subBase += winner.submissionAvgPer15 * 0.05;
  koBase += loser.sigStrikesAbsorbedPM * 0.02;

  // Normalize
  const total = koBase + subBase + decBase + 0.05;
  return {
    koProb: koBase / total,
    submissionProb: subBase / total,
    decisionProb: decBase / total,
    otherProb: 0.05 / total,
  };
}

// ─── Confidence tier ──────────────────────────────────────────────────────────

export function getConfidenceTier(prob: number): ConfidenceTier {
  const p = Math.max(prob, 1 - prob);
  if (p >= 0.72) return 'lock';
  if (p >= 0.65) return 'high_conviction';
  if (p >= 0.60) return 'strong';
  if (p >= 0.55) return 'lean';
  return 'coin_flip';
}

// ─── Market edge ──────────────────────────────────────────────────────────────

function moneylineToProb(ml: number): number {
  if (ml > 0) return 100 / (ml + 100);
  return Math.abs(ml) / (Math.abs(ml) + 100);
}

function removeVig(probA: number, probB: number): { cleanA: number; cleanB: number } {
  const total = probA + probB;
  return { cleanA: probA / total, cleanB: probB / total };
}

export function computeEdge(
  modelProb: number,
  moneylineA?: number,
  moneylineB?: number,
): { vegasProb: number; edge: number; edgeCategory: EdgeCategory } | null {
  if (moneylineA === undefined || moneylineB === undefined) return null;

  const rawA = moneylineToProb(moneylineA);
  const rawB = moneylineToProb(moneylineB);
  const { cleanA } = removeVig(rawA, rawB);

  const edge = modelProb - cleanA;
  let edgeCategory: EdgeCategory;

  if (Math.abs(edge) < 0.05) edgeCategory = 'none';
  else if (Math.abs(edge) < 0.10) edgeCategory = 'value';
  else if (Math.abs(edge) < 0.15) edgeCategory = 'priority';
  else edgeCategory = 'extreme';

  return { vegasProb: cleanA, edge, edgeCategory };
}

// ─── Main prediction function ─────────────────────────────────────────────────

export async function generatePredictions(
  card: FightCard,
): Promise<Prediction[]> {
  const winnerModel = loadModel<ModelWeights>('winner_model.json');
  const methodModel = loadModel<MethodModelWeights>('method_model.json');
  const calibration = loadModel<CalibrationWeights>('calibration.json');

  const predictions: Prediction[] = [];

  for (const bout of card.fights) {
    try {
      const fighterA = await resolveFighter(bout.fighterAId, bout.fighterA);
      const fighterB = await resolveFighter(bout.fighterBId, bout.fighterB);

      if (!fighterA || !fighterB) {
        logger.warn({ boutA: bout.fighterA, boutB: bout.fighterB }, 'Fighter not found in DB, skipping');
        continue;
      }

      const featureVector = buildFeatureVector(fighterA, fighterB, bout, card.location);
      const normalizedFeatures = normalizeFeatures(featureVector);

      let fighterAWinProb: number;
      if (winnerModel) {
        fighterAWinProb = predictWinner(normalizedFeatures, winnerModel, calibration);
      } else {
        fighterAWinProb = heuristicWinnerProb(normalizedFeatures);
      }

      const fighterBWinProb = 1 - fighterAWinProb;
      const predictedWinner = fighterAWinProb >= 0.5 ? fighterA : fighterB;

      let methodProbs: { koProb: number; submissionProb: number; decisionProb: number; otherProb: number };
      if (methodModel) {
        methodProbs = predictMethod(normalizedFeatures, fighterAWinProb, methodModel);
      } else {
        methodProbs = heuristicMethodProbs(fighterA, fighterB, fighterAWinProb);
      }

      const predictedMethod = determinePredictedMethod(methodProbs);
      const confidenceTier = getConfidenceTier(fighterAWinProb);

      const edgeResult = computeEdge(
        predictedWinner === fighterA ? fighterAWinProb : fighterBWinProb,
        bout.fighterAMoneyline,
        bout.fighterBMoneyline,
      );

      const prediction: Prediction = {
        predictionId: `${card.eventId}-${bout.fightId}-${Date.now()}`,
        fightId: bout.fightId,
        eventId: card.eventId,
        eventName: card.eventName,
        eventDate: card.eventDate,
        fighterAId: fighterA.fighterId,
        fighterBId: fighterB.fighterId,
        fighterAName: fighterA.name,
        fighterBName: fighterB.name,
        weightClass: bout.weightClass,
        cardPosition: bout.cardPosition,
        isMainEvent: bout.isMainEvent,
        isTitleFight: bout.isTitleFight,
        scheduledRounds: bout.scheduledRounds,
        featureVector,
        fighterAWinProb,
        fighterBWinProb,
        predictedWinnerId: predictedWinner.fighterId,
        predictedWinnerName: predictedWinner.name,
        confidenceTier,
        koProb: methodProbs.koProb,
        submissionProb: methodProbs.submissionProb,
        decisionProb: methodProbs.decisionProb,
        otherProb: methodProbs.otherProb,
        predictedMethod,
        fighterAMoneyline: bout.fighterAMoneyline,
        fighterBMoneyline: bout.fighterBMoneyline,
        vegasWinnerProb: edgeResult?.vegasProb,
        edge: edgeResult?.edge,
        edgeCategory: edgeResult?.edgeCategory,
        modelVersion: MODEL_VERSION,
        createdAt: new Date().toISOString(),
      };

      predictions.push(prediction);
      logger.info(
        { a: fighterA.name, b: fighterB.name, winner: predictedWinner.name, prob: fighterAWinProb.toFixed(3) },
        'Prediction generated'
      );
    } catch (err) {
      logger.error({ err, bout: bout.fighterA + ' vs ' + bout.fighterB }, 'Prediction failed');
    }
  }

  return predictions;
}

async function resolveFighter(id?: string, name?: string): Promise<Fighter | undefined> {
  if (id) {
    const f = getFighter(id);
    if (f) return f;
  }
  if (name) return getFighterByName(name) ?? undefined;
  return undefined;
}

function determinePredictedMethod(probs: {
  koProb: number; submissionProb: number; decisionProb: number; otherProb: number;
}): FightMethod {
  const { koProb, submissionProb, decisionProb } = probs;
  if (decisionProb >= koProb && decisionProb >= submissionProb) return 'Decision';
  if (koProb >= submissionProb) return 'KO/TKO';
  return 'Submission';
}
