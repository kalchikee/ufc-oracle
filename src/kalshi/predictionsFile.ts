// Writes today's UFC predictions to predictions/YYYY-MM-DD.json.
// The kalshi-safety service fetches this file via GitHub raw URL to
// decide which picks to back on Kalshi.
//
// UFC convention:
//   home = fighter A (red corner)
//   away = fighter B (blue corner)
//   pickedSide = 'home' if fighter A is favored, 'away' if fighter B is favored.

import { mkdirSync, writeFileSync } from 'fs';
import { resolve } from 'path';
import type { Prediction } from '../types.js';

interface Pick {
  gameId: string;
  home: string;
  away: string;
  startTime?: string;
  pickedTeam: string;
  pickedSide: 'home' | 'away';
  modelProb: number;
  vegasProb?: number;
  edge?: number;
  confidenceTier?: string;
  extra?: Record<string, unknown>;
}

interface PredictionsFile {
  sport: 'UFC';
  date: string;
  generatedAt: string;
  picks: Pick[];
}

const MIN_PROB = parseFloat(process.env.KALSHI_MIN_PROB ?? '0.58');

export function writePredictionsFile(date: string, predictions: Prediction[]): string {
  const dir = resolve(process.cwd(), 'predictions');
  mkdirSync(dir, { recursive: true });
  const path = resolve(dir, `${date}.json`);

  const picks: Pick[] = [];
  for (const p of predictions) {
    const aProb = p.fighterAWinProb;
    const bProb = p.fighterBWinProb;
    const favoredA = aProb >= bProb;
    const modelProb = Math.max(aProb, bProb);
    if (modelProb < MIN_PROB) continue;

    picks.push({
      gameId: `ufc-${date}-${slug(p.fighterAName)}-${slug(p.fighterBName)}`,
      home: p.fighterAName,
      away: p.fighterBName,
      pickedTeam: favoredA ? p.fighterAName : p.fighterBName,
      pickedSide: favoredA ? 'home' : 'away',
      modelProb,
      vegasProb: p.vegasWinnerProb,
      edge: p.edge,
      confidenceTier: p.confidenceTier,
      extra: {
        fightId: p.fightId,
        eventId: p.eventId,
        eventName: p.eventName,
        eventDate: p.eventDate,
        weightClass: p.weightClass,
        cardPosition: p.cardPosition,
        isMainEvent: p.isMainEvent,
        isTitleFight: p.isTitleFight,
        scheduledRounds: p.scheduledRounds,
        predictedMethod: p.predictedMethod,
        koProb: p.koProb,
        submissionProb: p.submissionProb,
        decisionProb: p.decisionProb,
      },
    });
  }

  const file: PredictionsFile = {
    sport: 'UFC',
    date,
    generatedAt: new Date().toISOString(),
    picks,
  };
  writeFileSync(path, JSON.stringify(file, null, 2));
  return path;
}

function slug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}
