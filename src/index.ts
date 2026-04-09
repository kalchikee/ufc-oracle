// UFC Oracle v4.1 — Main Entry Point
// Entrypoint for GitHub Actions workflows.
//
// Modes:
//   --alert picks      → Monday fight card predictions (fight-week-predictions.yml)
//   --alert updated    → Thursday updated predictions (updated-predictions.yml)
//   --alert recap      → Sunday post-event recap (post-event-recap.yml)
//   --run scrape-fighters → Manual fighter DB refresh (fighter-db-update.yml)
//   --run scrape-card  → Manual fight card fetch

import 'dotenv/config';
import { logger } from './logger.js';
import { initDb, upsertFighter, upsertPrediction, getLatestPredictions, getYTDAccuracy, getPredictionsByEvent, logEventAccuracy, persistDb } from './db/database.js';
import { getNextUFCEvent, isFightWeek, fetchFightCard, fetchEventResults } from './scraper/fightCardFetcher.js';
import { scrapeAllFighters } from './scraper/ufcStatsScraper.js';
import { generatePredictions } from './pipeline/predictionRunner.js';
import { sendFightCardPredictions, sendPostEventRecap, type RecapResult } from './discord/discord.js';
import { applyEloUpdate } from './elo/eloSystem.js';
import type { FightMethod } from './types.js';

const args = process.argv.slice(2);

async function main(): Promise<void> {
  await initDb();
  logger.info({ args }, 'UFC Oracle starting');

  const alertIdx = args.indexOf('--alert');
  const runIdx = args.indexOf('--run');

  if (alertIdx >= 0) {
    const mode = args[alertIdx + 1];
    if (mode === 'picks') await runFightCardPredictions(false);
    else if (mode === 'updated') await runFightCardPredictions(true);
    else if (mode === 'recap') await runPostEventRecap();
    else logger.error({ mode }, 'Unknown alert mode');
  } else if (runIdx >= 0) {
    const mode = args[runIdx + 1];
    if (mode === 'scrape-fighters') await runFighterScrape();
    else if (mode === 'scrape-card') await runFightCardFetch();
    else logger.error({ mode }, 'Unknown run mode');
  } else {
    logger.info('No mode specified. Use --alert picks|updated|recap or --run scrape-fighters|scrape-card');
  }

  persistDb();
}

// ─── Fight card predictions (Monday / Thursday) ───────────────────────────────

async function runFightCardPredictions(isUpdate: boolean): Promise<void> {
  logger.info({ isUpdate }, 'Running fight card predictions');

  const nextEvent = await getNextUFCEvent();
  if (!nextEvent) {
    logger.info('No upcoming UFC event found');
    return;
  }

  if (!isFightWeek(nextEvent.eventDate)) {
    logger.info({ eventDate: nextEvent.eventDate }, 'Not fight week — skipping predictions');
    return;
  }

  const card = await fetchFightCard(nextEvent.eventUrl);
  if (!card) {
    logger.error('Failed to fetch fight card');
    return;
  }

  logger.info({ event: card.eventName, fights: card.fights.length }, 'Fight card fetched');

  const predictions = await generatePredictions(card);
  if (predictions.length === 0) {
    logger.warn('No predictions generated');
    return;
  }

  for (const pred of predictions) {
    upsertPrediction(pred);
  }

  const stats = getYTDAccuracy();
  const sent = await sendFightCardPredictions(predictions, stats, isUpdate);

  if (sent) {
    logger.info({ count: predictions.length, isUpdate }, 'Fight card predictions sent to Discord');
  }
}

// ─── Post-event recap (Sunday) ────────────────────────────────────────────────

async function runPostEventRecap(): Promise<void> {
  logger.info('Running post-event recap');

  // Get the most recently completed event (Saturday's event)
  const nextEvent = await getNextUFCEvent();
  if (!nextEvent) {
    logger.warn('No event found for recap');
    return;
  }

  const results = await fetchEventResults(nextEvent.eventUrl);
  if (results.length === 0) {
    logger.warn('No fight results found');
    return;
  }

  // Match results to stored predictions
  const predictions = getPredictionsByEvent(nextEvent.eventId);
  const recapResults: RecapResult[] = [];

  for (const result of results) {
    const pred = predictions.find(p =>
      (p.fighterAId === result.winnerId || p.fighterBId === result.winnerId) &&
      (p.fighterAId === result.loserId || p.fighterBId === result.loserId)
    );

    if (pred) {
      const correct = pred.predictedWinnerId === result.winnerId;
      const method = result.method as FightMethod;
      const methodCorrect = pred.predictedMethod === method;

      // Update prediction with result
      pred.actualWinnerId = result.winnerId;
      pred.actualMethod = method;
      pred.correct = correct;
      pred.methodCorrect = methodCorrect;
      upsertPrediction(pred);

      recapResults.push({
        prediction: pred,
        actualWinnerName: result.winnerName,
        method,
        round: result.round,
      });

      // Update Elo ratings
      const loserId = result.winnerId === result.fighterAId ? result.fighterBId : result.fighterAId;
      await applyEloUpdate(result.winnerId, loserId, method, nextEvent.eventDate);
    }
  }

  if (recapResults.length === 0) {
    logger.warn('No matching predictions for event results');
    return;
  }

  // Log accuracy
  const correct = recapResults.filter(r => r.prediction.correct).length;
  const total = recapResults.length;
  const hcResults = recapResults.filter(r => r.prediction.confidenceTier === 'high_conviction' || r.prediction.confidenceTier === 'lock');
  const hcCorrect = hcResults.filter(r => r.prediction.correct).length;
  const methodCorrect = recapResults.filter(r => r.prediction.methodCorrect).length;

  logEventAccuracy(
    nextEvent.eventId,
    predictions[0]?.eventName ?? 'Unknown',
    nextEvent.eventDate,
    correct, total, hcCorrect, hcResults.length, methodCorrect, recapResults.length
  );

  const stats = getYTDAccuracy();
  await sendPostEventRecap(recapResults, stats);

  logger.info({ correct, total }, 'Post-event recap complete');
}

// ─── Fighter scrape ───────────────────────────────────────────────────────────

async function runFighterScrape(): Promise<void> {
  logger.info('Starting fighter database scrape from UFCStats.com');
  const count = await scrapeAllFighters((fighter) => {
    upsertFighter(fighter);
  });
  logger.info({ count }, 'Fighter scrape complete');
}

// ─── Fight card fetch (test / manual) ────────────────────────────────────────

async function runFightCardFetch(): Promise<void> {
  const nextEvent = await getNextUFCEvent();
  if (!nextEvent) {
    logger.info('No upcoming event found');
    return;
  }
  const card = await fetchFightCard(nextEvent.eventUrl);
  if (card) {
    logger.info({ event: card.eventName, fights: card.fights.length, location: card.location }, 'Fight card fetched');
    for (const fight of card.fights) {
      logger.info({ a: fight.fighterA, b: fight.fighterB, pos: fight.cardPosition }, 'Bout');
    }
  }
}

main().catch(err => {
  logger.error({ err }, 'Fatal error');
  process.exit(1);
});
