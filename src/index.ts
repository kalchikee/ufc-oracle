// UFC Oracle v4.1 — Main Entry Point
// Entrypoint for GitHub Actions workflows.
//
// Modes:
//   --run save-picks   → Monday: generate + save predictions to DB (no Discord)
//   --alert updated    → Thursday: re-generate + send Discord embed
//   --alert recap      → Sunday: score results, update Elo, send Discord recap
//   --alert picks      → Send predictions Discord embed immediately (manual/testing)
//   --run scrape-fighters → Manual fighter DB refresh (fighter-db-update.yml)
//   --run scrape-card  → Manual fight card fetch

import 'dotenv/config';
import { logger } from './logger.js';
import { initDb, upsertFighter, upsertPrediction, getLatestPredictions, getYTDAccuracy, getPredictionsByEvent, logEventAccuracy, persistDb } from './db/database.js';
import { getNextUFCEvent, isFightWeek, isFightDay, fetchFightCard, fetchEventResults } from './scraper/fightCardFetcher.js';
import { enrichWithOdds } from './scraper/oddsScraper.js';
import { scrapeAllFighters } from './scraper/ufcStatsScraper.js';
import { generatePredictions } from './pipeline/predictionRunner.js';
import { sendFightCardPredictions, sendPostEventRecap, type RecapResult } from './discord/discord.js';
import { applyEloUpdate } from './elo/eloSystem.js';
import type { FightMethod, Prediction } from './types.js';

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
    else if (mode === 'fight-day') await runFightDayPredictions();
    else if (mode === 'recap') await runPostEventRecap();
    else logger.error({ mode }, 'Unknown alert mode');
  } else if (runIdx >= 0) {
    const mode = args[runIdx + 1];
    if (mode === 'scrape-fighters') await runFighterScrape();
    else if (mode === 'scrape-card') await runFightCardFetch();
    else if (mode === 'save-picks') await runSavePicks();
    else logger.error({ mode }, 'Unknown run mode');
  } else {
    logger.info('No mode specified. Use --alert picks|updated|recap or --run scrape-fighters|scrape-card|save-picks');
  }

  persistDb();
}

// ─── Build + save predictions (shared logic) ─────────────────────────────────

async function buildAndSavePredictions(): Promise<{ predictions: Prediction[]; eventName: string } | null> {
  const nextEvent = await getNextUFCEvent();
  if (!nextEvent) {
    logger.info('No upcoming UFC event found');
    return null;
  }

  if (!isFightWeek(nextEvent.eventDate)) {
    logger.info({ eventDate: nextEvent.eventDate }, 'Not fight week — skipping predictions');
    return null;
  }

  const card = await fetchFightCard(nextEvent.eventUrl);
  if (!card) {
    logger.error('Failed to fetch fight card');
    return null;
  }

  logger.info({ event: card.eventName, fights: card.fights.length }, 'Fight card fetched');

  // Enrich bouts with live moneylines (no-op if ODDS_API_KEY not set)
  await enrichWithOdds(card.fights);

  const predictions = await generatePredictions(card);
  if (predictions.length === 0) {
    logger.warn('No predictions generated');
    return null;
  }

  for (const pred of predictions) {
    upsertPrediction(pred);
  }

  logger.info({ count: predictions.length, event: card.eventName }, 'Predictions saved to DB');
  return { predictions, eventName: card.eventName };
}

// ─── Monday: generate + save only (no Discord) ───────────────────────────────

async function runSavePicks(): Promise<void> {
  logger.info('Monday pipeline: generating predictions (saving to DB, no Discord alert)');
  await buildAndSavePredictions();
}

// ─── Thursday: regenerate + send Discord embed ────────────────────────────────

async function runFightCardPredictions(isUpdate: boolean): Promise<void> {
  logger.info({ isUpdate }, 'Running fight card predictions + Discord alert');

  const result = await buildAndSavePredictions();
  if (!result) return;

  const stats = getYTDAccuracy();
  const sent = await sendFightCardPredictions(result.predictions, stats, isUpdate);

  if (sent) {
    logger.info({ count: result.predictions.length, isUpdate }, 'Fight card predictions sent to Discord');
  }
}

// ─── Fight day: send Discord ONLY if event is today ───────────────────────────
// Used by the daily fight-night.yml workflow. Works regardless of what day
// of the week the event falls on (Sat, Wed international, Sun UK cards, etc).

async function runFightDayPredictions(): Promise<void> {
  const nextEvent = await getNextUFCEvent();
  if (!nextEvent) {
    logger.info('No upcoming UFC event found');
    return;
  }
  if (!isFightDay(nextEvent.eventDate)) {
    logger.info({ eventDate: nextEvent.eventDate }, 'Not fight day today — skipping Discord alert');
    return;
  }

  logger.info({ eventDate: nextEvent.eventDate }, 'Fight day detected — sending Discord predictions');
  const result = await buildAndSavePredictions();
  if (!result) return;

  const stats = getYTDAccuracy();
  const sent = await sendFightCardPredictions(result.predictions, stats, true);
  if (sent) {
    logger.info({ count: result.predictions.length }, 'Fight-day predictions sent to Discord');
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
    // FightResult has winnerId but NOT loserId. Derive loserId from the
    // two fighter IDs in the result. Previous code referenced result.loserId
    // which was always undefined → pred was always undefined → recap always
    // reported 0 matching predictions.
    const loserId = result.winnerId === result.fighterAId ? result.fighterBId : result.fighterAId;
    const pred = predictions.find(p =>
      (p.fighterAId === result.winnerId || p.fighterBId === result.winnerId) &&
      (p.fighterAId === loserId || p.fighterBId === loserId)
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
