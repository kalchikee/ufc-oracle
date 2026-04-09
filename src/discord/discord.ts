// UFC Oracle v4.1 — Discord Webhook Alerts
// Message 1: Monday fight card predictions embed (red #D20A11)
// Message 2: Thursday updated predictions
// Message 3: Sunday post-event recap embed

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getConfidenceTier } from '../pipeline/predictionRunner.js';
import type { Prediction, AccuracyStats, FightMethod } from '../types.js';
import { styleLabel, matchupDescription } from '../style-model/styleClassifier.js';

// ─── Colors ───────────────────────────────────────────────────────────────────

const COLORS = {
  ufc_red: 0xD20A11,        // UFC red — fight card predictions
  gold_win: 0xC4A43C,       // gold — winning recap night
  recap_loss: 0xD20A11,     // red — losing recap night
  updated: 0xE67E22,        // orange — Thursday update
} as const;

// ─── Discord types ────────────────────────────────────────────────────────────

interface DiscordField { name: string; value: string; inline?: boolean; }
interface DiscordEmbed {
  title?: string; description?: string; color?: number;
  fields?: DiscordField[]; footer?: { text: string }; timestamp?: string;
}
interface DiscordPayload { content?: string; embeds: DiscordEmbed[]; }

// ─── Webhook sender ───────────────────────────────────────────────────────────

async function sendWebhook(payload: DiscordPayload): Promise<boolean> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    logger.warn('DISCORD_WEBHOOK_URL not set — skipping Discord alert');
    return false;
  }
  try {
    const resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });
    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Discord webhook error');
      return false;
    }
    logger.info('Discord alert sent');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send Discord webhook');
    return false;
  }
}

// ─── Formatters ───────────────────────────────────────────────────────────────

function pct(p: number): string {
  return (p * 100).toFixed(1) + '%';
}

function confidenceEmoji(tier: Prediction['confidenceTier']): string {
  switch (tier) {
    case 'lock': return '🚀';
    case 'high_conviction': return '🟢';
    case 'strong': return '✅';
    case 'lean': return '📌';
    default: return '🪙';
  }
}

function methodLine(p: Prediction): string {
  return `KO ${pct(p.koProb)} · Sub ${pct(p.submissionProb)} · Dec ${pct(p.decisionProb)}`;
}

function edgeLine(p: Prediction): string {
  if (!p.edge || !p.edgeCategory) return '';
  if (p.edgeCategory === 'none') return '';
  if (p.edgeCategory === 'extreme') return `  🚀🚀 +${pct(Math.abs(p.edge))} vs Vegas`;
  if (p.edgeCategory === 'priority') return `  🚀 +${pct(Math.abs(p.edge))} vs Vegas`;
  return `  💡 +${pct(Math.abs(p.edge))} vs Vegas`;
}

function winnerLine(p: Prediction): string {
  const winProb = p.predictedWinnerId === p.fighterAId ? p.fighterAWinProb : p.fighterBWinProb;
  const emoji = confidenceEmoji(p.confidenceTier);
  return `${emoji} **${p.predictedWinnerName}** — ${pct(winProb)}${edgeLine(p)}`;
}

function accuracyFooter(stats: AccuracyStats): string {
  const ytd = stats.ytdTotal > 0
    ? `📊 YTD: ${stats.ytdCorrect}-${stats.ytdTotal - stats.ytdCorrect} (${pct(stats.ytdAccuracy)})`
    : '📊 YTD: No results yet';
  const hc = stats.highConvTotal > 0
    ? ` · 🟢 65%+: ${stats.highConvCorrect}/${stats.highConvTotal} (${pct(stats.highConvAccuracy)})`
    : '';
  const method = stats.methodTotal > 0
    ? ` · Method: ${pct(stats.methodAccuracy)}`
    : '';
  return ytd + hc + method;
}

// ─── MESSAGE 1: Fight Card Predictions (Monday / Thursday) ───────────────────

export async function sendFightCardPredictions(
  predictions: Prediction[],
  stats: AccuracyStats,
  isUpdate = false,
): Promise<boolean> {
  if (predictions.length === 0) {
    logger.warn('No predictions to send');
    return false;
  }

  const sorted = [...predictions].sort((a, b) => {
    // Main event first, then by confidence descending
    if (a.isMainEvent && !b.isMainEvent) return -1;
    if (!a.isMainEvent && b.isMainEvent) return 1;
    if (a.cardPosition === 'co_main' && b.cardPosition !== 'co_main') return -1;
    if (a.cardPosition !== 'co_main' && b.cardPosition === 'co_main') return 1;
    const confA = Math.max(a.fighterAWinProb, a.fighterBWinProb);
    const confB = Math.max(b.fighterAWinProb, b.fighterBWinProb);
    return confB - confA;
  });

  const eventName = sorted[0].eventName;
  const eventDate = sorted[0].eventDate;
  const mainEvent = sorted.find(p => p.isMainEvent);
  const highConv = sorted.filter(p => p.confidenceTier === 'high_conviction' || p.confidenceTier === 'lock');
  const valueBets = sorted.filter(p => p.edgeCategory === 'value' || p.edgeCategory === 'priority' || p.edgeCategory === 'extreme');

  const fields: DiscordField[] = [];

  // Main event — detailed breakdown
  if (mainEvent) {
    const winProb = mainEvent.predictedWinnerId === mainEvent.fighterAId
      ? mainEvent.fighterAWinProb : mainEvent.fighterBWinProb;
    const loserName = mainEvent.predictedWinnerId === mainEvent.fighterAId
      ? mainEvent.fighterBName : mainEvent.fighterAName;
    const loserProb = 1 - winProb;

    fields.push({
      name: `🥊 MAIN EVENT · ${mainEvent.fighterAName} vs ${mainEvent.fighterBName}`,
      value: [
        `**Pick:** ${confidenceEmoji(mainEvent.confidenceTier)} ${mainEvent.predictedWinnerName} (${pct(winProb)}) · ${loserName} (${pct(loserProb)})`,
        `**Method:** ${methodLine(mainEvent)}`,
        `**Predicted:** ${mainEvent.predictedWinnerName} by ${mainEvent.predictedMethod}`,
        mainEvent.edge ? `**Edge:** ${pct(Math.abs(mainEvent.edge))} vs Vegas line` : '',
        mainEvent.isTitleFight ? '🏆 Title Fight' : '',
        `${mainEvent.scheduledRounds} rounds · ${mainEvent.weightClass}`,
      ].filter(Boolean).join('\n'),
      inline: false,
    });
  }

  // Co-main and main card — individual fields
  const mainCard = sorted.filter(p => !p.isMainEvent && (p.cardPosition === 'co_main' || p.cardPosition === 'main_card'));
  for (const p of mainCard) {
    const winProb = p.predictedWinnerId === p.fighterAId ? p.fighterAWinProb : p.fighterBWinProb;
    fields.push({
      name: `${cardPositionLabel(p.cardPosition)} ${p.fighterAName} vs ${p.fighterBName}`,
      value: [
        winnerLine(p),
        `Method: ${methodLine(p)}`,
      ].join('\n'),
      inline: true,
    });
  }

  // Prelims — condensed
  const prelims = sorted.filter(p => p.cardPosition === 'prelim' || p.cardPosition === 'early_prelim');
  if (prelims.length > 0) {
    const prelimLines = prelims
      .filter(p => p.confidenceTier !== 'coin_flip')
      .map(p => {
        const winProb = p.predictedWinnerId === p.fighterAId ? p.fighterAWinProb : p.fighterBWinProb;
        return `${confidenceEmoji(p.confidenceTier)} **${p.predictedWinnerName}** (${pct(winProb)}) · ${p.fighterAName} vs ${p.fighterBName}`;
      });

    if (prelimLines.length > 0) {
      fields.push({
        name: '📺 Prelims — Notable Picks',
        value: prelimLines.slice(0, 8).join('\n') || 'No strong picks',
        inline: false,
      });
    }
  }

  // YTD accuracy field
  fields.push({
    name: '🏆 Running Record',
    value: accuracyFooter(stats),
    inline: false,
  });

  const lockCount = sorted.filter(p => p.confidenceTier === 'lock').length;
  const titlePrefix = isUpdate ? '🔄 UPDATED: ' : '';

  const embed: DiscordEmbed = {
    title: `🥊 ${titlePrefix}${eventName} — Fight Card Predictions`,
    description: [
      `${sorted.length} fights · ${eventDate}`,
      highConv.length > 0 ? `🟢 ${highConv.length} high-conviction pick${highConv.length > 1 ? 's' : ''}` : '',
      lockCount > 0 ? `🚀 ${lockCount} lock${lockCount > 1 ? 's' : ''}` : '',
      valueBets.length > 0 ? `💡 ${valueBets.length} value bet${valueBets.length > 1 ? 's' : ''}` : '',
    ].filter(Boolean).join(' · '),
    color: isUpdate ? COLORS.updated : COLORS.ufc_red,
    fields: fields.slice(0, 25),
    footer: {
      text: '🚀 Lock · 🟢 High Conviction · ✅ Strong · 📌 Lean · 🪙 Coin Flip · UFC Oracle v4.1',
    },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── MESSAGE 2: Post-Event Recap (Sunday) ────────────────────────────────────

export interface RecapResult {
  prediction: Prediction;
  actualWinnerName: string;
  method: FightMethod;
  round: number;
}

export async function sendPostEventRecap(
  results: RecapResult[],
  stats: AccuracyStats,
): Promise<boolean> {
  if (results.length === 0) {
    logger.warn('No results for recap');
    return false;
  }

  const correct = results.filter(r => r.prediction.correct).length;
  const total = results.length;
  const accPct = total > 0 ? correct / total : 0;

  const eventName = results[0].prediction.eventName;
  const isWinningNight = accPct >= 0.55;

  // Per-fight lines
  const fightLines = results.map(r => {
    const ok = r.prediction.correct ? '✅' : '❌';
    const hc = (r.prediction.confidenceTier === 'high_conviction' || r.prediction.confidenceTier === 'lock') ? ' ⭐' : '';
    const methodOk = r.prediction.methodCorrect ? '✓' : '';
    return `${ok}${hc} **${r.actualWinnerName}** def. ${r.prediction.fighterAName === r.actualWinnerName ? r.prediction.fighterBName : r.prediction.fighterAName} by ${r.method} R${r.round} ${methodOk}`;
  });

  const hcResults = results.filter(r => r.prediction.confidenceTier === 'high_conviction' || r.prediction.confidenceTier === 'lock');
  const hcCorrect = hcResults.filter(r => r.prediction.correct).length;

  const eventAccLine = `**This Event: ${correct}/${total} (${pct(accPct)})**`;
  const hcLine = hcResults.length > 0
    ? `⭐ High Conviction: ${hcCorrect}/${hcResults.length} (${pct(hcCorrect / hcResults.length)})`
    : '';

  const fields: DiscordField[] = [
    {
      name: '📈 Event Summary',
      value: [
        eventAccLine,
        hcLine,
        `📊 YTD: ${stats.ytdCorrect}-${stats.ytdTotal - stats.ytdCorrect} (${pct(stats.ytdAccuracy)})`,
        stats.highConvTotal > 0
          ? `🟢 YTD High Conv: ${stats.highConvCorrect}/${stats.highConvTotal} (${pct(stats.highConvAccuracy)})`
          : '',
        stats.methodTotal > 0
          ? `🎯 Method Accuracy: ${pct(stats.methodAccuracy)}`
          : '',
        `Brier: ${stats.brierScore.toFixed(4)}`,
      ].filter(Boolean).join('\n'),
      inline: false,
    },
    {
      name: '🎯 Results',
      value: fightLines.slice(0, 1024 / 60).join('\n').slice(0, 1020) || 'No results.',
      inline: false,
    },
  ];

  // Event-by-event trend (last 5)
  if (stats.eventRecord.length > 0) {
    const trend = stats.eventRecord.slice(0, 5).map(e =>
      `${e.correct >= e.total * 0.6 ? '🟢' : e.correct >= e.total * 0.5 ? '🟡' : '🔴'} ${e.eventName.replace('UFC ', '').substring(0, 20)}: ${e.correct}/${e.total}`
    ).join('\n');
    fields.push({ name: '📅 Recent Events', value: trend, inline: false });
  }

  return sendWebhook({
    embeds: [{
      title: `📊 ${eventName} — Recap`,
      color: isWinningNight ? COLORS.gold_win : COLORS.recap_loss,
      fields,
      footer: { text: '✅ Correct · ❌ Wrong · ⭐ High Conv · ✓ Method correct · UFC Oracle v4.1' },
      timestamp: new Date().toISOString(),
    }],
  });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function cardPositionLabel(pos: string): string {
  if (pos === 'co_main') return '🥈 CO-MAIN ·';
  if (pos === 'main_card') return '📺 MAIN CARD ·';
  return '📋';
}
