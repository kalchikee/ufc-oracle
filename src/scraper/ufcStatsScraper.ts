// UFC Oracle v4.1 — UFCStats.com Scraper
// Scrapes fighter career stats and fight history from UFCStats.com (FightMetric)
// All stats stored as per-minute or per-15-min rates for normalization across fight durations.

import fetch from 'node-fetch';
import * as cheerio from 'cheerio';
import { logger } from '../logger.js';
import type { Fighter, FightMethod, WeightClass } from '../types.js';
import { defaultElo } from '../elo/eloSystem.js';
import { classifyStyle } from '../style-model/styleClassifier.js';

const BASE_URL = 'http://www.ufcstats.com';
const FIGHTER_LIST_URL = `${BASE_URL}/statistics/fighters`;
const DELAY_MS = 1200; // polite crawl delay

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchHtml(url: string): Promise<cheerio.CheerioAPI> {
  await sleep(DELAY_MS);
  const resp = await fetch(url, {
    headers: { 'User-Agent': 'UFC-Oracle-Research-Bot/4.1 (educational prediction model)' },
    signal: AbortSignal.timeout(15000),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  const html = await resp.text();
  return cheerio.load(html);
}

// ─── Fighter list ─────────────────────────────────────────────────────────────

export async function fetchFighterList(): Promise<string[]> {
  const urls: string[] = [];
  const alphabet = 'abcdefghijklmnopqrstuvwxyz'.split('');

  for (const letter of alphabet) {
    try {
      const $ = await fetchHtml(`${FIGHTER_LIST_URL}?char=${letter}&page=all`);
      $('a.b-link.b-link_style_black').each((_i, el) => {
        const href = $(el).attr('href');
        if (href?.includes('/fighter-details/')) urls.push(href);
      });
      logger.debug({ letter, count: urls.length }, 'Fighter list page scraped');
    } catch (err) {
      logger.warn({ letter, err }, 'Failed to fetch fighter list page');
    }
  }

  return [...new Set(urls)];
}

// ─── Fighter detail page ──────────────────────────────────────────────────────

export async function scrapeFighter(fighterUrl: string): Promise<Fighter | null> {
  try {
    const $ = await fetchHtml(fighterUrl);

    const name = $('span.b-content__title-highlight').text().trim();
    if (!name) return null;

    const fighterId = fighterUrl.split('/fighter-details/')[1] ?? fighterUrl;

    // Physical attributes
    const infoItems = $('li.b-list__box-list-item');
    const info: Record<string, string> = {};
    infoItems.each((_i, el) => {
      const text = $(el).text().trim();
      const [label, ...rest] = text.split(':');
      if (label && rest.length) info[label.trim().toLowerCase()] = rest.join(':').trim();
    });

    const height = parseHeight(info['height']);
    const reach = parseReach(info['reach']);
    const stance = info['stance'] || undefined;
    const dateOfBirth = parseDate(info['dob']);

    // Career statistics (from the stats table)
    const statsRow = $('p.b-list__box-list-item_type_block');
    const stats: Record<string, string> = {};
    statsRow.each((_i, el) => {
      const text = $(el).text().replace(/\s+/g, ' ').trim();
      const parts = text.split(' ');
      // Stats come in pairs: label value label value
      for (let i = 0; i < parts.length - 1; i += 2) {
        stats[parts[i].toLowerCase()] = parts[i + 1];
      }
    });

    // Try alternate stat extraction from the summary boxes
    const sigStrikesLandedPM = parseFloat($('[data-b-col="SLpM"]').first().text()) || 0;
    const sigStrikeAccuracy = parsePercent($('[data-b-col="Str. Acc."]').first().text());
    const sigStrikesAbsorbedPM = parseFloat($('[data-b-col="SApM"]').first().text()) || 0;
    const sigStrikeDefense = parsePercent($('[data-b-col="Str. Def"]').first().text());
    const takedownAvgPer15 = parseFloat($('[data-b-col="TD Avg."]').first().text()) || 0;
    const takedownAccuracy = parsePercent($('[data-b-col="TD Acc."]').first().text());
    const takedownDefense = parsePercent($('[data-b-col="TD Def."]').first().text());
    const submissionAvgPer15 = parseFloat($('[data-b-col="Sub. Avg."]').first().text()) || 0;

    // Fallback: parse from the two-column stat boxes
    const statBoxes = $('li.b-list__box-list-item').map((_i, el) => $(el).text().trim()).get();
    const parsed = parseStatBoxes(statBoxes);

    const slpm = sigStrikesLandedPM || parsed.slpm;
    const strAcc = sigStrikeAccuracy || parsed.strAcc;
    const sapm = sigStrikesAbsorbedPM || parsed.sapm;
    const strDef = sigStrikeDefense || parsed.strDef;
    const tdAvg = takedownAvgPer15 || parsed.tdAvg;
    const tdAcc = takedownAccuracy || parsed.tdAcc;
    const tdDef = takedownDefense || parsed.tdDef;
    const subAvg = submissionAvgPer15 || parsed.subAvg;

    // Win/loss record
    const record = $('span.b-content__title-record').text().trim();
    const { wins, losses, draws } = parseRecord(record);
    const winPct = (wins + losses) > 0 ? wins / (wins + losses) : 0;

    // Fight history for recent form, knockdowns, control time
    const { knockdownRate, controlTimePct, avgFightTime, ufcWins, ufcLosses,
      finishRate, decisionRate, recentWins, recentLosses, recentSigStrikesPM, winStreak } =
      await parseFightHistory($, wins + losses);

    const style = classifyStyle({ sigStrikesLandedPM: slpm, takedownAvgPer15: tdAvg, submissionAvgPer15: subAvg, sigStrikeDefense: strDef, sigStrikeAccuracy: strAcc });

    const fighter: Fighter = {
      fighterId,
      name,
      weightClass: inferWeightClass(info['weight'] || ''),
      height,
      reach,
      stance,
      dateOfBirth,
      sigStrikesLandedPM: slpm,
      sigStrikesAbsorbedPM: sapm,
      sigStrikeAccuracy: strAcc,
      sigStrikeDefense: strDef,
      takedownAvgPer15: tdAvg,
      takedownAccuracy: tdAcc,
      takedownDefense: tdDef,
      submissionAvgPer15: subAvg,
      knockdownRate,
      controlTimePct,
      avgFightTime,
      wins,
      losses,
      draws,
      winPct,
      ufcWins,
      ufcLosses,
      finishRate,
      decisionRate,
      style,
      ...defaultElo(),
      recentWins,
      recentLosses,
      recentSigStrikesPM,
      winStreak,
      updatedAt: new Date().toISOString(),
    };

    return fighter;
  } catch (err) {
    logger.warn({ fighterUrl, err }, 'Failed to scrape fighter');
    return null;
  }
}

// ─── Fight history parsing ────────────────────────────────────────────────────

async function parseFightHistory(
  $: cheerio.CheerioAPI,
  totalFights: number,
): Promise<{
  knockdownRate: number; controlTimePct: number; avgFightTime: number;
  ufcWins: number; ufcLosses: number; finishRate: number; decisionRate: number;
  recentWins: number; recentLosses: number; recentSigStrikesPM: number; winStreak: number;
}> {
  let totalKDs = 0, totalControlSecs = 0, totalFightSecs = 0;
  let ufcWins = 0, ufcLosses = 0, finishes = 0, decisions = 0;
  let recentWins = 0, recentLosses = 0, recentStrikes = 0, recentFights = 0;
  let streak = 0, streakBroken = false;

  const rows = $('tr.b-fight-details__table-row[data-link]');
  let rowCount = 0;

  rows.each((_i, row) => {
    rowCount++;
    const cells = $(row).find('td');
    const result = cells.eq(0).text().trim().toLowerCase();
    const method = cells.eq(7).text().trim();
    const timeStr = cells.eq(9).text().trim();
    const roundsStr = cells.eq(8).text().trim();

    const fightSecs = parseFightSecs(timeStr, parseInt(roundsStr) || 1);
    if (fightSecs > 0) totalFightSecs += fightSecs;

    if (result === 'win') {
      ufcWins++;
      if (method.includes('KO') || method.includes('TKO') || method.includes('Sub')) finishes++;
      else decisions++;
      if (!streakBroken) streak++;
    } else if (result === 'loss') {
      ufcLosses++;
      decisions++;
      if (!streakBroken) { streak = 0; streakBroken = true; }
    }

    if (rowCount <= 3) {
      recentFights++;
      if (result === 'win') recentWins++;
      else if (result === 'loss') recentLosses++;
    }
  });

  const avgFightTime = rowCount > 0 ? totalFightSecs / 60 / rowCount : 9;
  const knockdownRate = totalFightSecs > 0 ? (totalKDs / (totalFightSecs / 60 / 15)) : 0;
  const controlTimePct = 0; // would need per-fight data
  const finishRate = ufcWins > 0 ? finishes / ufcWins : 0;
  const decisionRate = (ufcWins + ufcLosses) > 0 ? decisions / (ufcWins + ufcLosses) : 0;
  const recentSigStrikesPM = 0; // would need per-fight strikes

  return {
    knockdownRate, controlTimePct, avgFightTime,
    ufcWins, ufcLosses, finishRate, decisionRate,
    recentWins, recentLosses, recentSigStrikesPM,
    winStreak: streak,
  };
}

// ─── Parse helpers ────────────────────────────────────────────────────────────

function parseStatBoxes(boxes: string[]): {
  slpm: number; strAcc: number; sapm: number; strDef: number;
  tdAvg: number; tdAcc: number; tdDef: number; subAvg: number;
} {
  const result = { slpm: 0, strAcc: 0, sapm: 0, strDef: 0, tdAvg: 0, tdAcc: 0, tdDef: 0, subAvg: 0 };
  for (const box of boxes) {
    const lower = box.toLowerCase();
    if (lower.includes('slpm')) result.slpm = parseFloat(box.split(':').pop()?.trim() || '0') || 0;
    else if (lower.includes('str. acc')) result.strAcc = parsePercent(box.split(':').pop()?.trim() || '0%');
    else if (lower.includes('sapm')) result.sapm = parseFloat(box.split(':').pop()?.trim() || '0') || 0;
    else if (lower.includes('str. def')) result.strDef = parsePercent(box.split(':').pop()?.trim() || '0%');
    else if (lower.includes('td avg')) result.tdAvg = parseFloat(box.split(':').pop()?.trim() || '0') || 0;
    else if (lower.includes('td acc')) result.tdAcc = parsePercent(box.split(':').pop()?.trim() || '0%');
    else if (lower.includes('td def')) result.tdDef = parsePercent(box.split(':').pop()?.trim() || '0%');
    else if (lower.includes('sub. avg')) result.subAvg = parseFloat(box.split(':').pop()?.trim() || '0') || 0;
  }
  return result;
}

function parseHeight(s?: string): number | undefined {
  if (!s) return undefined;
  const m = s.match(/(\d+)'\s*(\d+)"/);
  if (!m) return undefined;
  return parseInt(m[1]) * 12 + parseInt(m[2]);
}

function parseReach(s?: string): number | undefined {
  if (!s || s === '--') return undefined;
  const m = s.match(/(\d+(?:\.\d+)?)/);
  return m ? parseFloat(m[1]) : undefined;
}

function parseDate(s?: string): string | undefined {
  if (!s || s === '--') return undefined;
  try {
    const d = new Date(s);
    return isNaN(d.getTime()) ? undefined : d.toISOString().split('T')[0];
  } catch { return undefined; }
}

function parsePercent(s: string): number {
  const m = s.match(/(\d+(?:\.\d+)?)\s*%/);
  if (m) return parseFloat(m[1]) / 100;
  const n = parseFloat(s);
  return isNaN(n) ? 0 : n > 1 ? n / 100 : n;
}

function parseRecord(s: string): { wins: number; losses: number; draws: number } {
  const m = s.match(/(\d+)-(\d+)-(\d+)/);
  if (!m) return { wins: 0, losses: 0, draws: 0 };
  return { wins: parseInt(m[1]), losses: parseInt(m[2]), draws: parseInt(m[3]) };
}

function parseFightSecs(time: string, round: number): number {
  const m = time.match(/(\d+):(\d+)/);
  if (!m) return 0;
  const roundSecs = (round - 1) * 300;
  return roundSecs + parseInt(m[1]) * 60 + parseInt(m[2]);
}

function inferWeightClass(weightStr: string): WeightClass {
  const w = weightStr.toLowerCase();
  if (w.includes('strawweight') && w.includes('women')) return 'WomenStrawweight';
  if (w.includes('flyweight') && w.includes('women')) return 'WomenFlyweight';
  if (w.includes('bantamweight') && w.includes('women')) return 'WomenBantamweight';
  if (w.includes('featherweight') && w.includes('women')) return 'WomenFeatherweight';
  if (w.includes('strawweight')) return 'Strawweight';
  if (w.includes('flyweight')) return 'Flyweight';
  if (w.includes('bantamweight')) return 'Bantamweight';
  if (w.includes('featherweight')) return 'Featherweight';
  if (w.includes('lightweight')) return 'Lightweight';
  if (w.includes('welterweight')) return 'Welterweight';
  if (w.includes('middleweight')) return 'Middleweight';
  if (w.includes('light heavyweight')) return 'LightHeavyweight';
  if (w.includes('heavyweight')) return 'Heavyweight';
  return 'Welterweight'; // fallback
}

// ─── Batch scrape ─────────────────────────────────────────────────────────────

export async function scrapeAllFighters(
  onFighter: (f: Fighter) => void,
  limit?: number,
): Promise<number> {
  const urls = await fetchFighterList();
  const toScrape = limit ? urls.slice(0, limit) : urls;
  let count = 0;

  for (const url of toScrape) {
    const fighter = await scrapeFighter(url);
    if (fighter) {
      onFighter(fighter);
      count++;
      if (count % 50 === 0) logger.info({ count, total: toScrape.length }, 'Scraping progress');
    }
  }

  logger.info({ count }, 'Scraping complete');
  return count;
}
