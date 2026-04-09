// UFC Oracle v4.1 — Fight Card Fetcher
// Fetches the upcoming UFC event card from UFCStats.com and UFC.com
// Returns structured fight card data for prediction pipeline.

import fetch from 'node-fetch';
import * as cheerio from 'cheerio';
import { logger } from '../logger.js';
import type { FightCard, FightCardBout, WeightClass } from '../types.js';

const UFCSTATS_EVENTS_URL = 'http://www.ufcstats.com/statistics/events/upcoming';
const BASE_URL = 'http://www.ufcstats.com';
const DELAY_MS = 1200;

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

// ─── Check if UFC event this Saturday ────────────────────────────────────────

export async function getNextUFCEvent(): Promise<{ eventId: string; eventUrl: string; eventDate: string } | null> {
  try {
    const $ = await fetchHtml(UFCSTATS_EVENTS_URL);
    const firstRow = $('tr.b-statistics__table-row').first();
    const link = firstRow.find('a.b-link').first();
    const eventUrl = link.attr('href') || '';
    const eventName = link.text().trim();
    const dateText = firstRow.find('span.b-statistics__date').text().trim();

    if (!eventUrl) return null;

    const eventId = eventUrl.split('/event-details/')[1] ?? eventUrl;
    const eventDate = parseDateString(dateText);

    logger.info({ eventName, eventDate, eventUrl }, 'Next UFC event found');
    return { eventId, eventUrl, eventDate };
  } catch (err) {
    logger.error({ err }, 'Failed to fetch upcoming UFC events');
    return null;
  }
}

export function isFightWeek(eventDate: string): boolean {
  const now = new Date();
  const event = new Date(eventDate);
  const diffDays = (event.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);
  // Fight week = event is within the next 7 days
  return diffDays >= 0 && diffDays <= 7;
}

// ─── Fight card scraping ──────────────────────────────────────────────────────

export async function fetchFightCard(eventUrl: string): Promise<FightCard | null> {
  try {
    const $ = await fetchHtml(eventUrl);

    const eventName = $('h2.b-content__title-headline').text().trim() ||
                      $('span.b-content__title-highlight').text().trim();
    const eventId = eventUrl.split('/event-details/')[1] ?? eventUrl;

    // Date and location
    const listItems = $('li.b-list__box-list-item');
    let eventDate = '';
    let location = '';

    listItems.each((_i, el) => {
      const text = $(el).text().trim();
      if (text.startsWith('Date:')) eventDate = parseDateString(text.replace('Date:', '').trim());
      if (text.startsWith('Location:')) location = text.replace('Location:', '').trim();
    });

    // Fight rows
    const bouts: FightCardBout[] = [];
    let fightIndex = 0;

    const fightRows = $('tr.b-fight-details__table-row');
    fightRows.each((_i, row) => {
      const cells = $(row).find('td.b-fight-details__table-col');
      if (cells.length < 2) return;

      const fighterLinks = cells.eq(1).find('a');
      const fighterAName = fighterLinks.eq(0).text().trim();
      const fighterBName = fighterLinks.eq(1).text().trim();
      if (!fighterAName || !fighterBName) return;

      const fighterAUrl = fighterLinks.eq(0).attr('href') || '';
      const fighterBUrl = fighterLinks.eq(1).attr('href') || '';
      const fighterAId = fighterAUrl.split('/fighter-details/')[1];
      const fighterBId = fighterBUrl.split('/fighter-details/')[1];

      const weightText = cells.eq(6).text().trim();
      const weightClass = parseWeightClass(weightText);

      const methodText = cells.eq(7).text().trim();
      const isTitleFight = weightText.toLowerCase().includes('title') ||
                           $(row).find('img[alt="title"]').length > 0;

      const cardPosition = inferCardPosition(fightIndex, fightRows.length);
      const isMainEvent = fightIndex === 0;
      const scheduledRounds = isTitleFight || isMainEvent ? 5 : 3;

      bouts.push({
        fightId: `${eventId}-${fightIndex}`,
        fighterA: fighterAName,
        fighterB: fighterBName,
        fighterAId,
        fighterBId,
        weightClass,
        isMainEvent,
        isTitleFight,
        scheduledRounds,
        cardPosition,
      });

      fightIndex++;
    });

    if (bouts.length === 0) {
      logger.warn({ eventUrl }, 'No bouts found on fight card page');
      return null;
    }

    return {
      eventId,
      eventName,
      eventDate,
      location,
      venue: location.split(',')[0]?.trim() ?? location,
      fights: bouts,
    };
  } catch (err) {
    logger.error({ err, eventUrl }, 'Failed to fetch fight card');
    return null;
  }
}

// ─── Post-event results ───────────────────────────────────────────────────────

export interface FightResult {
  fightId: string;
  fighterAId: string;
  fighterBId: string;
  winnerId: string;
  winnerName: string;
  loserName: string;
  method: string;
  round: number;
  time: string;
}

export async function fetchEventResults(eventUrl: string): Promise<FightResult[]> {
  try {
    const $ = await fetchHtml(eventUrl);
    const eventId = eventUrl.split('/event-details/')[1] ?? eventUrl;
    const results: FightResult[] = [];
    let fightIndex = 0;

    const fightRows = $('tr.b-fight-details__table-row[data-link]');

    fightRows.each((_i, row) => {
      const cells = $(row).find('td.b-fight-details__table-col');
      if (cells.length < 10) return;

      const winnersCell = cells.eq(1);
      const links = winnersCell.find('a');
      const winnerName = links.eq(0).text().trim();
      const loserName = links.eq(1).text().trim();
      const winnerUrl = links.eq(0).attr('href') || '';
      const loserUrl = links.eq(1).attr('href') || '';
      const winnerId = winnerUrl.split('/fighter-details/')[1] || winnerName;
      const loserId = loserUrl.split('/fighter-details/')[1] || loserName;

      const method = cells.eq(7).text().trim();
      const round = parseInt(cells.eq(8).text().trim()) || 1;
      const time = cells.eq(9).text().trim();

      results.push({
        fightId: `${eventId}-${fightIndex}`,
        fighterAId: winnerId,
        fighterBId: loserId,
        winnerId,
        winnerName,
        loserName,
        method: normalizeMethod(method),
        round,
        time,
      });

      fightIndex++;
    });

    return results;
  } catch (err) {
    logger.error({ err, eventUrl }, 'Failed to fetch event results');
    return [];
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function parseDateString(s: string): string {
  try {
    const d = new Date(s);
    if (!isNaN(d.getTime())) return d.toISOString().split('T')[0];
  } catch {}
  return s;
}

function parseWeightClass(s: string): WeightClass {
  const w = s.toLowerCase();
  if (w.includes('straw') && w.includes('women')) return 'WomenStrawweight';
  if (w.includes('fly') && w.includes('women')) return 'WomenFlyweight';
  if (w.includes('bantam') && w.includes('women')) return 'WomenBantamweight';
  if (w.includes('feather') && w.includes('women')) return 'WomenFeatherweight';
  if (w.includes('straw')) return 'Strawweight';
  if (w.includes('fly')) return 'Flyweight';
  if (w.includes('bantam')) return 'Bantamweight';
  if (w.includes('feather')) return 'Featherweight';
  if (w.includes('light') && w.includes('heavy')) return 'LightHeavyweight';
  if (w.includes('light')) return 'Lightweight';
  if (w.includes('welter')) return 'Welterweight';
  if (w.includes('middle')) return 'Middleweight';
  if (w.includes('heavy')) return 'Heavyweight';
  return 'Welterweight';
}

function inferCardPosition(index: number, total: number): FightCardBout['cardPosition'] {
  if (index === 0) return 'main_event';
  if (index === 1) return 'co_main';
  if (index < 5) return 'main_card';
  if (index < total - 3) return 'prelim';
  return 'early_prelim';
}

function normalizeMethod(method: string): string {
  const m = method.toUpperCase();
  if (m.includes('KO') || m.includes('TKO')) return 'KO/TKO';
  if (m.includes('SUB')) return 'Submission';
  if (m.includes('DEC')) return 'Decision';
  return 'Other';
}
