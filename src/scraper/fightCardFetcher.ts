// UFC Oracle v4.1 — Fight Card Fetcher
// Fetches the upcoming UFC event card from UFCStats.com (primary) with an
// ESPN MMA fallback. UFCStats added a JavaScript-based anti-bot challenge
// in mid-May 2026 — every request now returns a "Checking your browser…"
// page with zero data rows, so the cheerio scrape silently parses 0
// events and the entire UFC pipeline went dark. The ESPN MMA scoreboard
// endpoint exposes the same upcoming event + bout list without a
// challenge, so we use it as the discovery + card source and map fighter
// names against the existing UFCStats-keyed fighters table.

import fetch from 'node-fetch';
import * as cheerio from 'cheerio';
import { logger } from '../logger.js';
import { getFighterByName } from '../db/database.js';
import type { FightCard, FightCardBout, WeightClass } from '../types.js';

const UFCSTATS_EVENTS_URL = 'http://www.ufcstats.com/statistics/events/upcoming';
const UFCSTATS_COMPLETED_URL = 'http://www.ufcstats.com/statistics/events/completed';
const BASE_URL = 'http://www.ufcstats.com';
const ESPN_MMA_BASE = 'https://site.api.espn.com/apis/site/v2/sports/mma/ufc';
const DELAY_MS = 1200;
// Sentinel embedded in the eventUrl so fetchFightCard can route to the
// ESPN handler without ambiguity.
const ESPN_URL_PREFIX = 'espn-mma://';

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
  // UFCStats.com is quirky: events happening TODAY get moved from the "upcoming"
  // page to "completed" even before they start. So we check both pages and pick
  // the earliest event whose date is today-or-later.
  const candidates: Array<{ eventId: string; eventUrl: string; eventDate: string; eventName: string }> = [];

  for (const url of [UFCSTATS_EVENTS_URL, UFCSTATS_COMPLETED_URL]) {
    try {
      const $ = await fetchHtml(url);
      // Check the first ~5 rows with event links on each page
      const rows = $('tr.b-statistics__table-row').filter((_i, el) => $(el).find('a.b-link').length > 0).slice(0, 5);
      rows.each((_i, el) => {
        const link = $(el).find('a.b-link').first();
        const eventUrl = link.attr('href') || '';
        const eventName = link.text().trim();
        const dateText = $(el).find('span.b-statistics__date').text().trim();
        if (!eventUrl) return;
        const eventDate = parseDateString(dateText);
        if (!eventDate) return;
        candidates.push({
          eventId: eventUrl.split('/event-details/')[1] ?? eventUrl,
          eventUrl,
          eventDate,
          eventName,
        });
      });
    } catch (err) {
      logger.warn({ err, url }, 'Failed to fetch UFC events page (continuing)');
    }
  }

  // Pick the earliest event whose date is today or later (in local time)
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  const futureOrToday = candidates.filter(c => new Date(c.eventDate) >= now);
  if (futureOrToday.length > 0) {
    futureOrToday.sort((a, b) => new Date(a.eventDate).getTime() - new Date(b.eventDate).getTime());
    const next = futureOrToday[0];
    logger.info({ eventName: next.eventName, eventDate: next.eventDate, eventUrl: next.eventUrl }, 'Next UFC event found (UFCStats)');
    return { eventId: next.eventId, eventUrl: next.eventUrl, eventDate: next.eventDate };
  }

  // UFCStats returned nothing — either off-week or (since mid-May 2026) the
  // JS challenge blocked the scrape. Fall back to ESPN's MMA scoreboard.
  logger.info('UFCStats produced no events — falling back to ESPN MMA');
  const espn = await getNextUFCEventFromESPN();
  if (!espn) {
    logger.info('No upcoming or same-day UFC events found on UFCStats or ESPN');
    return null;
  }
  return espn;
}

async function getNextUFCEventFromESPN(): Promise<{ eventId: string; eventUrl: string; eventDate: string } | null> {
  try {
    const resp = await fetch(`${ESPN_MMA_BASE}/scoreboard`, {
      headers: { 'User-Agent': 'UFC-Oracle-Research-Bot/4.1' },
      signal: AbortSignal.timeout(15000),
    });
    if (!resp.ok) {
      logger.warn({ status: resp.status }, 'ESPN MMA scoreboard returned non-200');
      return null;
    }
    const data = await resp.json() as { events?: Array<{ id: string; name: string; date: string }> };
    const events = data.events ?? [];
    if (events.length === 0) return null;
    const now = new Date(); now.setHours(0, 0, 0, 0);
    const future = events
      .map(e => ({ id: e.id, name: e.name, date: e.date.slice(0, 10) }))
      .filter(e => new Date(e.date) >= now)
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    if (future.length === 0) return null;
    const next = future[0];
    logger.info({ eventName: next.name, eventDate: next.date, espnId: next.id }, 'Next UFC event found (ESPN)');
    return { eventId: next.id, eventUrl: `${ESPN_URL_PREFIX}${next.id}`, eventDate: next.date };
  } catch (err) {
    logger.warn({ err }, 'ESPN MMA scoreboard fetch failed');
    return null;
  }
}

export function isFightWeek(eventDate: string): boolean {
  const now = new Date();
  const event = new Date(eventDate);
  const diffDays = (event.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);
  // Fight week = event is within the next 7 days OR today (allow same-day run)
  return diffDays >= -1 && diffDays <= 7;
}

/** Returns true only if the event is TODAY (matches calendar date in local time).
 *  Used by the fight-night workflow to send Discord only on event day,
 *  regardless of what day of the week the event falls on. */
export function isFightDay(eventDate: string): boolean {
  const now = new Date();
  const event = new Date(eventDate);
  return (
    now.getFullYear() === event.getFullYear() &&
    now.getMonth()    === event.getMonth()    &&
    now.getDate()     === event.getDate()
  );
}

// ─── Fight card scraping ──────────────────────────────────────────────────────

export async function fetchFightCard(eventUrl: string): Promise<FightCard | null> {
  if (eventUrl.startsWith(ESPN_URL_PREFIX)) {
    return fetchFightCardFromESPN(eventUrl.slice(ESPN_URL_PREFIX.length));
  }
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

// ─── ESPN MMA event-card fetcher (fallback when UFCStats is blocked) ─────────

async function fetchFightCardFromESPN(eventId: string): Promise<FightCard | null> {
  try {
    const resp = await fetch(`${ESPN_MMA_BASE}/scoreboard?dates=`, {
      headers: { 'User-Agent': 'UFC-Oracle-Research-Bot/4.1' },
      signal: AbortSignal.timeout(15000),
    });
    // Pull the event from the broader scoreboard so we get its competitions
    // list. Could also hit the per-event endpoint but scoreboard is one call.
    const detailResp = await fetch(`${ESPN_MMA_BASE}/scoreboard`, {
      headers: { 'User-Agent': 'UFC-Oracle-Research-Bot/4.1' },
      signal: AbortSignal.timeout(15000),
    });
    void resp;
    if (!detailResp.ok) {
      logger.error({ eventId, status: detailResp.status }, 'ESPN MMA fetch failed');
      return null;
    }
    const data = await detailResp.json() as {
      events?: Array<{
        id: string; name: string; date: string;
        competitions?: Array<{
          id: string;
          competitors?: Array<{
            athlete?: { id?: string; displayName?: string; weightClass?: { text?: string } };
            winner?: boolean;
          }>;
          venue?: { fullName?: string; address?: { city?: string; state?: string; country?: string } };
        }>;
      }>;
    };
    const event = (data.events ?? []).find(e => e.id === eventId);
    if (!event) {
      logger.warn({ eventId }, 'Event not found in ESPN scoreboard');
      return null;
    }

    const comps = event.competitions ?? [];
    const venueObj = comps[0]?.venue;
    const venue = venueObj?.fullName ?? '';
    const addr = venueObj?.address;
    const location = [addr?.city, addr?.state, addr?.country].filter(Boolean).join(', ');

    // ESPN orders competitions chronologically (early prelims start first,
    // main event last). Reverse so the main event comes first — matches the
    // UFCStats path's ordering and predictionRunner's expectations. Look up
    // fighter_id by name from the existing fighters table; unknown fighters
    // keep undefined IDs and the prediction pipeline handles that case.
    const ordered = [...comps].reverse();
    const bouts: FightCardBout[] = [];
    for (let i = 0; i < ordered.length; i++) {
      const c = ordered[i];
      const competitors = c.competitors ?? [];
      const a = competitors[0];
      const b = competitors[1];
      if (!a?.athlete?.displayName || !b?.athlete?.displayName) continue;
      const aName = a.athlete.displayName;
      const bName = b.athlete.displayName;
      // DB may not be initialized in tests / dry-runs — fall back to
      // undefined fighter IDs (predictionRunner handles unknown fighters).
      let aRow, bRow;
      try { aRow = getFighterByName(aName); } catch { aRow = undefined; }
      try { bRow = getFighterByName(bName); } catch { bRow = undefined; }
      const weightText = a.athlete.weightClass?.text ?? '';
      const weightClass = parseWeightClass(weightText);
      const isMainEvent = i === 0;
      const isTitleFight = (event.name ?? '').toLowerCase().includes('title') ||
                           weightText.toLowerCase().includes('title');
      const cardPosition = inferCardPosition(i, ordered.length);
      bouts.push({
        fightId: `${eventId}-${i}`,
        fighterA: aName,
        fighterB: bName,
        fighterAId: aRow?.fighterId,
        fighterBId: bRow?.fighterId,
        weightClass,
        isMainEvent,
        isTitleFight,
        scheduledRounds: isTitleFight || isMainEvent ? 5 : 3,
        cardPosition,
      });
    }
    if (bouts.length === 0) {
      logger.warn({ eventId }, 'ESPN event had zero parseable bouts');
      return null;
    }
    const matched = bouts.filter(b => b.fighterAId && b.fighterBId).length;
    logger.info(
      { eventId, eventName: event.name, bouts: bouts.length, fighterIdsMatched: matched },
      'Fight card built from ESPN',
    );
    return {
      eventId,
      eventName: event.name,
      eventDate: event.date.slice(0, 10),
      location,
      venue,
      fights: bouts,
    };
  } catch (err) {
    logger.error({ err, eventId }, 'Failed to fetch fight card from ESPN');
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
  if (eventUrl.startsWith(ESPN_URL_PREFIX)) {
    return fetchEventResultsFromESPN(eventUrl.slice(ESPN_URL_PREFIX.length));
  }
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

async function fetchEventResultsFromESPN(eventId: string): Promise<FightResult[]> {
  try {
    const resp = await fetch(`${ESPN_MMA_BASE}/scoreboard`, {
      headers: { 'User-Agent': 'UFC-Oracle-Research-Bot/4.1' },
      signal: AbortSignal.timeout(15000),
    });
    if (!resp.ok) {
      logger.error({ eventId, status: resp.status }, 'ESPN MMA scoreboard fetch failed for results');
      return [];
    }
    const data = await resp.json() as {
      events?: Array<{
        id: string;
        competitions?: Array<{
          id: string;
          status?: { type?: { completed?: boolean } };
          competitors?: Array<{
            athlete?: { id?: string; displayName?: string };
            winner?: boolean;
          }>;
          notes?: Array<{ headline?: string }>;
        }>;
      }>;
    };
    const event = (data.events ?? []).find(e => e.id === eventId);
    if (!event) return [];
    const comps = event.competitions ?? [];
    const results: FightResult[] = [];
    let fightIndex = 0;
    for (const c of comps) {
      const completed = c.status?.type?.completed;
      if (!completed) { fightIndex++; continue; }
      const competitors = c.competitors ?? [];
      const winnerComp = competitors.find(x => x.winner) ?? competitors[0];
      const loserComp = competitors.find(x => x !== winnerComp);
      if (!winnerComp?.athlete?.displayName || !loserComp?.athlete?.displayName) {
        fightIndex++;
        continue;
      }
      const winnerName = winnerComp.athlete.displayName;
      const loserName = loserComp.athlete.displayName;
      let winnerRow, loserRow;
      try { winnerRow = getFighterByName(winnerName); } catch { winnerRow = undefined; }
      try { loserRow = getFighterByName(loserName); } catch { loserRow = undefined; }
      // ESPN doesn't expose round/method/time in the scoreboard payload —
      // recap can backfill these later if needed. winnerId / loserId from
      // the local DB are what the rest of the pipeline keys on.
      results.push({
        fightId: `${eventId}-${fightIndex}`,
        fighterAId: winnerRow?.fighterId ?? winnerName,
        fighterBId: loserRow?.fighterId ?? loserName,
        winnerId: winnerRow?.fighterId ?? winnerName,
        winnerName,
        loserName,
        method: 'Decision',  // placeholder; refine via event-detail endpoint later
        round: 3,
        time: '5:00',
      });
      fightIndex++;
    }
    logger.info(
      { eventId, completed: results.length, total: comps.length },
      'Event results pulled from ESPN',
    );
    return results;
  } catch (err) {
    logger.error({ err, eventId }, 'Failed to fetch event results from ESPN');
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
