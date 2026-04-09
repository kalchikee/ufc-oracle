// UFC Oracle v4.1 — Odds Scraper
// Fetches live UFC moneylines from The Odds API (https://the-odds-api.com)
// Enriches FightCard bouts with American-format moneylines automatically.
//
// Free tier: 500 requests/month. Set ODDS_API_KEY in env / GitHub Secrets.
// If the key is missing or the call fails, bouts keep whatever odds were
// already set (or undefined) — graceful degradation with no hard failure.

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type { FightCardBout } from '../types.js';

const ODDS_API_BASE = 'https://api.the-odds-api.com/v4';
const SPORT = 'mma_mixed_martial_arts';

// Preferred books in priority order (most accurate lines first)
const BOOK_PRIORITY = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'bovada', 'unibet_us'];

interface OddsOutcome {
  name: string;
  price: number; // American moneyline
}

interface OddsMarket {
  key: string;
  outcomes: OddsOutcome[];
}

interface OddsBookmaker {
  key: string;
  markets: OddsMarket[];
}

interface OddsEvent {
  id: string;
  commence_time: string;
  home_team: string;
  away_team: string;
  bookmakers: OddsBookmaker[];
}

// ─── Name normalization ───────────────────────────────────────────────────────
// Strips accents, punctuation, and normalises spacing so
// "Renato Moicano" matches "Renato Moicano" from any source.

function normalizeName(name: string): string {
  return name
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')  // strip diacritics
    .replace(/[^a-z\s]/g, '')         // letters + spaces only
    .replace(/\s+/g, ' ')
    .trim();
}

function namesMatch(a: string, b: string): boolean {
  const na = normalizeName(a);
  const nb = normalizeName(b);
  if (na === nb) return true;
  // One name contained in the other (handles "Jr.", initials, etc.)
  if (na.includes(nb) || nb.includes(na)) return true;
  // Last-name match as loose fallback (avoids false positives on short names)
  const lastA = na.split(' ').at(-1) ?? na;
  const lastB = nb.split(' ').at(-1) ?? nb;
  return lastA.length > 3 && lastA === lastB;
}

// ─── Best moneyline for a fighter ────────────────────────────────────────────

function bestLine(event: OddsEvent, fighterName: string): number | undefined {
  // Try books in priority order first
  for (const bookKey of BOOK_PRIORITY) {
    const bk = event.bookmakers.find(b => b.key === bookKey);
    const market = bk?.markets.find(m => m.key === 'h2h');
    const outcome = market?.outcomes.find(o => namesMatch(o.name, fighterName));
    if (outcome) return outcome.price;
  }
  // Fall back to any available bookmaker
  for (const bk of event.bookmakers) {
    const market = bk.markets.find(m => m.key === 'h2h');
    const outcome = market?.outcomes.find(o => namesMatch(o.name, fighterName));
    if (outcome) return outcome.price;
  }
  return undefined;
}

// ─── Main export ──────────────────────────────────────────────────────────────

export async function enrichWithOdds(bouts: FightCardBout[]): Promise<void> {
  const apiKey = process.env.ODDS_API_KEY;
  if (!apiKey) {
    logger.info('ODDS_API_KEY not set — skipping odds enrichment (add to GitHub Secrets to enable)');
    return;
  }

  try {
    const url =
      `${ODDS_API_BASE}/sports/${SPORT}/odds/` +
      `?apiKey=${apiKey}&regions=us&markets=h2h&oddsFormat=american`;

    const resp = await fetch(url, { signal: AbortSignal.timeout(12_000) });

    if (!resp.ok) {
      logger.warn({ status: resp.status }, 'Odds API request failed — continuing without odds');
      return;
    }

    const remaining = resp.headers.get('x-requests-remaining');
    const used = resp.headers.get('x-requests-used');
    logger.info({ remaining, used }, 'Odds API quota after request');

    const events = (await resp.json()) as OddsEvent[];
    let enriched = 0;

    for (const bout of bouts) {
      const event = events.find(e => {
        const fighters = [e.home_team, e.away_team];
        return (
          fighters.some(f => namesMatch(f, bout.fighterA)) &&
          fighters.some(f => namesMatch(f, bout.fighterB))
        );
      });

      if (!event) continue;

      const mlA = bestLine(event, bout.fighterA);
      const mlB = bestLine(event, bout.fighterB);

      if (mlA !== undefined) bout.fighterAMoneyline = mlA;
      if (mlB !== undefined) bout.fighterBMoneyline = mlB;

      if (mlA !== undefined || mlB !== undefined) {
        enriched++;
        logger.info(
          { a: bout.fighterA, b: bout.fighterB, mlA, mlB },
          'Bout odds enriched',
        );
      }
    }

    logger.info({ enriched, total: bouts.length }, 'Odds enrichment complete');
  } catch (err) {
    logger.error({ err }, 'Odds API error — continuing without odds');
  }
}
