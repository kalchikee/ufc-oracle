// UFC Oracle v4.1 — SQLite Database Layer (sql.js — pure JS, no native build)

import initSqlJs, { type Database as SqlJsDatabase } from 'sql.js';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { Fighter, Fight, Prediction, EloRatings, AccuracyStats, EventRecord } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DB_PATH = resolve(
  process.env.DB_PATH
    ? process.env.DB_PATH.startsWith('.')
      ? resolve(__dirname, '../../', process.env.DB_PATH)
      : process.env.DB_PATH
    : resolve(__dirname, '../../data/oracle.sqlite')
);

mkdirSync(dirname(DB_PATH), { recursive: true });

let _db: SqlJsDatabase | null = null;
let _SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

// ─── Init ─────────────────────────────────────────────────────────────────────

export async function initDb(): Promise<SqlJsDatabase> {
  if (_db) return _db;
  _SQL = await initSqlJs();
  if (existsSync(DB_PATH)) {
    const buf = readFileSync(DB_PATH);
    _db = new _SQL.Database(buf);
  } else {
    _db = new _SQL.Database();
  }
  initSchema(_db);
  persistDb();
  return _db;
}

export function getDb(): SqlJsDatabase {
  if (!_db) throw new Error('DB not initialized. Call initDb() first.');
  return _db;
}

export function persistDb(): void {
  if (!_db) return;
  writeFileSync(DB_PATH, Buffer.from(_db.export()));
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function run(sql: string, params: (string | number | null | undefined)[] = []): void {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.run(params.map(p => (p === undefined ? null : p)));
  stmt.free();
  persistDb();
}

function queryAll<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T[] {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results: T[] = [];
  while (stmt.step()) results.push(stmt.getAsObject() as T);
  stmt.free();
  return results;
}

function queryOne<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T | undefined {
  return queryAll<T>(sql, params)[0];
}

// ─── Schema ───────────────────────────────────────────────────────────────────

function initSchema(db: SqlJsDatabase): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS fighters (
      fighter_id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      nickname TEXT,
      weight_class TEXT NOT NULL,
      height REAL,
      reach REAL,
      stance TEXT,
      date_of_birth TEXT,
      camp TEXT,
      sig_strikes_landed_pm REAL NOT NULL DEFAULT 0,
      sig_strikes_absorbed_pm REAL NOT NULL DEFAULT 0,
      sig_strike_accuracy REAL NOT NULL DEFAULT 0,
      sig_strike_defense REAL NOT NULL DEFAULT 0,
      takedown_avg_per15 REAL NOT NULL DEFAULT 0,
      takedown_accuracy REAL NOT NULL DEFAULT 0,
      takedown_defense REAL NOT NULL DEFAULT 0,
      submission_avg_per15 REAL NOT NULL DEFAULT 0,
      knockdown_rate REAL NOT NULL DEFAULT 0,
      control_time_pct REAL NOT NULL DEFAULT 0,
      avg_fight_time REAL NOT NULL DEFAULT 0,
      wins INTEGER NOT NULL DEFAULT 0,
      losses INTEGER NOT NULL DEFAULT 0,
      draws INTEGER NOT NULL DEFAULT 0,
      win_pct REAL NOT NULL DEFAULT 0,
      ufc_wins INTEGER NOT NULL DEFAULT 0,
      ufc_losses INTEGER NOT NULL DEFAULT 0,
      finish_rate REAL NOT NULL DEFAULT 0,
      decision_rate REAL NOT NULL DEFAULT 0,
      style TEXT NOT NULL DEFAULT 'WellRounded',
      elo_overall REAL NOT NULL DEFAULT 1500,
      elo_striking REAL NOT NULL DEFAULT 1500,
      elo_grappling REAL NOT NULL DEFAULT 1500,
      last_fight_date TEXT,
      days_since_last_fight REAL,
      recent_wins INTEGER NOT NULL DEFAULT 0,
      recent_losses INTEGER NOT NULL DEFAULT 0,
      recent_sig_strikes_pm REAL NOT NULL DEFAULT 0,
      win_streak INTEGER NOT NULL DEFAULT 0,
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS fights (
      fight_id TEXT PRIMARY KEY,
      event_id TEXT NOT NULL,
      event_name TEXT NOT NULL,
      event_date TEXT NOT NULL,
      fighter_a_id TEXT NOT NULL,
      fighter_b_id TEXT NOT NULL,
      weight_class TEXT NOT NULL,
      is_main_event INTEGER NOT NULL DEFAULT 0,
      is_title_fight INTEGER NOT NULL DEFAULT 0,
      scheduled_rounds INTEGER NOT NULL DEFAULT 3,
      actual_rounds INTEGER,
      winner_id TEXT,
      method TEXT,
      round INTEGER,
      time TEXT
    );

    CREATE TABLE IF NOT EXISTS predictions (
      prediction_id TEXT PRIMARY KEY,
      fight_id TEXT NOT NULL,
      event_id TEXT NOT NULL,
      event_name TEXT NOT NULL,
      event_date TEXT NOT NULL,
      fighter_a_id TEXT NOT NULL,
      fighter_b_id TEXT NOT NULL,
      fighter_a_name TEXT NOT NULL,
      fighter_b_name TEXT NOT NULL,
      weight_class TEXT NOT NULL,
      card_position TEXT NOT NULL,
      is_main_event INTEGER NOT NULL DEFAULT 0,
      is_title_fight INTEGER NOT NULL DEFAULT 0,
      scheduled_rounds INTEGER NOT NULL DEFAULT 3,
      feature_vector TEXT NOT NULL,
      fighter_a_win_prob REAL NOT NULL,
      fighter_b_win_prob REAL NOT NULL,
      predicted_winner_id TEXT NOT NULL,
      predicted_winner_name TEXT NOT NULL,
      confidence_tier TEXT NOT NULL,
      ko_prob REAL NOT NULL DEFAULT 0,
      submission_prob REAL NOT NULL DEFAULT 0,
      decision_prob REAL NOT NULL DEFAULT 0,
      other_prob REAL NOT NULL DEFAULT 0,
      predicted_method TEXT NOT NULL,
      fighter_a_moneyline REAL,
      fighter_b_moneyline REAL,
      vegas_winner_prob REAL,
      edge REAL,
      edge_category TEXT,
      actual_winner_id TEXT,
      actual_method TEXT,
      correct INTEGER,
      method_correct INTEGER,
      model_version TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS accuracy_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      event_id TEXT NOT NULL,
      event_name TEXT NOT NULL,
      event_date TEXT NOT NULL,
      correct INTEGER NOT NULL,
      total INTEGER NOT NULL,
      accuracy REAL NOT NULL,
      hc_correct INTEGER NOT NULL DEFAULT 0,
      hc_total INTEGER NOT NULL DEFAULT 0,
      method_correct INTEGER NOT NULL DEFAULT 0,
      method_total INTEGER NOT NULL DEFAULT 0,
      recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);
}

// ─── Fighter CRUD ─────────────────────────────────────────────────────────────

export function upsertFighter(f: Fighter): void {
  run(`
    INSERT OR REPLACE INTO fighters (
      fighter_id, name, nickname, weight_class, height, reach, stance, date_of_birth, camp,
      sig_strikes_landed_pm, sig_strikes_absorbed_pm, sig_strike_accuracy, sig_strike_defense,
      takedown_avg_per15, takedown_accuracy, takedown_defense, submission_avg_per15,
      knockdown_rate, control_time_pct, avg_fight_time,
      wins, losses, draws, win_pct, ufc_wins, ufc_losses, finish_rate, decision_rate,
      style, elo_overall, elo_striking, elo_grappling,
      last_fight_date, days_since_last_fight, recent_wins, recent_losses, recent_sig_strikes_pm,
      win_streak, updated_at
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
  `, [
    f.fighterId, f.name, f.nickname ?? null, f.weightClass, f.height ?? null, f.reach ?? null,
    f.stance ?? null, f.dateOfBirth ?? null, f.camp ?? null,
    f.sigStrikesLandedPM, f.sigStrikesAbsorbedPM, f.sigStrikeAccuracy, f.sigStrikeDefense,
    f.takedownAvgPer15, f.takedownAccuracy, f.takedownDefense, f.submissionAvgPer15,
    f.knockdownRate, f.controlTimePct, f.avgFightTime,
    f.wins, f.losses, f.draws, f.winPct, f.ufcWins, f.ufcLosses, f.finishRate, f.decisionRate,
    f.style, f.eloOverall, f.eloStriking, f.eloGrappling,
    f.lastFightDate ?? null, f.daysSinceLastFight ?? null,
    f.recentWins, f.recentLosses, f.recentSigStrikesPM, f.winStreak,
  ]);
}

export function getFighter(fighterId: string): Fighter | undefined {
  const row = queryOne<Record<string, unknown>>('SELECT * FROM fighters WHERE fighter_id = ?', [fighterId]);
  if (!row) return undefined;
  return rowToFighter(row);
}

export function getFighterByName(name: string): Fighter | undefined {
  const row = queryOne<Record<string, unknown>>(
    "SELECT * FROM fighters WHERE LOWER(name) = LOWER(?)", [name]
  );
  if (!row) return undefined;
  return rowToFighter(row);
}

export function getAllFighters(): Fighter[] {
  return queryAll<Record<string, unknown>>('SELECT * FROM fighters').map(rowToFighter);
}

function rowToFighter(row: Record<string, unknown>): Fighter {
  return {
    fighterId: row.fighter_id as string,
    name: row.name as string,
    nickname: row.nickname as string | undefined,
    weightClass: row.weight_class as Fighter['weightClass'],
    height: row.height as number | undefined,
    reach: row.reach as number | undefined,
    stance: row.stance as string | undefined,
    dateOfBirth: row.date_of_birth as string | undefined,
    camp: row.camp as string | undefined,
    sigStrikesLandedPM: row.sig_strikes_landed_pm as number,
    sigStrikesAbsorbedPM: row.sig_strikes_absorbed_pm as number,
    sigStrikeAccuracy: row.sig_strike_accuracy as number,
    sigStrikeDefense: row.sig_strike_defense as number,
    takedownAvgPer15: row.takedown_avg_per15 as number,
    takedownAccuracy: row.takedown_accuracy as number,
    takedownDefense: row.takedown_defense as number,
    submissionAvgPer15: row.submission_avg_per15 as number,
    knockdownRate: row.knockdown_rate as number,
    controlTimePct: row.control_time_pct as number,
    avgFightTime: row.avg_fight_time as number,
    wins: row.wins as number,
    losses: row.losses as number,
    draws: row.draws as number,
    winPct: row.win_pct as number,
    ufcWins: row.ufc_wins as number,
    ufcLosses: row.ufc_losses as number,
    finishRate: row.finish_rate as number,
    decisionRate: row.decision_rate as number,
    style: row.style as Fighter['style'],
    eloOverall: row.elo_overall as number,
    eloStriking: row.elo_striking as number,
    eloGrappling: row.elo_grappling as number,
    lastFightDate: row.last_fight_date as string | undefined,
    daysSinceLastFight: row.days_since_last_fight as number | undefined,
    recentWins: row.recent_wins as number,
    recentLosses: row.recent_losses as number,
    recentSigStrikesPM: row.recent_sig_strikes_pm as number,
    winStreak: row.win_streak as number,
    updatedAt: row.updated_at as string,
  };
}

// ─── Fight CRUD ───────────────────────────────────────────────────────────────

export function upsertFight(f: Fight): void {
  run(`
    INSERT OR REPLACE INTO fights (
      fight_id, event_id, event_name, event_date, fighter_a_id, fighter_b_id,
      weight_class, is_main_event, is_title_fight, scheduled_rounds,
      actual_rounds, winner_id, method, round, time
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
  `, [
    f.fightId, f.eventId, f.eventName, f.eventDate, f.fighterAId, f.fighterBId,
    f.weightClass, f.isMainEvent ? 1 : 0, f.isTitleFight ? 1 : 0, f.scheduledRounds,
    f.actualRounds ?? null, f.winnerId ?? null, f.method ?? null, f.round ?? null, f.time ?? null,
  ]);
}

export function getFightsByEvent(eventId: string): Fight[] {
  return queryAll<Record<string, unknown>>('SELECT * FROM fights WHERE event_id = ?', [eventId]).map(rowToFight);
}

function rowToFight(row: Record<string, unknown>): Fight {
  return {
    fightId: row.fight_id as string,
    eventId: row.event_id as string,
    eventName: row.event_name as string,
    eventDate: row.event_date as string,
    fighterAId: row.fighter_a_id as string,
    fighterBId: row.fighter_b_id as string,
    weightClass: row.weight_class as Fight['weightClass'],
    isMainEvent: Boolean(row.is_main_event),
    isTitleFight: Boolean(row.is_title_fight),
    scheduledRounds: row.scheduled_rounds as number,
    actualRounds: row.actual_rounds as number | undefined,
    winnerId: row.winner_id as string | undefined,
    method: row.method as Fight['method'],
    round: row.round as number | undefined,
    time: row.time as string | undefined,
  };
}

// ─── Predictions CRUD ─────────────────────────────────────────────────────────

export function upsertPrediction(p: Prediction): void {
  run(`
    INSERT OR REPLACE INTO predictions (
      prediction_id, fight_id, event_id, event_name, event_date,
      fighter_a_id, fighter_b_id, fighter_a_name, fighter_b_name,
      weight_class, card_position, is_main_event, is_title_fight, scheduled_rounds,
      feature_vector, fighter_a_win_prob, fighter_b_win_prob,
      predicted_winner_id, predicted_winner_name, confidence_tier,
      ko_prob, submission_prob, decision_prob, other_prob, predicted_method,
      fighter_a_moneyline, fighter_b_moneyline, vegas_winner_prob, edge, edge_category,
      actual_winner_id, actual_method, correct, method_correct, model_version
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
  `, [
    p.predictionId, p.fightId, p.eventId, p.eventName, p.eventDate,
    p.fighterAId, p.fighterBId, p.fighterAName, p.fighterBName,
    p.weightClass, p.cardPosition, p.isMainEvent ? 1 : 0, p.isTitleFight ? 1 : 0, p.scheduledRounds,
    JSON.stringify(p.featureVector), p.fighterAWinProb, p.fighterBWinProb,
    p.predictedWinnerId, p.predictedWinnerName, p.confidenceTier,
    p.koProb, p.submissionProb, p.decisionProb, p.otherProb, p.predictedMethod,
    p.fighterAMoneyline ?? null, p.fighterBMoneyline ?? null,
    p.vegasWinnerProb ?? null, p.edge ?? null, p.edgeCategory ?? null,
    p.actualWinnerId ?? null, p.actualMethod ?? null,
    p.correct !== undefined ? (p.correct ? 1 : 0) : null,
    p.methodCorrect !== undefined ? (p.methodCorrect ? 1 : 0) : null,
    p.modelVersion,
  ]);
}

export function getPredictionsByEvent(eventId: string): Prediction[] {
  return queryAll<Record<string, unknown>>('SELECT * FROM predictions WHERE event_id = ?', [eventId]).map(rowToPrediction);
}

export function getLatestPredictions(): Prediction[] {
  return queryAll<Record<string, unknown>>(
    `SELECT * FROM predictions WHERE event_date = (SELECT MAX(event_date) FROM predictions WHERE event_date >= date('now'))`
  ).map(rowToPrediction);
}

function rowToPrediction(row: Record<string, unknown>): Prediction {
  return {
    predictionId: row.prediction_id as string,
    fightId: row.fight_id as string,
    eventId: row.event_id as string,
    eventName: row.event_name as string,
    eventDate: row.event_date as string,
    fighterAId: row.fighter_a_id as string,
    fighterBId: row.fighter_b_id as string,
    fighterAName: row.fighter_a_name as string,
    fighterBName: row.fighter_b_name as string,
    weightClass: row.weight_class as Prediction['weightClass'],
    cardPosition: row.card_position as string,
    isMainEvent: Boolean(row.is_main_event),
    isTitleFight: Boolean(row.is_title_fight),
    scheduledRounds: row.scheduled_rounds as number,
    featureVector: JSON.parse(row.feature_vector as string),
    fighterAWinProb: row.fighter_a_win_prob as number,
    fighterBWinProb: row.fighter_b_win_prob as number,
    predictedWinnerId: row.predicted_winner_id as string,
    predictedWinnerName: row.predicted_winner_name as string,
    confidenceTier: row.confidence_tier as Prediction['confidenceTier'],
    koProb: row.ko_prob as number,
    submissionProb: row.submission_prob as number,
    decisionProb: row.decision_prob as number,
    otherProb: row.other_prob as number,
    predictedMethod: row.predicted_method as Prediction['predictedMethod'],
    fighterAMoneyline: row.fighter_a_moneyline as number | undefined,
    fighterBMoneyline: row.fighter_b_moneyline as number | undefined,
    vegasWinnerProb: row.vegas_winner_prob as number | undefined,
    edge: row.edge as number | undefined,
    edgeCategory: row.edge_category as Prediction['edgeCategory'],
    actualWinnerId: row.actual_winner_id as string | undefined,
    actualMethod: row.actual_method as Prediction['actualMethod'],
    correct: row.correct !== null ? Boolean(row.correct) : undefined,
    methodCorrect: row.method_correct !== null ? Boolean(row.method_correct) : undefined,
    modelVersion: row.model_version as string,
    createdAt: row.created_at as string,
  };
}

// ─── Accuracy ─────────────────────────────────────────────────────────────────

export function getYTDAccuracy(): AccuracyStats {
  const year = new Date().getFullYear();
  const rows = queryAll<Record<string, unknown>>(
    `SELECT * FROM predictions WHERE correct IS NOT NULL AND event_date LIKE '${year}%'`
  ).map(rowToPrediction);

  const ytdTotal = rows.length;
  const ytdCorrect = rows.filter(r => r.correct).length;
  const hc = rows.filter(r => r.confidenceTier === 'high_conviction' || r.confidenceTier === 'lock');
  const hcCorrect = hc.filter(r => r.correct).length;
  const method = rows.filter(r => r.methodCorrect !== undefined);
  const methodCorrect = method.filter(r => r.methodCorrect).length;
  const underdogs = rows.filter(r => r.vegasWinnerProb !== undefined && r.fighterAWinProb > 0.5 && (r.vegasWinnerProb ?? 0) < 0.5);
  const underdogCorrect = underdogs.filter(r => r.correct).length;
  const mainEvent = rows.filter(r => r.isMainEvent);
  const mainEventCorrect = mainEvent.filter(r => r.correct).length;

  const brier = ytdTotal > 0
    ? rows.reduce((sum, r) => sum + Math.pow((r.correct ? 1 : 0) - r.fighterAWinProb, 2), 0) / ytdTotal
    : 0;

  // Event-by-event log
  const eventLog = queryAll<Record<string, unknown>>(
    `SELECT * FROM accuracy_log WHERE event_date LIKE '${year}%' ORDER BY event_date DESC LIMIT 20`
  );

  return {
    ytdCorrect,
    ytdTotal,
    ytdAccuracy: ytdTotal > 0 ? ytdCorrect / ytdTotal : 0,
    highConvCorrect: hcCorrect,
    highConvTotal: hc.length,
    highConvAccuracy: hc.length > 0 ? hcCorrect / hc.length : 0,
    methodCorrect,
    methodTotal: method.length,
    methodAccuracy: method.length > 0 ? methodCorrect / method.length : 0,
    valueBetROI: 0, // computed separately
    underdogCorrect,
    underdogTotal: underdogs.length,
    mainEventCorrect,
    mainEventTotal: mainEvent.length,
    brierScore: brier,
    eventRecord: eventLog.map(r => ({
      eventId: r.event_id as string,
      eventName: r.event_name as string,
      eventDate: r.event_date as string,
      correct: r.correct as number,
      total: r.total as number,
      accuracy: r.accuracy as number,
    })),
  };
}

export function logEventAccuracy(
  eventId: string, eventName: string, eventDate: string,
  correct: number, total: number, hcCorrect: number, hcTotal: number,
  methodCorrect: number, methodTotal: number
): void {
  run(`
    INSERT OR REPLACE INTO accuracy_log (event_id, event_name, event_date, correct, total, accuracy, hc_correct, hc_total, method_correct, method_total)
    VALUES (?,?,?,?,?,?,?,?,?,?)
  `, [eventId, eventName, eventDate, correct, total, total > 0 ? correct / total : 0, hcCorrect, hcTotal, methodCorrect, methodTotal]);
}
