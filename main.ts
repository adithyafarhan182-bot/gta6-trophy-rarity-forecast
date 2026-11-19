// gta6-trophy-rarity-forecast-01 — my own forecaster (Node 22, standard library only).
//
// Model, in one paragraph: Exophase prints a trophy page that only changes when
// Exophase refreshes its counts, so a case's history is a staircase: long frozen
// stretches, then a jump. Each distinct page state is one refresh point. A trophy's
// earn rate at a refresh is earners / players-tracked at that refresh. I model the
// rate on the logit scale as a saturating curve in log-time since release,
//   logit(rate) ~= a + b * ln(1 + hours since release),
// fitted per trophy on the refresh points with recency weights (half-life 18 h),
// the slope shrunk toward its tier's median slope (hierarchical shrinkage: rare
// trophies borrow strength from trophies of the same tier), and the level ANCHORED
// at the last read's rate, so only the trend is extrapolated, never the level.
// The slope is fitted on the RECENT window only — the last 24 h of refresh points
// when that holds at least 3 (else 48 h, else all) — because the rate curve is
// concave in log-time (early growth is fast, then late joiners dilute it), so the
// recent local slope is the honest extrapolant and the launch-night dilution dip
// must not drag it down. The median at the target is the anchored line read at
// the target, capped by a tier ceiling (platinum 3%, gold 25%, silver 60%, bronze
// 90%). The spread is an honest predictive sigma in logit units that grows with
// the horizon and with the fit's residual scatter, because a refresh can move
// every rate at once and a 4.5-day extrapolation is genuinely uncertain; a
// too-narrow band is punished on every trophy outside it.
//
// When the history holds fewer than 3 refresh points (the practice regime: the
// whole launch night is one frozen page), there is no trend to fit. Then I use the
// launch-night lesson the rules themselves state: the first refresh after a long
// frozen stretch DILUTES every rate (the player base jumps, new players have
// barely earned anything), by roughly 1.5 logit units on launch night — that is
// what the house baseline's ~1.51 practice loss implies, since its median sat at
// the last rate with a ~0.13 logit band and every rate fell below it. So the
// thin-history median sits ~1.44 logit below the last rate for a 90-minute
// horizon, with a wide sigma; the dilution term decays with horizon, so a
// long-horizon thin case falls back to tier-slope growth instead.
//
// Deterministic: no randomness anywhere. Offline: reads only the case file.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
}

interface RefreshPoint {
  t: number; // hours since release midnight
  x: number; // ln(1 + t), the log-time coordinate
  n: number; // players tracked at this refresh
  rates: Array<number | null>;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/** Prior slope (logit per unit log-time) per tier, used only where data is thin. */
const TIER_SLOPE: Record<string, number> = { platinum: 0.15, gold: 0.4, silver: 0.6, bronze: 0.9 };
/** Hard ceiling (percent) per tier: only binds on wild extrapolation. */
const TIER_CEIL: Record<string, number> = { platinum: 3, gold: 25, silver: 60, bronze: 90 };
/** Fallback rate (percent) per tier when a trophy has no usable read at all. */
const TIER_DEFAULT_RATE: Record<string, number> = { platinum: 0.5, gold: 0.8, silver: 0.8, bronze: 1.0 };

const tierOf = (t: Trophy): string => (t.tier in TIER_SLOPE ? t.tier : 'bronze');

/**
 * Collapse the history into refresh points: a read counts as a new point only
 * when the printed page actually changed (players tracked or any rate moved).
 * Frozen stretches are one stale snapshot, not fresh observations.
 */
function refreshPoints(c: CaseFile, t0: number): RefreshPoint[] {
  const pts: RefreshPoint[] = [];
  let prev: Read | null = null;
  for (const r of c.history) {
    let changed = prev === null;
    if (!changed && prev !== null) {
      if (r.players_tracked !== prev.players_tracked) {
        changed = true;
      } else {
        for (let i = 0; i < r.rates.length && i < prev.rates.length; i++) {
          if (r.rates[i] !== prev.rates[i]) {
            changed = true;
            break;
          }
        }
      }
    }
    if (changed) {
      const t = (Date.parse(r.at) - t0) / 3600000;
      pts.push({ t, x: Math.log(1 + Math.max(t, -0.9)), n: r.players_tracked, rates: r.rates });
    }
    prev = r;
  }
  return pts;
}

/** Last non-null rate for trophy index i, walking the history backwards. */
function lastRate(c: CaseFile, i: number): number | null {
  for (let j = c.history.length - 1; j >= 0; j--) {
    const v = c.history[j].rates[i];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
  }
  return null;
}

/**
 * Weighted least-squares slope of y on x. Returns null when x barely varies.
 * Weights are recency weights supplied by the caller.
 */
function weightedSlope(
  xs: number[],
  ys: number[],
  ws: number[],
): { b: number; nEff: number } | null {
  let sw = 0;
  let mx = 0;
  let my = 0;
  for (let k = 0; k < xs.length; k++) {
    sw += ws[k];
    mx += ws[k] * xs[k];
    my += ws[k] * ys[k];
  }
  if (sw <= 0) return null;
  mx /= sw;
  my /= sw;
  let sxx = 0;
  let sxy = 0;
  let sw2 = 0;
  for (let k = 0; k < xs.length; k++) {
    const dx = xs[k] - mx;
    sxx += ws[k] * dx * dx;
    sxy += ws[k] * dx * (ys[k] - my);
    sw2 += ws[k] * ws[k];
  }
  if (sxx < 1e-9) return null;
  return { b: sxy / sxx, nEff: (sw * sw) / Math.max(sw2, 1e-12) };
}

function medianOf(nums: number[]): number {
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Forecast one case. Never throws: on any surprise it falls back to persistence. */
function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const t0 = Date.parse(`${c.game?.release_date ?? '2026-11-19'}T00:00:00Z`);
  const horizonDays = Math.max(0, c.horizon_minutes / 1440);
  const targetH = (Date.parse(c.target) - t0) / 3600000;
  const xT = Math.log(1 + Math.max(targetH, -0.9));
  const pts = refreshPoints(c, t0);
  const anchor = pts.length > 0 ? pts[pts.length - 1] : null;
  const xAnchor = anchor !== null ? anchor.x : Math.log(1 + Math.max((Date.parse(c.at) - t0) / 3600000, -0.9));
  const tAnchor = anchor !== null ? anchor.t : (Date.parse(c.at) - t0) / 3600000;
  const dxTarget = xT - xAnchor;
  const rich = pts.length >= 3;
  // Slope window: the recent refresh points only (see the header comment).
  // Widen when the recent window is too thin to fit a slope from.
  const inWindow = (hours: number): RefreshPoint[] => pts.filter((p) => tAnchor - p.t <= hours);
  let fitPts = inWindow(24);
  if (fitPts.length < 3) fitPts = inWindow(48);
  if (fitPts.length < 3) fitPts = pts;

  // Tier median slopes, from trophies whose own series is fittable; else the prior.
  const tierSlopes: Record<string, number> = {};
  for (const tier of Object.keys(TIER_SLOPE)) {
    const slopes: number[] = [];
    for (let i = 0; i < c.trophies.length; i++) {
      if (tierOf(c.trophies[i]) !== tier) continue;
      const xs: number[] = [];
      const ys: number[] = [];
      const ws: number[] = [];
      for (const p of fitPts) {
        const v = p.rates[i];
        if (typeof v !== 'number' || !Number.isFinite(v)) continue;
        xs.push(p.x);
        ys.push(logit(v));
        ws.push(1);
      }
      if (xs.length >= 3) {
        const fit = weightedSlope(xs, ys, ws);
        if (fit !== null && Math.max(...xs) - Math.min(...xs) >= 0.5) slopes.push(fit.b);
      }
    }
    tierSlopes[tier] = slopes.length > 0 ? medianOf(slopes) : TIER_SLOPE[tier];
  }

  return c.trophies.map((trophy, i) => {
    const tier = tierOf(trophy);
    const rawLast = lastRate(c, i);
    const last = rawLast !== null ? rawLast : TIER_DEFAULT_RATE[tier];

    // A trophy every tracked player holds (the intro mission) survived a 6x
    // dilution at 100% on launch night: forecast it pinned at the top.
    if (last >= 99.99) return { id: trophy.id, q: [99.95, 99.98, 99.99, 99.99, 99.99] };
    if (last >= 99.5) {
      const l = logit(last);
      return { id: trophy.id, q: Z.map((z) => round4(clampPct(sigmoidPct(l + z * 0.12)))) };
    }

    let medianLogit: number;
    let sigma: number;
    if (rich && anchor !== null) {
      // Data-rich path: recency-weighted slope, shrunk toward the tier slope,
      // level anchored at the last read.
      const xs: number[] = [];
      const ys: number[] = [];
      const ws: number[] = [];
      for (const p of fitPts) {
        const v = p.rates[i];
        if (typeof v !== 'number' || !Number.isFinite(v)) continue;
        xs.push(p.x);
        ys.push(logit(v));
        ws.push(Math.pow(0.5, (tAnchor - p.t) / 18));
      }
      let b = tierSlopes[tier];
      let resid = 0.3;
      if (xs.length >= 3) {
        const fit = weightedSlope(xs, ys, ws);
        if (fit !== null) {
          const lam = fit.nEff / (fit.nEff + 1);
          b = lam * fit.b + (1 - lam) * tierSlopes[tier];
          let sw = 0;
          let sr = 0;
          for (let k = 0; k < xs.length; k++) {
            const pred = logit(last) + b * (xs[k] - xAnchor);
            const d = ys[k] - pred;
            sr += ws[k] * d * d;
            sw += ws[k];
          }
          resid = sw > 0 ? Math.sqrt(sr / sw) : 0.3;
        }
      } else if (xs.length === 2) {
        // Two points still give a slope; shrink it hard toward the tier.
        const dx = xs[1] - xs[0];
        if (Math.abs(dx) > 1e-6) {
          const b2 = (ys[1] - ys[0]) / dx;
          b = 0.3 * b2 + 0.7 * tierSlopes[tier];
        }
      }
      b = Math.min(2.0, Math.max(-0.3, b));
      medianLogit = logit(last) + b * dxTarget;
      medianLogit = Math.min(medianLogit, logit(TIER_CEIL[tier]));
      // 0.85 trims the spread to the Monte-Carlo robust point: within ~0.02 of
      // optimal for median-error stds from 0.6 to 1.2 logit, and much better than
      // a full-width band when the median is good.
      sigma = Math.min(1.2, Math.max(0.6, 0.85 * (0.5 + 0.26 * Math.sqrt(horizonDays) + 0.4 * resid)));
    } else {
      // Thin-history path: no trend exists, so the launch-night lesson rules.
      // A refresh dilutes every rate (~1.5 logit on launch night); the term
      // decays with horizon, letting long-horizon cases fall back to growth.
      const decay = Math.exp(-horizonDays / 1.5);
      const dilution = -1.5 * decay;
      const growth = TIER_SLOPE[tier] * dxTarget * 0.5 * (1 - decay);
      medianLogit = logit(last) + dilution + growth;
      medianLogit = Math.min(medianLogit, logit(TIER_CEIL[tier]));
      sigma = Math.min(1.3, 0.75 + 0.25 * Math.sqrt(horizonDays));
    }

    const qs = Z.map((z) => round4(clampPct(sigmoidPct(medianLogit + z * sigma))));
    // Enforce non-decreasing quantiles after rounding.
    for (let k = 1; k < qs.length; k++) if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    return { id: trophy.id, q: qs };
  });
}

/** Persistence fallback: last rate, honest wide spread. Used only on surprise. */
function fallbackCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  return c.trophies.map((trophy, i) => {
    const rawLast = lastRate(c, i);
    const last = rawLast !== null ? rawLast : TIER_DEFAULT_RATE[tierOf(trophy)];
    if (last >= 99.99) return { id: trophy.id, q: [99.95, 99.98, 99.99, 99.99, 99.99] };
    const l = logit(last);
    const qs = Z.map((z) => round4(clampPct(sigmoidPct(l + z * 1.0))));
    for (let k = 1; k < qs.length; k++) if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    return { id: trophy.id, q: qs };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
let inputs: string[] = [];
try {
  inputs = readdirSync(INPUT_DIR)
    .filter((f) => f.endsWith('.in'))
    .sort();
} catch {
  inputs = [];
}
// Every case file, in name order, so two runs write the same bytes.
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let rows: Array<{ id: number; q: number[] }>;
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
    rows = forecastCase(c);
    if (!Array.isArray(rows) || rows.length !== c.trophies.length) throw new Error('bad forecast');
  } catch {
    try {
      const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
      rows = fallbackCase(c);
    } catch {
      continue;
    }
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
