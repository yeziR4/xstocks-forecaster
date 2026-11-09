// main.ts — xstocks-movers-forecast-01: hourly-bar forecaster for the nine xStocks.
// Improvements over the house baseline (zero drift, independent normals):
//   - each name's hourly variance is shrunk toward the panel median (few bars per name make
//     single-name estimates noisy, and quiet pools understate a share's true vol);
//   - the order probabilities come from a JOINT normal with a shrunk correlation matrix, so
//     names that move together get closer-to-uniform place probabilities where the baseline's
//     independent draws overstate their separation.
// Everything is standard library and deterministic: fixed seed, no wall clock.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const SESSION_HOURS = 24;
const MIN_VOL = 0.005;
const FALLBACK_VOL = 0.02;
const HOUR_MS = 3_600_000;
const DRAWS = 20_000;
const SEED = 20_261_109;
const SMOOTHING = 0.5;
const MICRO = 1_000_000;
/** Shrinkage of a name's hourly variance toward the panel median (in return-count units). */
const VAR_SHRINK = 8;
/** Weight on the sample correlation matrix (rest goes to the identity). */
const CORR_WEIGHT = 0.55;

interface Task {
  places: number;
  output: string;
  bin_edges: number[];
  window: { days: Array<{ date: string }> };
  universe: Array<{ symbol: string; file: string }>;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const symbols = task.universe.map((u) => u.symbol);
const N = symbols.length;

function barsOf(file: string): Array<{ t: number; c: number; h: number; l: number }> {
  return readFileSync(join(INPUT_DIR, file), 'utf8')
    .split('\n')
    .filter((l) => l !== '')
    .map((l) => JSON.parse(l) as { t: number; c: string; h: string; l: string })
    .sort((a, b) => a.t - b.t)
    .map((b) => ({ t: b.t, c: Number(b.c), h: Number(b.h), l: Number(b.l) }));
}

const allBars = task.universe.map((u) => barsOf(u.file));

// Hour-to-hour log returns, pairing bars exactly an hour apart.
function hourReturns(bars: Array<{ t: number; c: number }>): Array<{ t: number; r: number }> {
  const out: Array<{ t: number; r: number }> = [];
  for (let i = 1; i < bars.length; i++) {
    if (bars[i]!.t - bars[i - 1]!.t === HOUR_MS) {
      out.push({ t: bars[i]!.t, r: Math.log(bars[i]!.c / bars[i - 1]!.c) });
    }
  }
  return out;
}

const returnsBySymbol = allBars.map(hourReturns);

// Hourly variance: close-to-close blended with the Parkinson range estimator (the bars' own
// high and low), which is more efficient on a short history than closes alone.
const rawVars = returnsBySymbol.map((rs, i) => {
  if (rs.length < 2) return null;
  const recent = rs.slice(-480);
  const ccVar = recent.reduce((s, x) => s + x.r * x.r, 0) / recent.length;
  const bars = allBars[i]!;
  let parkSum = 0;
  let parkN = 0;
  for (const b of bars) {
    const hl = Math.log(b.h / b.l);
    if (hl > 0) {
      parkSum += (hl * hl) / (4 * Math.log(2));
      parkN += 1;
    }
  }
  const parkVar = parkN > 0 ? parkSum / parkN : ccVar;
  return 0.5 * ccVar + 0.5 * parkVar;
});
const withVars = rawVars.filter((v): v is number => v !== null).sort((a, b) => a - b);
const panelVar = withVars.length
  ? withVars.length % 2 === 1
    ? withVars[(withVars.length - 1) / 2]!
    : 0.5 * (withVars[withVars.length / 2 - 1]! + withVars[withVars.length / 2]!)
  : 0.02 * 0.02 / SESSION_HOURS;

// Shrunk daily vols.
const vols = rawVars.map((v, i) => {
  const n = returnsBySymbol[i]!.length;
  if (v === null || n < 2) return FALLBACK_VOL;
  const shrunk = (n * v + VAR_SHRINK * panelVar) / (n + VAR_SHRINK);
  return Math.max(MIN_VOL, Math.sqrt(SESSION_HOURS * shrunk));
});

// Shrunk correlation matrix of the hourly returns, aligned by return timestamp.
function corrMatrix(): number[][] {
  const maps = returnsBySymbol.map((rs) => new Map(rs.map((x) => [x.t, x.r])));
  const R: number[][] = [];
  for (let a = 0; a < N; a++) {
    R.push([]);
    for (let b = 0; b < N; b++) R[a]!.push(a === b ? 1 : 0);
  }
  for (let a = 0; a < N; a++) {
    for (let b = a + 1; b < N; b++) {
      const pairs: Array<[number, number]> = [];
      for (const [t, ra] of maps[a]!) {
        const rb = maps[b]!.get(t);
        if (rb !== undefined) pairs.push([ra, rb]);
      }
      if (pairs.length < 6) continue; // too few to mean anything: leave the shrinkage to fill it
      const ma = pairs.reduce((s, p) => s + p[0], 0) / pairs.length;
      const mb = pairs.reduce((s, p) => s + p[1], 0) / pairs.length;
      let num = 0;
      let da = 0;
      let db = 0;
      for (const [x, y] of pairs) {
        num += (x - ma) * (y - mb);
        da += (x - ma) * (x - ma);
        db += (y - mb) * (y - mb);
      }
      if (da <= 0 || db <= 0) continue;
      const rho = Math.max(-0.95, Math.min(0.95, num / Math.sqrt(da * db)));
      R[a]![b] = rho;
      R[b]![a] = rho;
    }
  }
  // Convex combination with the identity: still positive semi-definite.
  return R.map((row, i) => row.map((v, j) => (i === j ? 1 : CORR_WEIGHT * v)));
}

// Cholesky factor (lower triangular) of a PSD matrix; null if it fails numerically.
function cholesky(M: number[][]): number[][] | null {
  const L: number[][] = Array.from({ length: N }, () => new Array<number>(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i]![j]!;
      for (let k = 0; k < j; k++) s -= L[i]![k]! * L[j]![k]!;
      if (i === j) {
        if (s <= 1e-12) return null;
        L[i]![j] = Math.sqrt(s);
      } else {
        L[i]![j] = s / L[j]![j]!;
      }
    }
  }
  return L;
}

const R = corrMatrix();
const L = cholesky(R) ?? cholesky(R.map((row, i) => row.map((v, j) => (i === j ? 1 : 0))))!;

function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const above = (x: number, sd: number): number => 0.5 * erfc(x / (sd * Math.SQRT2));

function binProbability(lo: number, hi: number, sd: number): number {
  if (lo < 0 && hi > 0) return 1 - above(-lo, sd) - above(hi, sd);
  if (hi <= 0) return binProbability(-hi, -lo, sd);
  return above(lo, sd) - (hi === Infinity ? 0 : above(hi, sd));
}

function uniforms(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normals(next: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u: number;
    let v: number;
    let s: number;
    do {
      u = 2 * next() - 1;
      v = 2 * next() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * f;
    return u * f;
  };
}

const micro = (p: number): number => Math.floor(p * MICRO);

// Sizes: each name's bins at its shrunk normal, analytic.
const edges = task.bin_edges;
const sizes: Array<{ symbol: string; bin: number; p: number }> = [];
symbols.forEach((symbol, i) => {
  const sd = vols[i]!;
  for (let bin = 0; bin <= edges.length; bin++) {
    const p = binProbability(bin === 0 ? -Infinity : edges[bin - 1]!, bin === edges.length ? Infinity : edges[bin]!, sd);
    const u = micro(p);
    if (u > 0) sizes.push({ symbol, bin, p: u / MICRO });
  }
});

// The order: place counts over DRAWS of the joint normal.
const gauss = normals(uniforms(SEED));
const counts: number[][] = Array.from({ length: task.places }, () => new Array<number>(N).fill(0));
const z = new Array<number>(N).fill(0);
const draw = new Array<number>(N).fill(0);
const idx = symbols.map((_, i) => i);
for (let d = 0; d < DRAWS; d++) {
  for (let i = 0; i < N; i++) z[i] = gauss();
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let j = 0; j <= i; j++) s += L[i]![j]! * z[j]!;
    draw[i] = vols[i]! * s;
  }
  idx.sort((a, b) => draw[b]! - draw[a]! || a - b);
  for (let k = 0; k < task.places; k++) counts[k]![idx[k]!]! += 1;
}
const units = counts.map((row) => row.map((c) => micro((c + SMOOTHING) / (DRAWS + SMOOTHING * N))));
const probabilities: Array<{ symbol: string; place: number; p: number }> = [];
units.forEach((row, k) => row.forEach((u, i) => probabilities.push({ symbol: symbols[i]!, place: k + 1, p: u / MICRO })));
const order: string[] = [];
for (let k = 0; k < task.places; k++) {
  let best = -1;
  for (let i = 0; i < N; i++) {
    if (order.includes(symbols[i]!)) continue;
    if (best < 0 || units[k]![i]! > units[k]![best]! || (units[k]![i] === units[k]![best] && symbols[i]! < symbols[best]!)) best = i;
  }
  order.push(symbols[best]!);
}

const days = task.window.days.map((d) => ({ date: d.date, probabilities, order, sizes }));
mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ days })}\n`);
