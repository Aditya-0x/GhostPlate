import { DEFAULT_DEHAZE_PARAMS } from './dehaze';
import { CHAR_VOCAB } from './glyphs';
import { runFullAlprPipeline } from './recognizer';
import { AblationReport, AblationVisualRow, ConfusionPair, WeatherAblationStats } from './types';
import { applyWeather, generateCleanPlateBuffer } from './weather';

const CHAR_INDEX: Record<string, number> = Object.fromEntries(
  Array.from(CHAR_VOCAB).map((ch, idx) => [ch, idx])
);

const BENCHMARK_PLATES: Record<'rain' | 'fog' | 'glare', Array<{ text: string; severity: number; seed: number }>> = {
  fog: [
    { text: 'DL01AB4829', severity: 0.80, seed: 1001 },
    { text: 'MH14CP7265', severity: 0.78, seed: 1002 },
    { text: 'KA05HN3910', severity: 0.82, seed: 1003 },
    { text: 'TN09BQ6482', severity: 0.79, seed: 1004 },
    { text: 'UP32KL8157', severity: 0.81, seed: 1005 },
    { text: 'HR26DK9034', severity: 0.77, seed: 1006 },
  ],
  rain: [
    { text: 'MH12DE7391', severity: 0.82, seed: 2001 },
    { text: 'PB10CX9184', severity: 0.80, seed: 2002 },
    { text: 'GJ01RW5620', severity: 0.84, seed: 2003 },
    { text: 'RJ14VM2749', severity: 0.78, seed: 2004 },
    { text: 'WB02AF6038', severity: 0.81, seed: 2005 },
    { text: 'AP28BT4519', severity: 0.79, seed: 2006 },
  ],
  glare: [
    { text: 'KA03MG5042', severity: 0.84, seed: 3001 },
    { text: 'DL08CR6193', severity: 0.82, seed: 3002 },
    { text: 'KL07BZ8426', severity: 0.80, seed: 3003 },
    { text: 'TS09EU1758', severity: 0.85, seed: 3004 },
    { text: 'CH01AN9204', severity: 0.81, seed: 3005 },
    { text: 'MP04HC3871', severity: 0.83, seed: 3006 },
  ],
};

function createEmptyMatrix36(): number[][] {
  return Array.from({ length: 36 }, () => new Array<number>(36).fill(0));
}

/**
 * Compute optimal sequence alignment (LCS-like greedy/DP alignment) between
 * ground-truth plate text and predicted string so missing/extra segments
 * don't unfairly shift all downstream character indices.
 */
function alignCharacters(
  gt: string,
  pred: string
): { matches: number; pairs: Array<{ gtIdx: number; predIdx: number }> } {
  const m = gt.length;
  const n = pred.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array<number>(n + 1).fill(0));

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (gt[i - 1] === pred[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  const matches = dp[m][n];
  const pairs: Array<{ gtIdx: number; predIdx: number }> = [];

  // Map each ground-truth character to closest predicted character for confusion matrix
  for (let i = 0; i < m; i++) {
    const gtIdx = CHAR_INDEX[gt[i]];
    if (gtIdx === undefined) continue;
    if (i < n && CHAR_INDEX[pred[i]] !== undefined) {
      pairs.push({ gtIdx, predIdx: CHAR_INDEX[pred[i]] });
    } else if (pred.length > 0 && CHAR_INDEX[pred[Math.min(i, pred.length - 1)]] !== undefined) {
      pairs.push({ gtIdx, predIdx: CHAR_INDEX[pred[Math.min(i, pred.length - 1)]] });
    } else {
      // Unsegmented character falls into nearest visual neighbor to mark failure
      pairs.push({ gtIdx, predIdx: (gtIdx + 1) % 36 });
    }
  }

  return { matches, pairs };
}

function extractTopConfusions(matrix: number[][], limit: number = 6): ConfusionPair[] {
  const list: ConfusionPair[] = [];
  for (let r = 0; r < 36; r++) {
    for (let c = 0; c < 36; c++) {
      if (r !== c && matrix[r][c] > 0) {
        list.push({
          truth: CHAR_VOCAB[r],
          predicted: CHAR_VOCAB[c],
          count: matrix[r][c],
        });
      }
    }
  }
  list.sort((a, b) => b.count - a.count);
  return list.slice(0, limit);
}

/**
 * Run a live, deterministic Ablation Study evaluating WITH vs WITHOUT
 * Dark-Channel Prior De-Hazing across Rain, Fog, and Glare conditions.
 */
export function runLiveAblationBenchmark(): AblationReport {
  const weathers: Array<{ key: 'fog' | 'rain' | 'glare'; label: string }> = [
    { key: 'fog', label: 'Dense Fog' },
    { key: 'rain', label: 'Heavy Rain' },
    { key: 'glare', label: 'Headlight Glare' },
  ];

  const cmWith = createEmptyMatrix36();
  const cmWithout = createEmptyMatrix36();

  const perWeather: WeatherAblationStats[] = [];
  const visualSamples: AblationVisualRow[] = [];

  let grandCorrectNo = 0;
  let grandCorrectYes = 0;
  let grandChars = 0;
  let grandLatencyNo = 0;
  let grandLatencyYes = 0;
  let grandPlates = 0;

  for (const wMeta of weathers) {
    const items = BENCHMARK_PLATES[wMeta.key];
    let correctNo = 0;
    let correctYes = 0;
    let totalChars = 0;
    let latencyNoSum = 0;
    let latencyYesSum = 0;

    items.forEach((item, idx) => {
      const clean = generateCleanPlateBuffer(item.text, false, item.seed);
      const degraded = applyWeather(clean, {
        mode: wMeta.key,
        severity: item.severity,
        seed: item.seed + 50,
      });

      // Path A: WITHOUT Dark-Channel Pre-Filter
      const resNo = runFullAlprPipeline(degraded, { ...DEFAULT_DEHAZE_PARAMS, enabled: false });
      const alignNo = alignCharacters(item.text, resNo.predictedPlate);
      correctNo += alignNo.matches;
      latencyNoSum += resNo.totalLatencyMs;
      for (const p of alignNo.pairs) cmWithout[p.gtIdx][p.predIdx]++;

      // Path B: WITH Dark-Channel De-Hazing + Glare Mitigation
      const resYes = runFullAlprPipeline(degraded, { ...DEFAULT_DEHAZE_PARAMS, enabled: true });
      const alignYes = alignCharacters(item.text, resYes.predictedPlate);
      correctYes += alignYes.matches;
      latencyYesSum += resYes.totalLatencyMs;
      for (const p of alignYes.pairs) cmWith[p.gtIdx][p.predIdx]++;

      totalChars += item.text.length;
      grandPlates++;

      if (idx === 0) {
        visualSamples.push({
          weather: wMeta.key,
          groundTruth: item.text,
          predWithoutDcp: resNo.predictedPlate || '—',
          predWithDcp: resYes.predictedPlate || '—',
          confWithoutDcp: resNo.meanConfidence,
          confWithDcp: resYes.meanConfidence,
          original: degraded,
          dehazed: resYes.dehaze.restored,
          annotated: resYes.segmentation.annotatedPlate,
          charCrops: resYes.segmentation.characters.map((c) => c.crop32),
        });
      }
    });

    const accNo = (100 * correctNo) / Math.max(1, totalChars);
    const accYes = (100 * correctYes) / Math.max(1, totalChars);

    perWeather.push({
      weather: wMeta.key,
      label: wMeta.label,
      withoutDcpAcc: Number(accNo.toFixed(1)),
      withDcpAcc: Number(accYes.toFixed(1)),
      gainPct: Number((accYes - accNo).toFixed(1)),
      withoutDcpLatencyMs: Number((latencyNoSum / items.length).toFixed(1)),
      withDcpLatencyMs: Number((latencyYesSum / items.length).toFixed(1)),
      totalChars,
    });

    grandCorrectNo += correctNo;
    grandCorrectYes += correctYes;
    grandChars += totalChars;
    grandLatencyNo += latencyNoSum;
    grandLatencyYes += latencyYesSum;
  }

  const overallNo = (100 * grandCorrectNo) / Math.max(1, grandChars);
  const overallYes = (100 * grandCorrectYes) / Math.max(1, grandChars);

  return {
    overallWithoutDcpAcc: Number(overallNo.toFixed(1)),
    overallWithDcpAcc: Number(overallYes.toFixed(1)),
    accuracyGainPct: Number((overallYes - overallNo).toFixed(1)),
    overallWithoutDcpLatencyMs: Number((grandLatencyNo / Math.max(1, grandPlates)).toFixed(1)),
    overallWithDcpLatencyMs: Number((grandLatencyYes / Math.max(1, grandPlates)).toFixed(1)),
    perWeather,
    confusionMatrix: cmWith,
    confusionMatrixWithoutDcp: cmWithout,
    topConfusionsWithoutDcp: extractTopConfusions(cmWithout),
    topConfusionsWithDcp: extractTopConfusions(cmWith),
    visualSamples,
    totalPlatesEvaluated: grandPlates,
    totalCharsEvaluated: grandChars,
  };
}
