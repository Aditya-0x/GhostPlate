import { dehazeImage } from './dehaze';
import { CHAR_VOCAB, renderCanonicalGlyph32 } from './glyphs';
import { localizeAndSegmentPlate } from './segment';
import { CharacterPrediction, DehazeParams, GrayBuffer, PipelineResult, RgbaBuffer } from './types';

interface ClassPrototype {
  char: string;
  intensityNorm: Float32Array;
  gradXNorm: Float32Array;
  gradYNorm: Float32Array;
  zonesNorm: Float32Array;
}

/** Extract L2-normalized spatial intensity, Sobel-X/Y gradient, and 4x4 zone pooling features. */
function extractFeatures32(crop: GrayBuffer): {
  intensityNorm: Float32Array;
  gradXNorm: Float32Array;
  gradYNorm: Float32Array;
  zonesNorm: Float32Array;
} {
  const size = 32;
  const n = size * size;
  const raw = new Float32Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) {
    const v = crop.data[i] / 255;
    raw[i] = v;
    mean += v;
  }
  mean /= n;

  // Mild 3x3 Gaussian-like smoothing to make cosine matching tolerant to 1-2px rain erosion
  const smooth = new Float32Array(n);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let acc = 0;
      let wSum = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= size) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= size) continue;
          const wt = dx === 0 && dy === 0 ? 4 : dx === 0 || dy === 0 ? 2 : 1;
          acc += raw[yy * size + xx] * wt;
          wSum += wt;
        }
      }
      smooth[y * size + x] = acc / wSum;
    }
  }

  // Zero-mean L2-normalized intensity vector
  const intensityNorm = new Float32Array(n);
  let normI = 0;
  for (let i = 0; i < n; i++) {
    const centered = smooth[i] - mean;
    intensityNorm[i] = centered;
    normI += centered * centered;
  }
  normI = Math.sqrt(normI) || 1;
  for (let i = 0; i < n; i++) intensityNorm[i] /= normI;

  // Sobel X and Y edge maps
  const gradXNorm = new Float32Array(n);
  const gradYNorm = new Float32Array(n);
  let normGx = 0;
  let normGy = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const gx =
        -smooth[(y - 1) * size + (x - 1)] +
        smooth[(y - 1) * size + (x + 1)] -
        2 * smooth[y * size + (x - 1)] +
        2 * smooth[y * size + (x + 1)] -
        smooth[(y + 1) * size + (x - 1)] +
        smooth[(y + 1) * size + (x + 1)];
      const gy =
        -smooth[(y - 1) * size + (x - 1)] -
        2 * smooth[(y - 1) * size + x] -
        smooth[(y - 1) * size + (x + 1)] +
        smooth[(y + 1) * size + (x - 1)] +
        2 * smooth[(y + 1) * size + x] +
        smooth[(y + 1) * size + (x + 1)];
      const idx = y * size + x;
      gradXNorm[idx] = gx;
      gradYNorm[idx] = gy;
      normGx += gx * gx;
      normGy += gy * gy;
    }
  }
  normGx = Math.sqrt(normGx) || 1;
  normGy = Math.sqrt(normGy) || 1;
  for (let i = 0; i < n; i++) {
    gradXNorm[i] /= normGx;
    gradYNorm[i] /= normGy;
  }

  // 4x4 spatial density pooling (16 zones)
  const zonesNorm = new Float32Array(16);
  let normZ = 0;
  for (let zy = 0; zy < 4; zy++) {
    for (let zx = 0; zx < 4; zx++) {
      let sum = 0;
      for (let y = zy * 8; y < (zy + 1) * 8; y++) {
        for (let x = zx * 8; x < (zx + 1) * 8; x++) {
          sum += raw[y * size + x];
        }
      }
      const val = sum / 64;
      zonesNorm[zy * 4 + zx] = val;
      normZ += val * val;
    }
  }
  normZ = Math.sqrt(normZ) || 1;
  for (let i = 0; i < 16; i++) zonesNorm[i] /= normZ;

  return { intensityNorm, gradXNorm, gradYNorm, zonesNorm };
}

let cachedPrototypes: ClassPrototype[] | null = null;

function getPrototypes(): ClassPrototype[] {
  if (cachedPrototypes) return cachedPrototypes;
  cachedPrototypes = Array.from(CHAR_VOCAB).map((char) => {
    const glyph = renderCanonicalGlyph32(char);
    const feats = extractFeatures32(glyph);
    return { char, ...feats };
  });
  return cachedPrototypes;
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Classify a single 32x32 normalized character tensor across all 36 classes (0-9, A-Z)
 * with calibrated softmax temperature producing realistic probabilities & top-3 candidates.
 */
export function classifyCharacter32(crop: GrayBuffer): CharacterPrediction {
  const prototypes = getPrototypes();
  const feats = extractFeatures32(crop);

  const logits = new Float32Array(prototypes.length);
  const temperature = 24.0;
  let maxLogit = -Infinity;

  for (let i = 0; i < prototypes.length; i++) {
    const p = prototypes[i];
    const sInt = dot(feats.intensityNorm, p.intensityNorm);
    const sGx = dot(feats.gradXNorm, p.gradXNorm);
    const sGy = dot(feats.gradYNorm, p.gradYNorm);
    const sZone = dot(feats.zonesNorm, p.zonesNorm);

    const combined = 0.48 * sInt + 0.20 * sGx + 0.20 * sGy + 0.12 * sZone;
    const logit = combined * temperature;
    logits[i] = logit;
    if (logit > maxLogit) maxLogit = logit;
  }

  // Numerically stable softmax
  const probs = new Float32Array(prototypes.length);
  let sumExp = 0;
  for (let i = 0; i < prototypes.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    probs[i] = e;
    sumExp += e;
  }

  const ranked: Array<{ char: string; prob: number }> = [];
  for (let i = 0; i < prototypes.length; i++) {
    probs[i] /= sumExp;
    ranked.push({ char: prototypes[i].char, prob: probs[i] });
  }
  ranked.sort((a, b) => b.prob - a.prob);

  return {
    char: ranked[0].char,
    confidence: ranked[0].prob,
    top3: ranked.slice(0, 3),
  };
}

/** Run the complete end-to-end ALPR pipeline on an input RGBA image buffer. */
export function runFullAlprPipeline(input: RgbaBuffer, dehazeParams: DehazeParams): PipelineResult {
  const t0 = performance.now();

  // Stage 1: Dark-Channel Prior De-Hazing & Glare/Rain Mitigation
  const dehaze = dehazeImage(input, dehazeParams);

  // Stage 2: Plate Localization & Character Segmentation
  const segmentation = localizeAndSegmentPlate(dehaze.restored);

  // Stage 3: 36-Class Character Recognition
  const tRec = performance.now();
  const predictions = segmentation.characters.map((seg) => classifyCharacter32(seg.crop32));
  const recognitionLatencyMs = Number((performance.now() - tRec).toFixed(2));

  const predictedPlate = predictions.map((p) => p.char).join('');
  const meanConfidence =
    predictions.length > 0
      ? predictions.reduce((acc, p) => acc + p.confidence, 0) / predictions.length
      : 0;

  return {
    input,
    dehaze,
    segmentation,
    predictions,
    predictedPlate,
    meanConfidence,
    recognitionLatencyMs,
    totalLatencyMs: Number((performance.now() - t0).toFixed(2)),
  };
}
