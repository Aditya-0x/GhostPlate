import { DehazeParams, DehazeResult, RgbaBuffer } from './types';

export const DEFAULT_DEHAZE_PARAMS: DehazeParams = {
  enabled: true,
  omega: 0.92,
  patchSize: 13,
  tMin: 0.14,
  guidedRadius: 16,
  guidedEps: 1e-3,
  glareSuppression: true,
  claheClipLimit: 2.6,
};

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}

/** Compute RMS grayscale luminance contrast (standard deviation in 0..255). */
export function computeRmsContrast(buf: RgbaBuffer): number {
  const n = buf.width * buf.height;
  if (n === 0) return 0;
  let sum = 0;
  let sumSq = 0;
  const d = buf.data;
  for (let i = 0; i < d.length; i += 4) {
    const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    sum += lum;
    sumSq += lum * lum;
  }
  const mean = sum / n;
  return Math.sqrt(Math.max(0, sumSq / n - mean * mean));
}

/** Fast separable 2D minimum filter (horizontal + vertical sliding window). */
function minFilter2D(src: Float32Array, w: number, h: number, patchSize: number): Float32Array {
  const r = Math.max(1, Math.floor(patchSize / 2));
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let m = 1e9;
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      for (let k = x0; k <= x1; k++) {
        const val = src[row + k];
        if (val < m) m = val;
      }
      tmp[row + x] = m;
    }
  }

  // Vertical pass
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      let m = 1e9;
      for (let k = y0; k <= y1; k++) {
        const val = tmp[k * w + x];
        if (val < m) m = val;
      }
      out[y * w + x] = m;
    }
  }
  return out;
}

/** O(N) Box filter using 2D integral image (summed area table). */
function boxFilterIntegral(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  const stride = w + 1;
  const integral = new Float64Array((w + 1) * (h + 1));

  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    const srcOff = y * w;
    const curOff = (y + 1) * stride;
    const prevOff = y * stride;
    for (let x = 0; x < w; x++) {
      rowSum += src[srcOff + x];
      integral[curOff + x + 1] = integral[prevOff + x + 1] + rowSum;
    }
  }

  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(h - 1, y + radius);
    const r0 = y0 * stride;
    const r1 = (y1 + 1) * stride;
    const outOff = y * w;
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(w - 1, x + radius);
      const area = (y1 - y0 + 1) * (x1 - x0 + 1);
      const sum = integral[r1 + x1 + 1] - integral[r0 + x1 + 1] - integral[r1 + x0] + integral[r0 + x0];
      out[outOff + x] = sum / area;
    }
  }
  return out;
}

/**
 * Fast O(N) Guided Filter (He et al., ECCV 2010).
 * Refines coarse transmission map along grayscale luminance edges.
 */
export function guidedFilter(
  guide: Float32Array,
  src: Float32Array,
  w: number,
  h: number,
  radius: number,
  eps: number
): Float32Array {
  const n = w * h;
  const prodIp = new Float32Array(n);
  const prodII = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    prodIp[i] = guide[i] * src[i];
    prodII[i] = guide[i] * guide[i];
  }

  const meanI = boxFilterIntegral(guide, w, h, radius);
  const meanP = boxFilterIntegral(src, w, h, radius);
  const meanIp = boxFilterIntegral(prodIp, w, h, radius);
  const meanII = boxFilterIntegral(prodII, w, h, radius);

  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const covIp = meanIp[i] - meanI[i] * meanP[i];
    const varI = meanII[i] - meanI[i] * meanI[i];
    a[i] = covIp / (varI + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }

  const meanA = boxFilterIntegral(a, w, h, radius);
  const meanB = boxFilterIntegral(b, w, h, radius);

  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = clamp01(meanA[i] * guide[i] + meanB[i]);
  }
  return out;
}

/** Tile-interpolated CLAHE on an 8-bit luminance channel (L-channel of CIE-LAB). */
function applyClaheChannel(
  channel: Uint8Array,
  w: number,
  h: number,
  clipLimit: number,
  gridX: number = 6,
  gridY: number = 3
): Uint8Array {
  const tilesX = Math.max(1, gridX);
  const tilesY = Math.max(1, gridY);
  const tileW = w / tilesX;
  const tileH = h / tilesY;
  const bins = 256;

  // Build clipped cumulative distribution LUT for each tile
  const luts: Uint8Array[] = new Array(tilesX * tilesY);

  for (let ty = 0; ty < tilesY; ty++) {
    const yStart = Math.floor(ty * tileH);
    const yEnd = Math.min(h, Math.floor((ty + 1) * tileH));
    for (let tx = 0; tx < tilesX; tx++) {
      const xStart = Math.floor(tx * tileW);
      const xEnd = Math.min(w, Math.floor((tx + 1) * tileW));

      const hist = new Uint32Array(bins);
      let count = 0;
      for (let y = yStart; y < yEnd; y++) {
        const row = y * w;
        for (let x = xStart; x < xEnd; x++) {
          hist[channel[row + x]]++;
          count++;
        }
      }

      if (count === 0) {
        const identity = new Uint8Array(bins);
        for (let i = 0; i < bins; i++) identity[i] = i;
        luts[ty * tilesX + tx] = identity;
        continue;
      }

      const limit = Math.max(1, Math.floor((clipLimit * count) / bins));
      let excess = 0;
      for (let i = 0; i < bins; i++) {
        if (hist[i] > limit) {
          excess += hist[i] - limit;
          hist[i] = limit;
        }
      }

      const redist = Math.floor(excess / bins);
      let rem = excess - redist * bins;
      for (let i = 0; i < bins; i++) {
        hist[i] += redist;
        if (rem > 0) {
          hist[i]++;
          rem--;
        }
      }

      const lut = new Uint8Array(bins);
      let cdf = 0;
      for (let i = 0; i < bins; i++) {
        cdf += hist[i];
        lut[i] = clamp255((cdf / count) * 255);
      }
      luts[ty * tilesX + tx] = lut;
    }
  }

  // Bilinearly interpolate LUT lookups across neighboring tile centers
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = (y + 0.5) / tileH - 0.5;
    const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(fy)));
    const ty1 = Math.min(tilesY - 1, ty0 + 1);
    const wy = clamp01(fy - ty0);

    for (let x = 0; x < w; x++) {
      const fx = (x + 0.5) / tileW - 0.5;
      const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(fx)));
      const tx1 = Math.min(tilesX - 1, tx0 + 1);
      const wx = clamp01(fx - tx0);

      const v = channel[y * w + x];
      const v00 = luts[ty0 * tilesX + tx0][v];
      const v10 = luts[ty0 * tilesX + tx1][v];
      const v01 = luts[ty1 * tilesX + tx0][v];
      const v11 = luts[ty1 * tilesX + tx1][v];

      const top = v00 * (1 - wx) + v10 * wx;
      const bot = v01 * (1 - wx) + v11 * wx;
      out[y * w + x] = clamp255(top * (1 - wy) + bot * wy);
    }
  }
  return out;
}

/** Fast edge-preserving 3x3 bilateral filter to suppress rain streaks without blurring character edges. */
function bilateralFilter3x3(buf: RgbaBuffer, sigmaColor: number = 32): RgbaBuffer {
  const { width: w, height: h, data } = buf;
  const out = new Uint8ClampedArray(w * h * 4);
  const invTwoSigmaSq = 1 / (2 * sigmaColor * sigmaColor);

  // Spatial weights for 3x3 neighborhood
  const spatial = [0.6, 0.85, 0.6, 0.85, 1.0, 0.85, 0.6, 0.85, 0.6];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const cIdx = (y * w + x) * 4;
      const cr = data[cIdx];
      const cg = data[cIdx + 1];
      const cb = data[cIdx + 2];

      let sumR = 0;
      let sumG = 0;
      let sumB = 0;
      let sumW = 0;
      let k = 0;

      for (let dy = -1; dy <= 1; dy++) {
        const yy = Math.max(0, Math.min(h - 1, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const xx = Math.max(0, Math.min(w - 1, x + dx));
          const nIdx = (yy * w + xx) * 4;
          const nr = data[nIdx];
          const ng = data[nIdx + 1];
          const nb = data[nIdx + 2];
          const diffSq = (nr - cr) * (nr - cr) + (ng - cg) * (ng - cg) + (nb - cb) * (nb - cb);
          const weight = spatial[k++] * Math.exp(-diffSq * invTwoSigmaSq);
          sumR += nr * weight;
          sumG += ng * weight;
          sumB += nb * weight;
          sumW += weight;
        }
      }

      out[cIdx] = clamp255(sumR / sumW);
      out[cIdx + 1] = clamp255(sumG / sumW);
      out[cIdx + 2] = clamp255(sumB / sumW);
      out[cIdx + 3] = 255;
    }
  }
  return { width: w, height: h, data: out };
}

/** Ocean-inspired scientific colormap for refined transmission map visualization. */
function scalarToOceanRgba(map: Float32Array, w: number, h: number): RgbaBuffer {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = clamp01(map[i]);
    // Deep navy -> electric cyan -> pale ice white
    let r = 0;
    let g = 0;
    let b = 0;
    if (v < 0.5) {
      const t = v * 2;
      r = 8 + t * 8;
      g = 22 + t * 145;
      b = 54 + t * 165;
    } else {
      const t = (v - 0.5) * 2;
      r = 16 + t * 215;
      g = 167 + t * 78;
      b = 219 + t * 34;
    }
    const idx = i * 4;
    data[idx] = clamp255(r);
    data[idx + 1] = clamp255(g);
    data[idx + 2] = clamp255(b);
    data[idx + 3] = 255;
  }
  return { width: w, height: h, data };
}

function scalarToGrayRgba(map: Float32Array, w: number, h: number): RgbaBuffer {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = clamp255(clamp01(map[i]) * 255);
    const idx = i * 4;
    data[idx] = v;
    data[idx + 1] = v;
    data[idx + 2] = v;
    data[idx + 3] = 255;
  }
  return { width: w, height: h, data };
}

/**
 * Full Dark-Channel Prior De-Hazing + Glare/Rain Mitigation Pipeline.
 */
export function dehazeImage(input: RgbaBuffer, params: DehazeParams): DehazeResult {
  const tStart = performance.now();
  const { width: w, height: h, data } = input;
  const n = w * h;
  const contrastBefore = computeRmsContrast(input);

  // If DCP pre-filter is toggled OFF, return pass-through copy cleanly
  if (!params.enabled) {
    const zeros = new Float32Array(n);
    const roundedContrast = Number(contrastBefore.toFixed(1));
    return {
      restored: { width: w, height: h, data: new Uint8ClampedArray(data) },
      darkChannelVis: scalarToGrayRgba(zeros, w, h),
      transmissionVis: scalarToOceanRgba(zeros, w, h),
      atmosphericLight: [0, 0, 0],
      gammaApplied: 1.0,
      contrastBefore: roundedContrast,
      contrastAfter: roundedContrast,
      elapsedMs: Number(Math.max(0.05, performance.now() - tStart).toFixed(2)),
    };
  }

  // Convert input RGBA to float RGB [0, 1], grayscale guide, and per-pixel channel min
  const rArr = new Float32Array(n);
  const gArr = new Float32Array(n);
  const bArr = new Float32Array(n);
  const grayGuide = new Float32Array(n);
  const minRgb = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    const r = data[idx] / 255;
    const g = data[idx + 1] / 255;
    const b = data[idx + 2] / 255;
    rArr[i] = r;
    gArr[i] = g;
    bArr[i] = b;
    grayGuide[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    minRgb[i] = r < g ? (r < b ? r : b) : g < b ? g : b;
  }

  // 1. Dark Channel Extraction
  const darkChannel = minFilter2D(minRgb, w, h, params.patchSize);

  // 2. Atmospheric Light Estimation (top 0.1% brightest pixels in dark channel)
  const topCount = Math.max(1, Math.floor(n * 0.001));
  const sortedDark = Float32Array.from(darkChannel).sort();
  const thresholdDark = sortedDark[Math.max(0, n - topCount)];

  const candidates: Array<{ lum: number; idx: number }> = [];
  for (let i = 0; i < n; i++) {
    if (darkChannel[i] >= thresholdDark) {
      candidates.push({ lum: grayGuide[i], idx: i });
    }
  }
  candidates.sort((a, b) => b.lum - a.lum);
  const bestK = Math.max(1, Math.min(20, candidates.length));

  let aR = 0;
  let aG = 0;
  let aB = 0;
  for (let k = 0; k < bestK; k++) {
    const pIdx = candidates[k].idx;
    aR += rArr[pIdx];
    aG += gArr[pIdx];
    aB += bArr[pIdx];
  }
  const cap = params.airlightCap ?? 0.96;
  aR = Math.max(0.18, Math.min(cap, aR / bestK));
  aG = Math.max(0.18, Math.min(cap, aG / bestK));
  aB = Math.max(0.18, Math.min(cap, aB / bestK));

  // 3. Initial Coarse Transmission Map
  const normMin = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const nr = rArr[i] / aR;
    const ng = gArr[i] / aG;
    const nb = bArr[i] / aB;
    normMin[i] = nr < ng ? (nr < nb ? nr : nb) : ng < nb ? ng : nb;
  }
  const darkNorm = minFilter2D(normMin, w, h, params.patchSize);
  const rawTransmission = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    rawTransmission[i] = Math.max(0.05, Math.min(1.0, 1.0 - params.omega * darkNorm[i]));
  }

  // 4. Guided Filter Transmission Refinement
  const refinedTransmission = guidedFilter(
    grayGuide,
    rawTransmission,
    w,
    h,
    params.guidedRadius,
    params.guidedEps
  );

  // 5. Scene Radiance Recovery: J(x) = (I(x) - A) / max(t(x), tMin) + A
  const dcpData = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const t = Math.max(params.tMin, refinedTransmission[i]);
    const jr = clamp01((rArr[i] - aR) / t + aR);
    const jg = clamp01((gArr[i] - aG) / t + aG);
    const jb = clamp01((bArr[i] - aB) / t + aB);
    const idx = i * 4;
    dcpData[idx] = clamp255(jr * 255);
    dcpData[idx + 1] = clamp255(jg * 255);
    dcpData[idx + 2] = clamp255(jb * 255);
    dcpData[idx + 3] = 255;
  }

  let restored: RgbaBuffer = { width: w, height: h, data: dcpData };
  let gammaApplied = 1.0;

  // 6. Glare & Rain Mitigation Stage (Adaptive Gamma + L-Channel CLAHE + Bilateral)
  if (params.glareSuppression) {
    let meanLum = 0;
    let highlightPixels = 0;
    for (let i = 0; i < n; i++) {
      const idx = i * 4;
      const lum = 0.299 * dcpData[idx] + 0.587 * dcpData[idx + 1] + 0.114 * dcpData[idx + 2];
      meanLum += lum;
      if (lum > 205) highlightPixels++;
    }
    meanLum /= n * 255;
    const highlightRatio = highlightPixels / n;

    if (highlightRatio > 0.12 || meanLum > 0.58) {
      gammaApplied = Math.max(
        1.15,
        Math.min(2.6, 1.0 + 1.9 * (meanLum - 0.44) + 0.95 * highlightRatio)
      );
    } else if (meanLum < 0.34) {
      gammaApplied = Math.max(0.62, Math.min(0.95, 0.65 + 0.7 * meanLum));
    }

    const gammaLut = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      gammaLut[i] = clamp255(Math.pow(i / 255, gammaApplied) * 255);
    }

    // Apply Adaptive Gamma + Luminance-channel CLAHE (stretches glare/fog micro-contrast)
    const lumChan = new Uint8Array(n);
    const gammaRgb = new Uint8ClampedArray(n * 4);
    for (let i = 0; i < n; i++) {
      const idx = i * 4;
      const r = gammaLut[dcpData[idx]];
      const g = gammaLut[dcpData[idx + 1]];
      const b = gammaLut[dcpData[idx + 2]];
      gammaRgb[idx] = r;
      gammaRgb[idx + 1] = g;
      gammaRgb[idx + 2] = b;
      gammaRgb[idx + 3] = 255;
      lumChan[i] = clamp255(0.299 * r + 0.587 * g + 0.114 * b);
    }

    const claheLum = applyClaheChannel(lumChan, w, h, params.claheClipLimit, 6, 3);
    const enhancedData = new Uint8ClampedArray(n * 4);
    for (let i = 0; i < n; i++) {
      const idx = i * 4;
      const oldL = Math.max(1, lumChan[i]);
      const newL = claheLum[i];
      // Preserve chromaticity ratios cleanly while locking luminance to CLAHE L-channel
      const ratio = newL / oldL;
      const nr = clamp255(gammaRgb[idx] * ratio);
      const ng = clamp255(gammaRgb[idx + 1] * ratio);
      const nb = clamp255(gammaRgb[idx + 2] * ratio);
      enhancedData[idx] = nr;
      enhancedData[idx + 1] = ng;
      enhancedData[idx + 2] = nb;
      enhancedData[idx + 3] = 255;
    }

    // Suppress narrow 1–2px bright raindrop streaks slicing across dark character strokes
    const streakSuppressed = new Uint8ClampedArray(enhancedData);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 2; x < w - 2; x++) {
        const c = (y * w + x) * 4;
        const lumC = 0.299 * enhancedData[c] + 0.587 * enhancedData[c + 1] + 0.114 * enhancedData[c + 2];
        const l2 = (y * w + (x - 2)) * 4;
        const r2 = (y * w + (x + 2)) * 4;
        const lumL = 0.299 * enhancedData[l2] + 0.587 * enhancedData[l2 + 1] + 0.114 * enhancedData[l2 + 2];
        const lumR = 0.299 * enhancedData[r2] + 0.587 * enhancedData[r2 + 1] + 0.114 * enhancedData[r2 + 2];
        // Bright 1-2px vertical streak bracketed horizontally by dark character ink
        if (lumL < 110 && lumR < 110 && lumC > Math.max(lumL, lumR) + 45) {
          streakSuppressed[c] = (enhancedData[l2] + enhancedData[r2]) >> 1;
          streakSuppressed[c + 1] = (enhancedData[l2 + 1] + enhancedData[r2 + 1]) >> 1;
          streakSuppressed[c + 2] = (enhancedData[l2 + 2] + enhancedData[r2 + 2]) >> 1;
        }
      }
    }

    // Edge-preserving bilateral filter to suppress rain streaks without blurring characters
    restored = bilateralFilter3x3({ width: w, height: h, data: streakSuppressed }, 36);
  }

  return {
    restored,
    darkChannelVis: scalarToGrayRgba(darkChannel, w, h),
    transmissionVis: scalarToOceanRgba(refinedTransmission, w, h),
    atmosphericLight: [Number(aR.toFixed(3)), Number(aG.toFixed(3)), Number(aB.toFixed(3))],
    gammaApplied: Number(gammaApplied.toFixed(2)),
    contrastBefore: Number(contrastBefore.toFixed(1)),
    contrastAfter: Number(computeRmsContrast(restored).toFixed(1)),
    elapsedMs: Number((performance.now() - tStart).toFixed(2)),
  };
}
