import { BBox, GrayBuffer, RgbaBuffer, SegmentationResult, SegmentedCharacter } from './types';

function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}

/** Resize an RGBA buffer using bilinear interpolation. */
export function resizeRgba(src: RgbaBuffer, dstW: number, dstH: number): RgbaBuffer {
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  const { width: sw, height: sh, data } = src;
  for (let y = 0; y < dstH; y++) {
    const sy = ((y + 0.5) * sh) / dstH - 0.5;
    const y0 = Math.max(0, Math.min(sh - 1, Math.floor(sy)));
    const y1 = Math.min(sh - 1, y0 + 1);
    const dy = Math.max(0, sy - y0);
    for (let x = 0; x < dstW; x++) {
      const sx = ((x + 0.5) * sw) / dstW - 0.5;
      const x0 = Math.max(0, Math.min(sw - 1, Math.floor(sx)));
      const x1 = Math.min(sw - 1, x0 + 1);
      const dx = Math.max(0, sx - x0);

      const i00 = (y0 * sw + x0) * 4;
      const i10 = (y0 * sw + x1) * 4;
      const i01 = (y1 * sw + x0) * 4;
      const i11 = (y1 * sw + x1) * 4;
      const o = (y * dstW + x) * 4;

      for (let c = 0; c < 3; c++) {
        const top = data[i00 + c] * (1 - dx) + data[i10 + c] * dx;
        const bot = data[i01 + c] * (1 - dx) + data[i11 + c] * dx;
        out[o + c] = clamp255(top * (1 - dy) + bot * dy);
      }
      out[o + 3] = 255;
    }
  }
  return { width: dstW, height: dstH, data: out };
}

function cropRgba(src: RgbaBuffer, bbox: BBox): RgbaBuffer {
  const out = new Uint8ClampedArray(bbox.w * bbox.h * 4);
  for (let y = 0; y < bbox.h; y++) {
    const sy = Math.min(src.height - 1, Math.max(0, bbox.y + y));
    for (let x = 0; x < bbox.w; x++) {
      const sx = Math.min(src.width - 1, Math.max(0, bbox.x + x));
      const si = (sy * src.width + sx) * 4;
      const di = (y * bbox.w + x) * 4;
      out[di] = src.data[si];
      out[di + 1] = src.data[si + 1];
      out[di + 2] = src.data[si + 2];
      out[di + 3] = 255;
    }
  }
  return { width: bbox.w, height: bbox.h, data: out };
}

function drawRectOnRgba(
  buf: RgbaBuffer,
  bbox: BBox,
  color: [number, number, number],
  thickness: number = 2
): void {
  const { width: w, height: h, data } = buf;
  const x0 = Math.max(0, bbox.x);
  const y0 = Math.max(0, bbox.y);
  const x1 = Math.min(w - 1, bbox.x + bbox.w - 1);
  const y1 = Math.min(h - 1, bbox.y + bbox.h - 1);

  for (let t = 0; t < thickness; t++) {
    for (let x = x0; x <= x1; x++) {
      const topY = Math.min(h - 1, y0 + t);
      const botY = Math.max(0, y1 - t);
      const ti = (topY * w + x) * 4;
      const bi = (botY * w + x) * 4;
      data[ti] = color[0];
      data[ti + 1] = color[1];
      data[ti + 2] = color[2];
      data[bi] = color[0];
      data[bi + 1] = color[1];
      data[bi + 2] = color[2];
    }
    for (let y = y0; y <= y1; y++) {
      const leftX = Math.min(w - 1, x0 + t);
      const rightX = Math.max(0, x1 - t);
      const li = (y * w + leftX) * 4;
      const ri = (y * w + rightX) * 4;
      data[li] = color[0];
      data[li + 1] = color[1];
      data[li + 2] = color[2];
      data[ri] = color[0];
      data[ri + 1] = color[1];
      data[ri + 2] = color[2];
    }
  }
}

/**
 * Localize license plate region via Sobel-X vertical stroke density +
 * sliding bright-plate window search. Falls back cleanly to full image
 * when pre-cropped plates (aspect >= 2.35) are supplied.
 */
export function localizePlate(src: RgbaBuffer): {
  crop: RgbaBuffer;
  bbox: BBox;
  usedFallback: boolean;
} {
  const { width: w, height: h, data } = src;
  const aspect = w / Math.max(1, h);

  if (aspect >= 2.35 && h <= 220) {
    return {
      crop: { width: w, height: h, data: new Uint8ClampedArray(data) },
      bbox: { x: 0, y: 0, w, h },
      usedFallback: true,
    };
  }

  // Compute grayscale + Sobel-X edge energy
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }

  const gradX = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx =
        -gray[(y - 1) * w + (x - 1)] +
        gray[(y - 1) * w + (x + 1)] -
        2 * gray[y * w + (x - 1)] +
        2 * gray[y * w + (x + 1)] -
        gray[(y + 1) * w + (x - 1)] +
        gray[(y + 1) * w + (x + 1)];
      gradX[y * w + x] = Math.abs(gx);
    }
  }

  // Row projection to find highest-energy horizontal strip (bright plate + dense vertical strokes)
  const rowScore = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let sumG = 0;
    let brightCount = 0;
    for (let x = 0; x < w; x++) {
      sumG += gradX[y * w + x];
      if (gray[y * w + x] > 120) brightCount++;
    }
    rowScore[y] = (sumG / w) * (0.4 + brightCount / w);
  }

  let bestScore = -1;
  let bestBox: BBox | null = null;
  const minH = Math.max(36, Math.floor(h * 0.18));
  const maxH = Math.min(h - 4, Math.floor(h * 0.52));

  for (let bh = minH; bh <= maxH; bh += 8) {
    const bw = Math.min(w - 8, Math.round(bh * 3.85));
    if (bw < 90) continue;
    for (let y = 4; y <= h - bh - 4; y += 6) {
      let stripEnergy = 0;
      for (let yy = y; yy < y + bh; yy++) stripEnergy += rowScore[yy];
      if (stripEnergy < bestScore * 0.35) continue;

      for (let x = 4; x <= w - bw - 4; x += 12) {
        let edgeSum = 0;
        let lumSum = 0;
        let samples = 0;
        for (let yy = y; yy < y + bh; yy += 3) {
          for (let xx = x; xx < x + bw; xx += 4) {
            edgeSum += gradX[yy * w + xx];
            lumSum += gray[yy * w + xx];
            samples++;
          }
        }
        const meanEdge = edgeSum / Math.max(1, samples);
        const meanLum = lumSum / Math.max(1, samples);
        // Plate region combines elevated luminance and dense vertical character edges
        const score = meanEdge * (0.5 + meanLum / 200);
        if (score > bestScore) {
          bestScore = score;
          bestBox = { x, y, w: bw, h: bh };
        }
      }
    }
  }

  if (!bestBox || bestScore < 12) {
    return {
      crop: { width: w, height: h, data: new Uint8ClampedArray(data) },
      bbox: { x: 0, y: 0, w, h },
      usedFallback: true,
    };
  }

  // Expand/refine plate rectangle around bestBox using bright-plate luminance mask
  const centerX = bestBox.x + Math.floor(bestBox.w / 2);
  const centerY = bestBox.y + Math.floor(bestBox.h / 2);

  // Measure background dark bumper luminance vs inner bright plate luminance
  let innerBright = 0;
  let innerCount = 0;
  for (let y = Math.max(0, centerY - 12); y < Math.min(h, centerY + 12); y++) {
    for (let x = Math.max(0, centerX - 60); x < Math.min(w, centerX + 60); x++) {
      if (gray[y * w + x] > 110) {
        innerBright += gray[y * w + x];
        innerCount++;
      }
    }
  }
  const plateLum = innerCount > 10 ? innerBright / innerCount : 175;
  const edgeThresh = plateLum * 0.56;

  // Scan up/down from centerY to find exact top & bottom plate edges
  let top = centerY;
  while (top > 2) {
    let bright = 0;
    const xStart = Math.max(0, centerX - Math.floor(bestBox.w * 0.35));
    const xEnd = Math.min(w, centerX + Math.floor(bestBox.w * 0.35));
    for (let x = xStart; x < xEnd; x++) {
      if (gray[top * w + x] > edgeThresh) bright++;
    }
    if (bright / Math.max(1, xEnd - xStart) < 0.32) break;
    top--;
  }
  let bottom = centerY;
  while (bottom < h - 3) {
    let bright = 0;
    const xStart = Math.max(0, centerX - Math.floor(bestBox.w * 0.35));
    const xEnd = Math.min(w, centerX + Math.floor(bestBox.w * 0.35));
    for (let x = xStart; x < xEnd; x++) {
      if (gray[bottom * w + x] > edgeThresh) bright++;
    }
    if (bright / Math.max(1, xEnd - xStart) < 0.32) break;
    bottom++;
  }

  // Scan left/right from centerX across the verified vertical span to capture full plate width
  const spanY0 = Math.min(bottom, top + Math.floor((bottom - top) * 0.18));
  const spanY1 = Math.max(spanY0 + 1, bottom - Math.floor((bottom - top) * 0.18));
  let left = centerX;
  while (left > 2) {
    let bright = 0;
    for (let y = spanY0; y < spanY1; y++) {
      if (gray[y * w + left] > edgeThresh) bright++;
    }
    if (bright / Math.max(1, spanY1 - spanY0) < 0.25) break;
    left--;
  }
  let right = centerX;
  while (right < w - 3) {
    let bright = 0;
    for (let y = spanY0; y < spanY1; y++) {
      if (gray[y * w + right] > edgeThresh) bright++;
    }
    if (bright / Math.max(1, spanY1 - spanY0) < 0.25) break;
    right++;
  }

  const finalW = right - left + 1;
  const finalH = bottom - top + 1;
  const useScanned = finalW > 120 && finalH > 32 && finalW / finalH >= 2.0;

  const refined: BBox = useScanned
    ? {
        x: Math.max(0, left - 6),
        y: Math.max(0, top - 4),
        w: Math.min(w - Math.max(0, left - 6), finalW + 12),
        h: Math.min(h - Math.max(0, top - 4), finalH + 8),
      }
    : {
        x: Math.max(0, bestBox.x - 4),
        y: Math.max(0, bestBox.y - 3),
        w: Math.min(w - Math.max(0, bestBox.x - 4), bestBox.w + 8),
        h: Math.min(h - Math.max(0, bestBox.y - 3), bestBox.h + 6),
      };

  return {
    crop: cropRgba(src, refined),
    bbox: refined,
    usedFallback: false,
  };
}

/** Normalize a segmented binary glyph mask onto a centered 32x32 square canvas. */
export function normalizeCharCrop32(
  binary: Uint8Array,
  binW: number,
  labels: Int32Array,
  targetLabel: number,
  bbox: BBox
): GrayBuffer {
  const size = 32;
  const out = new Uint8Array(size * size);
  const inner = 26; // 3px clean border matching canonical glyphs
  const scale = Math.min(inner / Math.max(1, bbox.w), inner / Math.max(1, bbox.h));
  const dstW = Math.max(1, Math.round(bbox.w * scale));
  const dstH = Math.max(1, Math.round(bbox.h * scale));
  const offX = Math.floor((size - dstW) / 2);
  const offY = Math.floor((size - dstH) / 2);

  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(bbox.h - 1, Math.floor(((y + 0.5) * bbox.h) / dstH));
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(bbox.w - 1, Math.floor(((x + 0.5) * bbox.w) / dstW));
      const idx = (bbox.y + sy) * binW + (bbox.x + sx);
      if (labels[idx] === targetLabel && binary[idx] > 0) {
        out[(offY + y) * size + (offX + x)] = 255;
      }
    }
  }
  return { width: size, height: size, data: out };
}

/**
 * Full License Plate Localization & Character Segmentation pipeline:
 * 1. Localize plate (or clean pre-cropped fallback)
 * 2. Normalize plate height to 96px
 * 3. Fast O(N) integral-image adaptive thresholding (sensitive to low-contrast haze!)
 * 4. 8-connected component labeling with union-find + geometry filters
 * 5. Left-to-right sorting + 32x32 character tensor normalization
 */
export function localizeAndSegmentPlate(source: RgbaBuffer): SegmentationResult {
  const tStart = performance.now();
  const loc = localizePlate(source);

  const annotatedSource: RgbaBuffer = {
    width: source.width,
    height: source.height,
    data: new Uint8ClampedArray(source.data),
  };
  if (!loc.usedFallback) {
    drawRectOnRgba(annotatedSource, loc.bbox, [6, 182, 212], 3);
  }

  // Normalize cropped plate to fixed 96px height
  const targetH = 96;
  const targetW = Math.max(180, Math.round((loc.crop.width / Math.max(1, loc.crop.height)) * targetH));
  const plateResized = resizeRgba(loc.crop, targetW, targetH);

  // Convert to grayscale + skip left blue IND/EU strip if strongly blue
  const gray = new Float32Array(targetW * targetH);
  const isBlueStrip = new Uint8Array(targetW * targetH);
  for (let i = 0; i < targetW * targetH; i++) {
    const idx = i * 4;
    const r = plateResized.data[idx];
    const g = plateResized.data[idx + 1];
    const b = plateResized.data[idx + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    // Detect left-side country band (strong blue dominance) so "IND" micro-text is excluded
    const x = i % targetW;
    if (x < targetW * 0.11 && b - r > 32 && b > g) {
      isBlueStrip[i] = 1;
    }
  }

  // Compute integral image for fast O(N) adaptive mean thresholding (window radius = 14)
  const stride = targetW + 1;
  const integral = new Float64Array((targetW + 1) * (targetH + 1));
  for (let y = 0; y < targetH; y++) {
    let rowSum = 0;
    for (let x = 0; x < targetW; x++) {
      rowSum += gray[y * targetW + x];
      integral[(y + 1) * stride + (x + 1)] = integral[y * stride + (x + 1)] + rowSum;
    }
  }

  const radius = 14;
  // Constant C=22 ensures genuine dark character contrast is required (severely fogged or
  // washed-out glare plates legitimately lose faint strokes until DCP restores contrast!)
  const C = 22;
  const binary = new Uint8Array(targetW * targetH);
  const marginY = 5;
  const marginX = 5;

  for (let y = marginY; y < targetH - marginY; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(targetH - 1, y + radius);
    for (let x = marginX; x < targetW - marginX; x++) {
      const idx = y * targetW + x;
      if (isBlueStrip[idx]) continue;

      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(targetW - 1, x + radius);
      const area = (y1 - y0 + 1) * (x1 - x0 + 1);
      const sum =
        integral[(y1 + 1) * stride + (x1 + 1)] -
        integral[y0 * stride + (x1 + 1)] -
        integral[(y1 + 1) * stride + x0] +
        integral[y0 * stride + x0];
      const localMean = sum / area;

      if (gray[idx] < localMean - C) {
        binary[idx] = 255;
      }
    }
  }

  // Mild 3x3 majority denoise to remove single-pixel rain speckles
  const cleaned = new Uint8Array(targetW * targetH);
  for (let y = 1; y < targetH - 1; y++) {
    for (let x = 1; x < targetW - 1; x++) {
      const idx = y * targetW + x;
      if (binary[idx] === 0) continue;
      let neighbors = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (binary[(y + dy) * targetW + (x + dx)] > 0) neighbors++;
        }
      }
      if (neighbors >= 4) cleaned[idx] = 255;
    }
  }

  // 8-Connected Component Labeling (two-pass with disjoint set union-find)
  const labels = new Int32Array(targetW * targetH);
  const parent: number[] = [0];
  let nextLabel = 1;

  const find = (a: number): number => {
    let root = a;
    while (parent[root] !== root) root = parent[root];
    let curr = a;
    while (curr !== root) {
      const next = parent[curr];
      parent[curr] = root;
      curr = next;
    }
    return root;
  };

  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) {
      if (ra < rb) parent[rb] = ra;
      else parent[ra] = rb;
    }
  };

  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const idx = y * targetW + x;
      if (cleaned[idx] === 0) continue;

      const nbrs: number[] = [];
      if (x > 0 && labels[idx - 1] > 0) nbrs.push(labels[idx - 1]);
      if (y > 0) {
        if (x > 0 && labels[idx - targetW - 1] > 0) nbrs.push(labels[idx - targetW - 1]);
        if (labels[idx - targetW] > 0) nbrs.push(labels[idx - targetW]);
        if (x < targetW - 1 && labels[idx - targetW + 1] > 0) nbrs.push(labels[idx - targetW + 1]);
      }

      if (nbrs.length === 0) {
        parent.push(nextLabel);
        labels[idx] = nextLabel++;
      } else {
        let minL = nbrs[0];
        for (let k = 1; k < nbrs.length; k++) {
          if (nbrs[k] < minL) minL = nbrs[k];
        }
        labels[idx] = minL;
        for (let k = 0; k < nbrs.length; k++) {
          union(minL, nbrs[k]);
        }
      }
    }
  }

  // Aggregate bounding boxes & pixel areas per root label
  const stats = new Map<
    number,
    { minX: number; minY: number; maxX: number; maxY: number; area: number }
  >();

  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const idx = y * targetW + x;
      const rawL = labels[idx];
      if (rawL === 0) continue;
      const root = find(rawL);
      labels[idx] = root;
      const st = stats.get(root);
      if (!st) {
        stats.set(root, { minX: x, minY: y, maxX: x, maxY: y, area: 1 });
      } else {
        if (x < st.minX) st.minX = x;
        if (y < st.minY) st.minY = y;
        if (x > st.maxX) st.maxX = x;
        if (y > st.maxY) st.maxY = y;
        st.area++;
      }
    }
  }

  // Geometric filtering for alphanumeric plate characters
  const candidates: Array<{ label: number; bbox: BBox; area: number }> = [];
  for (const [label, st] of stats.entries()) {
    const cw = st.maxX - st.minX + 1;
    const ch = st.maxY - st.minY + 1;
    const heightRatio = ch / targetH;
    const aspect = cw / Math.max(1, ch);
    const fillRatio = st.area / Math.max(1, cw * ch);

    if (
      heightRatio >= 0.34 &&
      heightRatio <= 0.86 &&
      aspect >= 0.14 &&
      aspect <= 1.05 &&
      st.area >= 55 &&
      fillRatio >= 0.15 &&
      fillRatio <= 0.92
    ) {
      candidates.push({
        label,
        bbox: { x: st.minX, y: st.minY, w: cw, h: ch },
        area: st.area,
      });
    }
  }

  // Suppress nested boxes & sort left-to-right by x
  candidates.sort((a, b) => a.bbox.x - b.bbox.x);
  let filtered = candidates.filter((c, idx) => {
    for (let j = 0; j < candidates.length; j++) {
      if (idx === j) continue;
      const o = candidates[j].bbox;
      if (
        c.bbox.x >= o.x &&
        c.bbox.y >= o.y &&
        c.bbox.x + c.bbox.w <= o.x + o.w &&
        c.bbox.y + c.bbox.h <= o.y + o.h
      ) {
        return false;
      }
    }
    return true;
  });

  if (filtered.length > 12) {
    filtered = filtered
      .sort((a, b) => b.area - a.area)
      .slice(0, 12)
      .sort((a, b) => a.bbox.x - b.bbox.x);
  }

  const annotatedPlate: RgbaBuffer = {
    width: targetW,
    height: targetH,
    data: new Uint8ClampedArray(plateResized.data),
  };

  const characters: SegmentedCharacter[] = filtered.map((item) => {
    drawRectOnRgba(annotatedPlate, item.bbox, [6, 182, 212], 2);
    const crop32 = normalizeCharCrop32(cleaned, targetW, labels, item.label, item.bbox);
    return { bbox: item.bbox, crop32 };
  });

  // Convert cleaned binary mask to RGBA for display
  const binaryRgba = new Uint8ClampedArray(targetW * targetH * 4);
  for (let i = 0; i < targetW * targetH; i++) {
    const v = cleaned[i];
    const o = i * 4;
    binaryRgba[o] = v;
    binaryRgba[o + 1] = v;
    binaryRgba[o + 2] = v;
    binaryRgba[o + 3] = 255;
  }

  return {
    plateBbox: loc.bbox,
    usedPrecropFallback: loc.usedFallback,
    plateCrop: plateResized,
    plateBinary: { width: targetW, height: targetH, data: binaryRgba },
    annotatedPlate,
    annotatedSource,
    characters,
    elapsedMs: Number((performance.now() - tStart).toFixed(2)),
  };
}
