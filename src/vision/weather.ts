import { drawGlyphOnRgba } from './glyphs';
import { RgbaBuffer, SamplePreset, WeatherConfig } from './types';

/** Deterministic PRNG (Mulberry32) so every sample and benchmark is 100% reproducible. */
export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const SAMPLE_PRESETS: SamplePreset[] = [
  {
    id: 'fog-dl01',
    label: 'Dense Fog — DL01AB4829',
    plateText: 'DL01AB4829',
    weather: 'fog',
    severity: 0.78,
    seed: 101,
    embedInScene: false,
    description: 'Heavy atmospheric Koschmieder scattering reducing RMS plate contrast by >70%.',
  },
  {
    id: 'rain-mh12',
    label: 'Heavy Rain — MH12DE7391',
    plateText: 'MH12DE7391',
    weather: 'rain',
    severity: 0.76,
    seed: 202,
    embedInScene: false,
    description: 'Slanted high-density rain streaks combined with wet mist veil across character strokes.',
  },
  {
    id: 'glare-ka03',
    label: 'Headlight Glare — KA03MG5042',
    plateText: 'KA03MG5042',
    weather: 'glare',
    severity: 0.80,
    seed: 303,
    embedInScene: false,
    description: 'Nighttime off-axis halogen headlight bloom washing out central characters.',
  },
  {
    id: 'mixed-pb10',
    label: 'Mixed Storm — PB10CX9184',
    plateText: 'PB10CX9184',
    weather: 'mixed',
    severity: 0.72,
    seed: 404,
    embedInScene: false,
    description: 'Compound degradation combining fog haze, rain streaks, and localized glare.',
  },
  {
    id: 'scene-gj05',
    label: 'Vehicle Bumper Scene — GJ05KR8316',
    plateText: 'GJ05KR8316',
    weather: 'fog',
    severity: 0.72,
    seed: 505,
    embedInScene: true,
    description: 'Full vehicle rear scene testing morphological plate localization before character segmentation.',
  },
];

function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}

/**
 * Render a realistic license plate (with metallic off-white reflective surface,
 * dark border, IND blue strip, subtle plate rivets, and crisp embossed characters).
 * Optionally embeds the plate onto a dark vehicle bumper scene for localization testing.
 */
export function generateCleanPlateBuffer(
  plateText: string,
  embedInScene: boolean = false,
  seed: number = 42
): RgbaBuffer {
  const rng = createSeededRng(seed);
  const pw = 440;
  const ph = 112;
  const plateData = new Uint8ClampedArray(pw * ph * 4);

  const baseR = 236 + Math.round((rng() - 0.5) * 10);
  const baseG = 238 + Math.round((rng() - 0.5) * 10);
  const baseB = 240 + Math.round((rng() - 0.5) * 10);

  // Fill plate aluminum surface with subtle vertical metallic sheen
  for (let y = 0; y < ph; y++) {
    const sheen = 1.0 - 0.045 * Math.abs(y / ph - 0.35);
    for (let x = 0; x < pw; x++) {
      const idx = (y * pw + x) * 4;
      const isBorder = x < 4 || x >= pw - 4 || y < 4 || y >= ph - 4;
      const isBlueBand = !isBorder && x >= 5 && x < 34;

      if (isBorder) {
        plateData[idx] = 30;
        plateData[idx + 1] = 34;
        plateData[idx + 2] = 42;
      } else if (isBlueBand) {
        plateData[idx] = 26;
        plateData[idx + 1] = 84;
        plateData[idx + 2] = 166;
      } else {
        plateData[idx] = clamp255(baseR * sheen);
        plateData[idx + 1] = clamp255(baseG * sheen);
        plateData[idx + 2] = clamp255(baseB * sheen);
      }
      plateData[idx + 3] = 255;
    }
  }

  // Draw small "IND" tag on the blue strip
  const indStr = 'IND';
  for (let i = 0; i < indStr.length; i++) {
    drawGlyphOnRgba(plateData, pw, ph, indStr[i], 7 + i * 8, ph - 22, 6, 10, [230, 238, 248]);
  }

  // Draw main plate characters centered in the white field
  const cleanText = plateText.toUpperCase().replace(/[^0-9A-Z]/g, '');
  const count = Math.max(cleanText.length, 1);
  const marginLeft = 44;
  const marginRight = 14;
  const usableW = pw - marginLeft - marginRight;
  const slotW = usableW / count;
  const glyphW = Math.min(slotW * 0.78, 34);
  const glyphH = Math.min(ph * 0.62, glyphW * (11 / 7));
  const yTop = (ph - glyphH) / 2;

  for (let i = 0; i < cleanText.length; i++) {
    const cx = marginLeft + i * slotW + (slotW - glyphW) * 0.5;
    drawGlyphOnRgba(plateData, pw, ph, cleanText[i], cx, yTop, glyphW, glyphH, [18, 20, 26]);
  }

  if (!embedInScene) {
    return { width: pw, height: ph, data: plateData };
  }

  // Embed plate into a 560x280 dark vehicle bumper scene
  const sw = 560;
  const sh = 280;
  const sceneData = new Uint8ClampedArray(sw * sh * 4);

  for (let y = 0; y < sh; y++) {
    const ny = y / sh;
    // Metallic dark slate car trunk & bumper contour shading
    const bumperBand = ny > 0.72 ? 0.72 : ny < 0.16 ? 0.85 : 1.0;
    const spec = Math.exp(-Math.pow((ny - 0.25) / 0.12, 2)) * 22;
    for (let x = 0; x < sw; x++) {
      const nx = x / sw;
      const idx = (y * sw + x) * 4;
      const vignette = 1 - 0.28 * Math.pow(nx - 0.5, 2) * 4;
      let r = (34 + spec) * bumperBand * vignette;
      let g = (40 + spec) * bumperBand * vignette;
      let b = (50 + spec * 1.1) * bumperBand * vignette;

      // Subtle red tail-light reflections on far left & right upper corners
      const leftTail = Math.exp(-(Math.pow((nx - 0.06) / 0.08, 2) + Math.pow((ny - 0.24) / 0.12, 2)));
      const rightTail = Math.exp(-(Math.pow((nx - 0.94) / 0.08, 2) + Math.pow((ny - 0.24) / 0.12, 2)));
      const tail = leftTail + rightTail;
      r += tail * 145;
      g += tail * 22;
      b += tail * 18;

      // Recessed dark license plate bezel behind the plate
      if (x > 96 && x < 464 && y > 86 && y < 194) {
        r *= 0.55;
        g *= 0.55;
        b *= 0.55;
      }

      sceneData[idx] = clamp255(r);
      sceneData[idx + 1] = clamp255(g);
      sceneData[idx + 2] = clamp255(b);
      sceneData[idx + 3] = 255;
    }
  }

  // Blit downscaled plate (336x86) into bumper recess
  const dstW = 336;
  const dstH = 86;
  const ox = Math.floor((sw - dstW) / 2);
  const oy = Math.floor((sh - dstH) / 2);

  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(ph - 1, Math.floor((y / dstH) * ph));
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(pw - 1, Math.floor((x / dstW) * pw));
      const srcIdx = (sy * pw + sx) * 4;
      const dstIdx = ((oy + y) * sw + (ox + x)) * 4;
      sceneData[dstIdx] = plateData[srcIdx];
      sceneData[dstIdx + 1] = plateData[srcIdx + 1];
      sceneData[dstIdx + 2] = plateData[srcIdx + 2];
      sceneData[dstIdx + 3] = 255;
    }
  }

  return { width: sw, height: sh, data: sceneData };
}

/** Fast 3x3 box blur for atmospheric scattering */
function boxBlurRgba(src: RgbaBuffer, radius: number): RgbaBuffer {
  if (radius <= 0) return src;
  const { width: w, height: h, data } = src;
  const tmp = new Float32Array(w * h * 3);
  const out = new Uint8ClampedArray(w * h * 4);
  const win = 2 * radius + 1;

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sr = 0;
      let sg = 0;
      let sb = 0;
      for (let k = -radius; k <= radius; k++) {
        const xx = Math.max(0, Math.min(w - 1, x + k));
        const idx = (y * w + xx) * 4;
        sr += data[idx];
        sg += data[idx + 1];
        sb += data[idx + 2];
      }
      const o = (y * w + x) * 3;
      tmp[o] = sr / win;
      tmp[o + 1] = sg / win;
      tmp[o + 2] = sb / win;
    }
  }

  // Vertical pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sr = 0;
      let sg = 0;
      let sb = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = Math.max(0, Math.min(h - 1, y + k));
        const idx = (yy * w + x) * 3;
        sr += tmp[idx];
        sg += tmp[idx + 1];
        sb += tmp[idx + 2];
      }
      const o = (y * w + x) * 4;
      out[o] = clamp255(sr / win);
      out[o + 1] = clamp255(sg / win);
      out[o + 2] = clamp255(sb / win);
      out[o + 3] = 255;
    }
  }
  return { width: w, height: h, data: out };
}

/** Apply Koschmieder atmospheric fog degradation with realistic dense scattering & sensor noise. */
export function applyFogDegradation(
  src: RgbaBuffer,
  severity: number,
  rng: () => number
): RgbaBuffer {
  const { width: w, height: h, data } = src;
  const out = new Uint8ClampedArray(w * h * 4);
  const phaseX = rng() * Math.PI * 2;
  const phaseY = rng() * Math.PI * 2;
  const airR = 224;
  const airG = 229;
  const airB = 235;

  for (let y = 0; y < h; y++) {
    const ny = y / h;
    for (let x = 0; x < w; x++) {
      const nx = x / w;
      // Spatially varying dense fog veil (transmission drops to 0.08–0.16 at severity ~0.8)
      const bank = 0.90 + 0.10 * Math.sin(nx * 4.2 + phaseX) * Math.cos(ny * 3.1 + phaseY);
      const t = Math.max(0.085, 1.0 - Math.min(0.95, severity * 1.12) * bank);
      const noise = (rng() - 0.5) * 4.5 * severity;
      const idx = (y * w + x) * 4;
      out[idx] = clamp255(data[idx] * t + airR * (1 - t) + noise);
      out[idx + 1] = clamp255(data[idx + 1] * t + airG * (1 - t) + noise);
      out[idx + 2] = clamp255(data[idx + 2] * t + airB * (1 - t) + noise);
      out[idx + 3] = 255;
    }
  }

  return boxBlurRgba({ width: w, height: h, data: out }, severity > 0.62 ? 1 : 0);
}

/** Apply dense slanted rain streaks + wet atmospheric haze */
export function applyRainDegradation(
  src: RgbaBuffer,
  severity: number,
  rng: () => number
): RgbaBuffer {
  // Wet haze base (stronger mist so rain noticeably degrades raw thresholding)
  const misty = applyFogDegradation(src, severity * 0.78, rng);
  const { width: w, height: h, data } = misty;
  const out = new Uint8ClampedArray(data);

  const streakCount = Math.floor(((w * h) / 240) * (0.6 + severity * 1.5));
  const slant = (rng() - 0.5) * 0.40;

  for (let i = 0; i < streakCount; i++) {
    const x0 = rng() * w;
    const y0 = rng() * h;
    const len = 12 + rng() * (20 * severity);
    const alpha = (0.34 + rng() * 0.44) * severity;
    const steps = Math.ceil(len);

    for (let s = 0; s < steps; s++) {
      const px = Math.round(x0 + s * slant);
      const py = Math.round(y0 + s);
      if (px < 0 || px >= w || py < 0 || py >= h) continue;
      for (let dx = 0; dx <= 1; dx++) {
        const xx = px + dx;
        if (xx >= w) continue;
        const a = dx === 0 ? alpha : alpha * 0.42;
        const idx = (py * w + xx) * 4;
        out[idx] = clamp255(out[idx] * (1 - a) + 236 * a);
        out[idx + 1] = clamp255(out[idx + 1] * (1 - a) + 240 * a);
        out[idx + 2] = clamp255(out[idx + 2] * (1 - a) + 246 * a);
      }
    }
  }

  return { width: w, height: h, data: out };
}

/**
 * Apply optical nighttime headlight veiling glare using soft highlight roll-off
 * (preserves faint ~10–18 gray-level residual signal beneath the halogen bloom,
 * just like real camera tone-mapping, so raw thresholding fails while adaptive
 * gamma + LAB CLAHE + DCP can recover the hidden characters).
 */
export function applyGlareDegradation(
  src: RgbaBuffer,
  severity: number,
  rng: () => number
): RgbaBuffer {
  const { width: w, height: h, data } = src;
  const out = new Uint8ClampedArray(w * h * 4);

  const cx = (0.30 + rng() * 0.40) * w;
  const cy = (0.32 + rng() * 0.36) * h;
  const sigX = w * (0.24 + rng() * 0.08);
  const sigY = h * (0.42 + rng() * 0.14);

  const cx2 = cx > w * 0.5 ? w * 0.24 : w * 0.76;
  const cy2 = h * 0.48;
  const sigX2 = w * 0.20;
  const sigY2 = h * 0.38;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const g1 = Math.exp(
        -(((x - cx) * (x - cx)) / (2 * sigX * sigX) + ((y - cy) * (y - cy)) / (2 * sigY * sigY))
      );
      const g2 =
        0.72 *
        Math.exp(
          -(((x - cx2) * (x - cx2)) / (2 * sigX2 * sigX2) +
            ((y - cy2) * (y - cy2)) / (2 * sigY2 * sigY2))
        );

      // Soft optical veil strength bounded below 0.90 so sensor never clips to 0-bit pure white
      const veil = Math.min(0.89, 0.38 * severity + (g1 + g2) * 0.56 * severity);

      const idx = (y * w + x) * 4;
      const r = data[idx] / 255;
      const g = data[idx + 1] / 255;
      const b = data[idx + 2] / 255;

      // Warm halogen/xenon glare compression (preserves micro-contrast inside glare shoulder)
      const nr = r * (1 - veil) + 0.99 * veil;
      const ng = g * (1 - veil) + 0.97 * veil;
      const nb = b * (1 - veil) + 0.91 * veil;

      out[idx] = clamp255(nr * 255);
      out[idx + 1] = clamp255(ng * 255);
      out[idx + 2] = clamp255(nb * 255);
      out[idx + 3] = 255;
    }
  }

  return { width: w, height: h, data: out };
}

export function applyWeather(src: RgbaBuffer, config: WeatherConfig): RgbaBuffer {
  if (config.mode === 'none' || config.severity <= 0.01) {
    return { width: src.width, height: src.height, data: new Uint8ClampedArray(src.data) };
  }
  const rng = createSeededRng(config.seed);
  if (config.mode === 'fog') {
    return applyFogDegradation(src, config.severity, rng);
  }
  if (config.mode === 'rain') {
    return applyRainDegradation(src, config.severity, rng);
  }
  if (config.mode === 'glare') {
    return applyGlareDegradation(src, config.severity, rng);
  }
  let img = applyFogDegradation(src, config.severity * 0.62, rng);
  img = applyRainDegradation(img, config.severity * 0.65, rng);
  img = applyGlareDegradation(img, config.severity * 0.48, rng);
  return img;
}

export function applyCompoundWeather(
  src: RgbaBuffer,
  config: { fogOpacity: number; rainDensity: number; glareIntensity: number; seed: number }
): RgbaBuffer {
  let out: RgbaBuffer = { width: src.width, height: src.height, data: new Uint8ClampedArray(src.data) };
  const rng = createSeededRng(config.seed);
  if (config.fogOpacity > 0.01) {
    out = applyFogDegradation(out, config.fogOpacity, rng);
  }
  if (config.rainDensity > 0.01) {
    out = applyRainDegradation(out, config.rainDensity, rng);
  }
  if (config.glareIntensity > 0.01) {
    out = applyGlareDegradation(out, config.glareIntensity, rng);
  }
  return out;
}
