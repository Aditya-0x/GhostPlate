import React, { useMemo, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { ChevronDown, RotateCcw, SlidersHorizontal, Upload } from 'lucide-react';
import { DEFAULT_DEHAZE_PARAMS } from '../vision/dehaze';
import { runFullAlprPipeline } from '../vision/recognizer';
import { DehazeParams, RgbaBuffer, SamplePreset } from '../vision/types';
import { applyCompoundWeather, generateCleanPlateBuffer, SAMPLE_PRESETS } from '../vision/weather';
import { BeforeAfterSlider } from './BeforeAfterSlider';
import { CanvasView } from './CanvasView';

function presetToWeatherSliders(preset: SamplePreset): {
  fogOpacity: number;
  rainDensity: number;
  glareIntensity: number;
} {
  if (preset.weather === 'fog') {
    return { fogOpacity: preset.severity, rainDensity: 0, glareIntensity: 0 };
  }
  if (preset.weather === 'rain') {
    return { fogOpacity: 0, rainDensity: preset.severity, glareIntensity: 0 };
  }
  if (preset.weather === 'glare') {
    return { fogOpacity: 0, rainDensity: 0, glareIntensity: preset.severity };
  }
  if (preset.weather === 'mixed') {
    return {
      fogOpacity: Number((preset.severity * 0.65).toFixed(2)),
      rainDensity: Number((preset.severity * 0.65).toFixed(2)),
      glareIntensity: Number((preset.severity * 0.5).toFixed(2)),
    };
  }
  return { fogOpacity: 0, rainDensity: 0, glareIntensity: 0 };
}

export const VisionLabSection: React.FC = () => {
  const initialSliders = presetToWeatherSliders(SAMPLE_PRESETS[0]);

  const [selectedPresetId, setSelectedPresetId] = useState<string>(SAMPLE_PRESETS[0].id);
  const [uploadedImage, setUploadedImage] = useState<{ name: string; buffer: RgbaBuffer } | null>(null);
  const [customPlateText, setCustomPlateText] = useState<string>(SAMPLE_PRESETS[0].plateText);
  const [fogOpacity, setFogOpacity] = useState<number>(initialSliders.fogOpacity);
  const [rainDensity, setRainDensity] = useState<number>(initialSliders.rainDensity);
  const [glareIntensity, setGlareIntensity] = useState<number>(initialSliders.glareIntensity);
  const [dehazeParams, setDehazeParams] = useState<DehazeParams>({
    ...DEFAULT_DEHAZE_PARAMS,
    airlightCap: 0.96,
  });
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const [mapMode, setMapMode] = useState<'transmission' | 'dark'>('transmission');
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activePreset = useMemo(
    () => SAMPLE_PRESETS.find((p) => p.id === selectedPresetId) ?? SAMPLE_PRESETS[0],
    [selectedPresetId]
  );

  const handleSelectPreset = (preset: SamplePreset) => {
    const sliders = presetToWeatherSliders(preset);
    setUploadedImage(null);
    setUploadError(null);
    setSelectedPresetId(preset.id);
    setCustomPlateText(preset.plateText);
    setFogOpacity(sliders.fogOpacity);
    setRainDensity(sliders.rainDensity);
    setGlareIntensity(sliders.glareIntensity);
  };

  const loadImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setUploadError('Please provide a PNG, JPG, or WebP vehicle or plate image.');
      return;
    }
    setUploadError(null);
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const maxW = 640;
        const scale = img.width > maxW ? maxW / img.width : 1;
        const w = Math.max(64, Math.round(img.width * scale));
        const h = Math.max(32, Math.round(img.height * scale));
        const off = document.createElement('canvas');
        off.width = w;
        off.height = h;
        const ctx = off.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(img, 0, 0, w, h);
        const raw = ctx.getImageData(0, 0, w, h);
        setUploadedImage({
          name: file.name,
          buffer: { width: w, height: h, data: new Uint8ClampedArray(raw.data) },
        });
        setFogOpacity(0);
        setRainDensity(0);
        setGlareIntensity(0);
      };
      img.onerror = () => setUploadError('Unable to decode the selected image file.');
      img.src = String(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadImageFile(file);
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) loadImageFile(file);
  };

  const handleReset = () => {
    setDehazeParams({ ...DEFAULT_DEHAZE_PARAMS, airlightCap: 0.96 });
    if (!uploadedImage) {
      const sliders = presetToWeatherSliders(activePreset);
      setCustomPlateText(activePreset.plateText);
      setFogOpacity(sliders.fogOpacity);
      setRainDensity(sliders.rainDensity);
      setGlareIntensity(sliders.glareIntensity);
    } else {
      setFogOpacity(0);
      setRainDensity(0);
      setGlareIntensity(0);
    }
  };

  const inputBuffer = useMemo<RgbaBuffer>(() => {
    const base = uploadedImage
      ? uploadedImage.buffer
      : generateCleanPlateBuffer(
          customPlateText || 'DL01AB4829',
          activePreset.embedInScene,
          activePreset.seed
        );
    return applyCompoundWeather(base, {
      fogOpacity,
      rainDensity,
      glareIntensity,
      seed: activePreset.seed + 50,
    });
  }, [uploadedImage, customPlateText, activePreset, fogOpacity, rainDensity, glareIntensity]);

  const result = useMemo(
    () => runFullAlprPipeline(inputBuffer, dehazeParams),
    [inputBuffer, dehazeParams]
  );

  const groundTruth = uploadedImage ? null : customPlateText.toUpperCase().replace(/[^0-9A-Z]/g, '');
  const contrastDeltaPct =
    result.dehaze.contrastBefore > 0
      ? Math.round(
          ((result.dehaze.contrastAfter - result.dehaze.contrastBefore) /
            result.dehaze.contrastBefore) *
            100
        )
      : 0;

  return (
    <section
      id="lab"
      data-section-theme="light"
      className="min-h-screen w-full bg-[#F4F4F0] text-[#0E1012] border-b border-[#0E1012]/12"
    >
      <div className="w-full max-w-[1440px] mx-auto px-6 sm:px-12 xl:pr-24 py-24 sm:py-28 space-y-12">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
          className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-end pb-8 border-b border-[#0E1012]/12"
        >
          <div className="lg:col-span-7 space-y-3">
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#6E7378]">
              02 / 05 · Interactive Vision Lab
            </div>
            <h2 className="font-display text-3xl sm:text-5xl font-light tracking-[-0.03em] leading-[1.06] text-[#0E1012] [text-wrap:balance]">
              Real-time optical restoration & character inference.
            </h2>
          </div>
          <div className="lg:col-span-5 text-sm text-[#6E7378] leading-relaxed">
            Every frame is processed pixel-by-pixel inside HTML5 Canvas using Dark-Channel Prior
            transmission refinement, LAB-space CLAHE, adaptive gamma correction, and 8-connected
            component segmentation.
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-10 items-start">
          <aside className="lg:col-span-4 space-y-7 border-b lg:border-b-0 lg:border-r border-[#0E1012]/12 pb-8 lg:pb-0 lg:pr-8">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#6E7378]">
                  Pre-Filter State
                </span>
                <button
                  type="button"
                  onClick={handleReset}
                  className="inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.12em] text-[#6E7378] hover:text-[#0E1012] transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Reset</span>
                </button>
              </div>

              <button
                type="button"
                role="switch"
                aria-checked={dehazeParams.enabled}
                onClick={() => setDehazeParams((p) => ({ ...p, enabled: !p.enabled }))}
                className={`w-full flex items-center justify-between px-4 py-3.5 border text-left transition-colors cursor-pointer ${
                  dehazeParams.enabled
                    ? 'bg-[#0E1012] border-[#0E1012] text-[#F4F4F0]'
                    : 'bg-transparent border-[#0E1012]/30 text-[#0E1012]'
                }`}
              >
                <div>
                  <div className="text-sm font-medium">Dark-Channel De-Hazing Pre-Filter</div>
                  <div
                    className={`text-xs mt-0.5 ${
                      dehazeParams.enabled ? 'text-[#A6ACB2]' : 'text-[#6E7378]'
                    }`}
                  >
                    {dehazeParams.enabled
                      ? 'Active · Guided Filter + Radiance Recovery'
                      : 'Bypassed · Raw Baseline Evaluation'}
                  </div>
                </div>
                <span
                  className={`font-mono text-xs font-semibold tracking-[0.12em] px-2.5 py-1 ${
                    dehazeParams.enabled
                      ? 'bg-[#1E6B7B] text-[#F4F4F0]'
                      : 'border border-[#0E1012]/30 text-[#0E1012]'
                  }`}
                >
                  {dehazeParams.enabled ? 'ON' : 'OFF'}
                </span>
              </button>

              <label className="flex items-center justify-between gap-3 pt-1 cursor-pointer select-none">
                <span className="text-xs text-[#2B3036]">
                  Glare & Rain Suppression (Adaptive Gamma + CLAHE)
                </span>
                <input
                  type="checkbox"
                  checked={dehazeParams.glareSuppression}
                  disabled={!dehazeParams.enabled}
                  onChange={(e) =>
                    setDehazeParams((p) => ({ ...p, glareSuppression: e.target.checked }))
                  }
                  className="w-4 h-4 accent-[#1E6B7B] disabled:opacity-40 cursor-pointer"
                />
              </label>

              <div className="border-t border-[#0E1012]/10 pt-3">
                <button
                  type="button"
                  aria-expanded={drawerOpen}
                  onClick={() => setDrawerOpen((v) => !v)}
                  className="w-full flex items-center justify-between py-1.5 text-xs font-mono uppercase tracking-[0.14em] text-[#2B3036] hover:text-[#1E6B7B] transition-colors cursor-pointer"
                >
                  <span className="inline-flex items-center gap-2">
                    <SlidersHorizontal className="w-3.5 h-3.5" />
                    <span>DCP Parameters</span>
                  </span>
                  <ChevronDown
                    className={`w-4 h-4 transition-transform duration-200 ${
                      drawerOpen ? 'rotate-180' : ''
                    }`}
                  />
                </button>

                {drawerOpen && (
                  <div className="pt-4 space-y-4">
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <label htmlFor="lab-patch-slider" className="text-[#2B3036]">
                          Patch Size
                        </label>
                        <span className="font-mono tabular-nums text-[#1E6B7B]">
                          {dehazeParams.patchSize} px
                        </span>
                      </div>
                      <input
                        id="lab-patch-slider"
                        type="range"
                        min={5}
                        max={21}
                        step={2}
                        disabled={!dehazeParams.enabled}
                        value={dehazeParams.patchSize}
                        onChange={(e) =>
                          setDehazeParams((p) => ({ ...p, patchSize: parseInt(e.target.value, 10) }))
                        }
                        className="w-full disabled:opacity-40"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <label htmlFor="lab-omega-slider" className="text-[#2B3036]">
                          Haze Removal Strength (ω)
                        </label>
                        <span className="font-mono tabular-nums text-[#1E6B7B]">
                          {dehazeParams.omega.toFixed(2)}
                        </span>
                      </div>
                      <input
                        id="lab-omega-slider"
                        type="range"
                        min={0.5}
                        max={0.98}
                        step={0.02}
                        disabled={!dehazeParams.enabled}
                        value={dehazeParams.omega}
                        onChange={(e) =>
                          setDehazeParams((p) => ({ ...p, omega: parseFloat(e.target.value) }))
                        }
                        className="w-full disabled:opacity-40"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <label htmlFor="lab-airlight-slider" className="text-[#2B3036]">
                          Atmospheric Light Threshold
                        </label>
                        <span className="font-mono tabular-nums text-[#1E6B7B]">
                          {(dehazeParams.airlightCap ?? 0.96).toFixed(2)}
                        </span>
                      </div>
                      <input
                        id="lab-airlight-slider"
                        type="range"
                        min={0.65}
                        max={0.98}
                        step={0.01}
                        disabled={!dehazeParams.enabled}
                        value={dehazeParams.airlightCap ?? 0.96}
                        onChange={(e) =>
                          setDehazeParams((p) => ({
                            ...p,
                            airlightCap: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full disabled:opacity-40"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-3 pt-6 border-t border-[#0E1012]/12">
              <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#6E7378]">
                Input Source & Presets
              </div>

              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragOver(true);
                }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                className={`border border-dashed p-4 transition-colors ${
                  isDragOver
                    ? 'border-[#1E6B7B] bg-[#1E6B7B]/8'
                    : 'border-[#0E1012]/25 hover:border-[#0E1012]/50'
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="text-xs text-[#2B3036] min-w-0">
                    {uploadedImage ? (
                      <span className="font-mono truncate block text-[#1E6B7B]">
                        {uploadedImage.name}
                      </span>
                    ) : (
                      <span>Drop a vehicle or plate photo</span>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.12em] px-3 py-1.5 border border-[#0E1012]/25 hover:bg-[#0E1012] hover:text-[#F4F4F0] transition-colors cursor-pointer whitespace-nowrap shrink-0"
                  >
                    <Upload className="w-3 h-3" />
                    <span>Browse</span>
                  </button>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileInput}
                  className="hidden"
                  aria-label="Upload vehicle or license plate image"
                />
              </div>

              {uploadError && (
                <p className="text-xs text-rose-700 border border-rose-700/30 bg-rose-50 px-3 py-2">
                  {uploadError}
                </p>
              )}

              <div className="space-y-1.5" role="group" aria-label="Sample adverse weather presets">
                {SAMPLE_PRESETS.map((preset) => {
                  const isActive = !uploadedImage && selectedPresetId === preset.id;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => handleSelectPreset(preset)}
                      className={`w-full flex items-center justify-between px-3.5 py-2.5 border text-left text-xs transition-colors cursor-pointer ${
                        isActive
                          ? 'bg-[#0E1012] border-[#0E1012] text-[#F4F4F0]'
                          : 'bg-transparent border-[#0E1012]/15 text-[#2B3036] hover:border-[#0E1012]/45'
                      }`}
                    >
                      <span className="truncate font-medium">{preset.label}</span>
                      <span
                        className={`font-mono text-[10px] uppercase tracking-[0.12em] shrink-0 ml-2 ${
                          isActive ? 'text-[#4CA8B8]' : 'text-[#6E7378]'
                        }`}
                      >
                        {preset.weather}
                      </span>
                    </button>
                  );
                })}
              </div>

              {!uploadedImage && (
                <div className="pt-2 space-y-1">
                  <label
                    htmlFor="synthetic-plate-input"
                    className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]"
                  >
                    Synthetic Plate Text
                  </label>
                  <input
                    id="synthetic-plate-input"
                    type="text"
                    maxLength={10}
                    value={customPlateText}
                    onChange={(e) =>
                      setCustomPlateText(e.target.value.toUpperCase().replace(/[^0-9A-Z]/g, ''))
                    }
                    className="w-full px-3 py-2 bg-white border border-[#0E1012]/20 font-mono text-sm tracking-[0.18em] text-[#0E1012] focus:outline-none focus:border-[#1E6B7B]"
                  />
                </div>
              )}
            </div>

            <div className="space-y-4 pt-6 border-t border-[#0E1012]/12">
              <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#6E7378]">
                Live Weather Degradation Sliders
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <label htmlFor="slider-fog" className="text-[#2B3036]">
                    Fog Opacity
                  </label>
                  <span className="font-mono tabular-nums text-[#0E1012]">
                    {Math.round(fogOpacity * 100)}%
                  </span>
                </div>
                <input
                  id="slider-fog"
                  type="range"
                  min={0}
                  max={0.9}
                  step={0.02}
                  value={fogOpacity}
                  onChange={(e) => setFogOpacity(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <label htmlFor="slider-rain" className="text-[#2B3036]">
                    Rain Density
                  </label>
                  <span className="font-mono tabular-nums text-[#0E1012]">
                    {Math.round(rainDensity * 100)}%
                  </span>
                </div>
                <input
                  id="slider-rain"
                  type="range"
                  min={0}
                  max={0.9}
                  step={0.02}
                  value={rainDensity}
                  onChange={(e) => setRainDensity(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <label htmlFor="slider-glare" className="text-[#2B3036]">
                    Glare Intensity
                  </label>
                  <span className="font-mono tabular-nums text-[#0E1012]">
                    {Math.round(glareIntensity * 100)}%
                  </span>
                </div>
                <input
                  id="slider-glare"
                  type="range"
                  min={0}
                  max={0.9}
                  step={0.02}
                  value={glareIntensity}
                  onChange={(e) => setGlareIntensity(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </aside>

          <div className="lg:col-span-8 space-y-8 min-w-0">
            <BeforeAfterSlider
              beforeBuffer={result.input}
              afterBuffer={result.dehaze.restored}
              beforeLabel="01 · Degraded Input"
              afterLabel="02 · Dark-Channel Prior + Glare Suppressed"
              dcpEnabled={dehazeParams.enabled}
            />

            <div className="border border-[#0E1012]/15 bg-white p-6 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-5 border-b border-[#0E1012]/10">
                <div className="space-y-1.5">
                  <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                    <span>Predicted License Plate</span>
                    {groundTruth && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span>Ground Truth {groundTruth}</span>
                      </>
                    )}
                  </div>
                  <div
                    data-testid="predicted-plate-readout"
                    className="font-mono text-3xl sm:text-5xl font-semibold tracking-[0.18em] text-[#0E1012] break-all"
                  >
                    {result.predictedPlate || 'UNREADABLE'}
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-6 shrink-0">
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-[#6E7378]">
                      Confidence
                    </div>
                    <div
                      data-testid="mean-confidence"
                      className="mt-1 font-mono text-xl font-semibold tabular-nums text-[#1E6B7B]"
                    >
                      {(result.meanConfidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-[#6E7378]">
                      Latency
                    </div>
                    <div
                      data-testid="latency-ms"
                      className="mt-1 font-mono text-xl font-semibold tabular-nums text-[#0E1012]"
                    >
                      {result.totalLatencyMs.toFixed(1)}
                      <span className="text-xs font-normal text-[#6E7378] ml-0.5">ms</span>
                    </div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-[#6E7378]">
                      RMS Lift
                    </div>
                    <div className="mt-1 font-mono text-xl font-semibold tabular-nums text-[#0E1012]">
                      {contrastDeltaPct >= 0 ? `+${contrastDeltaPct}%` : `${contrastDeltaPct}%`}
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-2.5">
                <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                  <span>Per-Character Confidence (0–9, A–Z)</span>
                  <span className="tabular-nums">{result.predictions.length} characters segmented</span>
                </div>

                {result.predictions.length > 0 ? (
                  <div
                    className="grid grid-cols-2 sm:grid-cols-5 gap-2.5"
                    data-testid="character-confidence-grid"
                  >
                    {result.predictions.map((pred, idx) => {
                      const pct = Math.round(pred.confidence * 100);
                      const isMismatch =
                        groundTruth !== null &&
                        (idx >= groundTruth.length || pred.char !== groundTruth[idx]);
                      return (
                        <div
                          key={idx}
                          className="border border-[#0E1012]/12 px-3 py-2 space-y-1.5 bg-[#F9F9F6]"
                        >
                          <div className="flex items-baseline justify-between font-mono">
                            <span
                              className={`text-base font-semibold ${
                                isMismatch ? 'text-amber-700' : 'text-[#0E1012]'
                              }`}
                            >
                              {pred.char}
                            </span>
                            <span className="text-[11px] tabular-nums text-[#6E7378]">{pct}%</span>
                          </div>
                          <div className="h-1 w-full bg-[#0E1012]/10 overflow-hidden">
                            <div
                              className={`h-full transition-all duration-200 ${
                                isMismatch ? 'bg-amber-600' : 'bg-[#1E6B7B]'
                              }`}
                              style={{ width: `${Math.max(4, pct)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="p-4 border border-dashed border-[#0E1012]/25 text-xs font-mono text-[#6E7378]">
                    No character components survived current weather severity. Enable the Dark-Channel
                    Pre-Filter to restore stroke contrast.
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#6E7378]">
                  Four-Stage Visual Pipeline Strip
                </span>
                <span className="font-mono text-[11px] tabular-nums text-[#6E7378]">
                  {dehazeParams.enabled
                    ? `A=[${result.dehaze.atmosphericLight.join(', ')}] · γ=${result.dehaze.gammaApplied} · σ=${result.dehaze.contrastBefore.toFixed(1)}→${result.dehaze.contrastAfter.toFixed(1)}`
                    : `Filter bypassed · σ=${result.dehaze.contrastBefore.toFixed(1)}`}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="border border-[#0E1012]/15 bg-white p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.12em] text-[#6E7378]">
                    <span>01 Degraded Input</span>
                    <span className="tabular-nums">
                      {inputBuffer.width}×{inputBuffer.height}
                    </span>
                  </div>
                  <div className="bg-[#0E1012] border border-[#0E1012]/10 overflow-hidden">
                    <CanvasView
                      buffer={result.segmentation.annotatedSource}
                      alt="01 Degraded Input"
                    />
                  </div>
                </div>

                <div className="border border-[#0E1012]/15 bg-white p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between gap-2 font-mono text-[11px] uppercase tracking-[0.12em] text-[#6E7378]">
                    <span>02 Dark Ch. & Transmission</span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setMapMode('transmission')}
                        className={`cursor-pointer ${
                          mapMode === 'transmission'
                            ? 'text-[#1E6B7B] underline underline-offset-4'
                            : 'hover:text-[#0E1012]'
                        }`}
                      >
                        t(x)
                      </button>
                      <button
                        type="button"
                        onClick={() => setMapMode('dark')}
                        className={`cursor-pointer ${
                          mapMode === 'dark'
                            ? 'text-[#1E6B7B] underline underline-offset-4'
                            : 'hover:text-[#0E1012]'
                        }`}
                      >
                        Dark
                      </button>
                    </div>
                  </div>
                  <div className="bg-[#0E1012] border border-[#0E1012]/10 overflow-hidden">
                    <CanvasView
                      buffer={
                        mapMode === 'transmission'
                          ? result.dehaze.transmissionVis
                          : result.dehaze.darkChannelVis
                      }
                      alt="02 Dark Channel and Transmission Map"
                    />
                  </div>
                </div>

                <div className="border border-[#0E1012]/15 bg-white p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.12em] text-[#6E7378]">
                    <span>03 De-Hazed & Glare-Suppressed</span>
                    <span className="tabular-nums text-[#1E6B7B]">{result.dehaze.elapsedMs}ms</span>
                  </div>
                  <div className="bg-[#0E1012] border border-[#0E1012]/10 overflow-hidden">
                    <CanvasView
                      buffer={result.dehaze.restored}
                      alt="03 De-hazed and Glare-Suppressed Plate"
                    />
                  </div>
                </div>

                <div className="border border-[#0E1012]/15 bg-white p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.12em] text-[#6E7378]">
                    <span>04 Segmented Bounding Boxes</span>
                    <span className="tabular-nums">
                      {result.segmentation.characters.length} crops
                    </span>
                  </div>
                  <div className="bg-[#0E1012] border border-[#0E1012]/10 overflow-hidden">
                    <CanvasView
                      buffer={result.segmentation.annotatedPlate}
                      alt="04 Segmented Character Bounding Boxes"
                    />
                  </div>
                </div>
              </div>

              {result.segmentation.characters.length > 0 && (
                <div className="border border-[#0E1012]/15 bg-white p-4 space-y-3">
                  <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                    Sorted 32×32 Character Tensors (Left-to-Right CNN Batch Input)
                  </div>
                  <div className="flex flex-wrap gap-2.5" data-testid="character-strip">
                    {result.segmentation.characters.map((seg, i) => {
                      const pred = result.predictions[i];
                      return (
                        <div
                          key={i}
                          className="w-[62px] border border-[#0E1012]/12 bg-[#F9F9F6] p-1.5 flex flex-col items-center gap-1"
                        >
                          <div className="w-11 h-11 bg-[#0E1012] overflow-hidden">
                            <CanvasView
                              buffer={seg.crop32}
                              kind="gray"
                              pixelated
                              alt={`Character tensor ${i + 1}: ${pred?.char ?? '?'}`}
                            />
                          </div>
                          <div className="w-full flex items-center justify-between font-mono text-[11px] px-0.5">
                            <span className="font-semibold text-[#0E1012]">{pred?.char ?? '?'}</span>
                            <span className="tabular-nums text-[#1E6B7B]">
                              {pred ? `${Math.round(pred.confidence * 100)}%` : '—'}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
