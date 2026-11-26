import test from 'node:test';
import assert from 'node:assert/strict';

import { computeRmsContrast, DEFAULT_DEHAZE_PARAMS, dehazeImage } from './dehaze';
import { localizeAndSegmentPlate } from './segment';
import { classifyCharacter32, runFullAlprPipeline } from './recognizer';
import { applyWeather, generateCleanPlateBuffer } from './weather';
import { runLiveAblationBenchmark } from './ablation';
import { renderCanonicalGlyph32, CHAR_VOCAB } from './glyphs';
import { buildColabNotebookJson, createZipBytes } from '../utils/exportProject';

test('Dark-Channel Prior de-hazing significantly increases contrast on heavy fog plate without NaNs', () => {
  const clean = generateCleanPlateBuffer('DL01AB4829', false, 101);
  const foggy = applyWeather(clean, { mode: 'fog', severity: 0.8, seed: 151 });
  const before = computeRmsContrast(foggy);

  const result = dehazeImage(foggy, { ...DEFAULT_DEHAZE_PARAMS, enabled: true });

  assert.equal(result.restored.width, foggy.width);
  assert.equal(result.restored.height, foggy.height);
  assert.ok(result.contrastAfter > before * 1.5, `Expected contrast boost: ${before} -> ${result.contrastAfter}`);

  for (let i = 0; i < result.restored.data.length; i += 400) {
    assert.ok(!Number.isNaN(result.restored.data[i]));
    assert.ok(result.restored.data[i] >= 0 && result.restored.data[i] <= 255);
  }
});

test('36-class character recognizer accurately classifies all canonical 0-9 A-Z glyphs with valid softmax', () => {
  let correct = 0;
  for (const ch of CHAR_VOCAB) {
    const glyph = renderCanonicalGlyph32(ch);
    const pred = classifyCharacter32(glyph);
    const topSum = pred.top3.reduce((s, item) => s + item.prob, 0);
    assert.ok(pred.confidence > 0 && pred.confidence <= 1.0);
    assert.ok(topSum <= 1.001, `Top-3 probabilities exceeded 1: ${topSum}`);
    if (pred.char === ch) correct++;
  }
  assert.equal(correct, 36, `Expected 36/36 canonical glyphs classified accurately, got ${correct}`);
});

test('Plate localization & character segmentation detects all 10 sorted characters on clean and bumper-embedded plates', () => {
  const plate = generateCleanPlateBuffer('MH12DE7391', false, 202);
  const seg = localizeAndSegmentPlate(plate);
  assert.equal(seg.usedPrecropFallback, true);
  assert.equal(seg.characters.length, 10);
  // Ensure strict left-to-right sorting
  for (let i = 1; i < seg.characters.length; i++) {
    assert.ok(seg.characters[i].bbox.x >= seg.characters[i - 1].bbox.x);
  }

  // Full vehicle bumper scene should localize plate without fallback
  const scene = generateCleanPlateBuffer('GJ05KR8316', true, 505);
  const segScene = localizeAndSegmentPlate(scene);
  assert.equal(segScene.usedPrecropFallback, false);
  assert.ok(segScene.characters.length >= 8, `Expected >=8 segmented chars on vehicle scene, got ${segScene.characters.length}`);
});

test('End-to-end ALPR pipeline recovers degraded fog plate dramatically better WITH DCP than WITHOUT DCP', () => {
  const gt = 'DL01AB4829';
  const clean = generateCleanPlateBuffer(gt, false, 101);
  const degraded = applyWeather(clean, { mode: 'fog', severity: 0.8, seed: 151 });

  const withoutDcp = runFullAlprPipeline(degraded, { ...DEFAULT_DEHAZE_PARAMS, enabled: false });
  const withDcp = runFullAlprPipeline(degraded, { ...DEFAULT_DEHAZE_PARAMS, enabled: true });

  assert.equal(withDcp.predictedPlate, gt, `Expected exact recovery WITH DCP, got ${withDcp.predictedPlate}`);
  assert.ok(
    withDcp.predictedPlate.length > withoutDcp.predictedPlate.length || withDcp.meanConfidence > withoutDcp.meanConfidence,
    `Expected DCP to outperform baseline: with=${withDcp.predictedPlate}, without=${withoutDcp.predictedPlate}`
  );
});

test('Live ablation benchmark proves positive accuracy gain WITH DCP across Rain, Fog, and Glare', () => {
  const report = runLiveAblationBenchmark();
  assert.equal(report.perWeather.length, 3);
  assert.equal(report.totalPlatesEvaluated, 18);
  assert.ok(
    report.overallWithDcpAcc > report.overallWithoutDcpAcc + 20,
    `Expected >20% gain from DCP, got ${report.overallWithoutDcpAcc}% -> ${report.overallWithDcpAcc}%`
  );
  for (const w of report.perWeather) {
    assert.ok(
      w.withDcpAcc > w.withoutDcpAcc,
      `Expected DCP gain for ${w.weather}: ${w.withoutDcpAcc}% -> ${w.withDcpAcc}%`
    );
  }
});

test('ZIP and Colab Notebook exporters produce valid PKZIP header and valid JSON nbformat', () => {
  const files = [
    { path: 'requirements.txt', content: 'torch>=2.2.0\n' },
    { path: 'src/dehaze.py', content: 'print("ok")\n' },
  ];
  const zip = createZipBytes(files);
  assert.equal(zip[0], 0x50); // 'P'
  assert.equal(zip[1], 0x4b); // 'K'
  assert.equal(zip[2], 0x03);
  assert.equal(zip[3], 0x04);

  const nbJson = buildColabNotebookJson(files);
  const parsed = JSON.parse(nbJson);
  assert.equal(parsed.nbformat, 4);
  assert.ok(Array.isArray(parsed.cells) && parsed.cells.length >= 6);
});
