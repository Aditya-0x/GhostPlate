export interface RgbaBuffer {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface GrayBuffer {
  width: number;
  height: number;
  data: Uint8Array;
}

export type WeatherMode = 'none' | 'fog' | 'rain' | 'glare' | 'mixed';

export interface WeatherConfig {
  mode: WeatherMode;
  severity: number;
  seed: number;
}

export interface CompoundWeatherConfig {
  fogOpacity: number;
  rainDensity: number;
  glareIntensity: number;
  seed: number;
}

export interface DehazeParams {
  enabled: boolean;
  omega: number;
  patchSize: number;
  tMin: number;
  guidedRadius: number;
  guidedEps: number;
  glareSuppression: boolean;
  claheClipLimit: number;
  airlightCap?: number;
}

export interface DehazeResult {
  restored: RgbaBuffer;
  darkChannelVis: RgbaBuffer;
  transmissionVis: RgbaBuffer;
  atmosphericLight: [number, number, number];
  gammaApplied: number;
  contrastBefore: number;
  contrastAfter: number;
  elapsedMs: number;
}

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SegmentedCharacter {
  bbox: BBox;
  crop32: GrayBuffer;
}

export interface SegmentationResult {
  plateBbox: BBox;
  usedPrecropFallback: boolean;
  plateCrop: RgbaBuffer;
  plateBinary: RgbaBuffer;
  annotatedPlate: RgbaBuffer;
  annotatedSource: RgbaBuffer;
  characters: SegmentedCharacter[];
  elapsedMs: number;
}

export interface CharacterPrediction {
  char: string;
  confidence: number;
  top3: Array<{ char: string; prob: number }>;
}

export interface PipelineResult {
  input: RgbaBuffer;
  dehaze: DehazeResult;
  segmentation: SegmentationResult;
  predictions: CharacterPrediction[];
  predictedPlate: string;
  meanConfidence: number;
  totalLatencyMs: number;
  recognitionLatencyMs: number;
}

export interface SamplePreset {
  id: string;
  label: string;
  plateText: string;
  weather: WeatherMode;
  severity: number;
  seed: number;
  embedInScene: boolean;
  description: string;
}

export interface WeatherAblationStats {
  weather: 'rain' | 'fog' | 'glare';
  label: string;
  withoutDcpAcc: number;
  withDcpAcc: number;
  gainPct: number;
  withoutDcpLatencyMs: number;
  withDcpLatencyMs: number;
  totalChars: number;
}

export interface AblationVisualRow {
  weather: 'rain' | 'fog' | 'glare';
  groundTruth: string;
  predWithoutDcp: string;
  predWithDcp: string;
  confWithoutDcp: number;
  confWithDcp: number;
  original: RgbaBuffer;
  dehazed: RgbaBuffer;
  annotated: RgbaBuffer;
  charCrops: GrayBuffer[];
}

export interface ConfusionPair {
  truth: string;
  predicted: string;
  count: number;
}

export interface AblationReport {
  overallWithoutDcpAcc: number;
  overallWithDcpAcc: number;
  accuracyGainPct: number;
  overallWithoutDcpLatencyMs: number;
  overallWithDcpLatencyMs: number;
  perWeather: WeatherAblationStats[];
  confusionMatrix: number[][]; // 36x36 for WITH DCP
  confusionMatrixWithoutDcp: number[][]; // 36x36 for WITHOUT DCP
  topConfusionsWithoutDcp: ConfusionPair[];
  topConfusionsWithDcp: ConfusionPair[];
  visualSamples: AblationVisualRow[];
  totalPlatesEvaluated: number;
  totalCharsEvaluated: number;
}
