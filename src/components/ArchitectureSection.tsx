import React, { useState } from 'react';
import { motion } from 'motion/react';

interface PipelineStepSpec {
  id: string;
  index: string;
  title: string;
  modulePath: string;
  summary: string;
  equationLabel: string;
  equation: string;
  details: Array<{ label: string; value: string }>;
  highlights: string[];
}

const PIPELINE_STEPS: PipelineStepSpec[] = [
  {
    id: 'dcp',
    index: '01',
    title: 'Dark-Channel Prior De-Hazing',
    modulePath: 'src/dehaze.py',
    summary:
      'Inverts the physical Koschmieder atmospheric scattering model from scratch in OpenCV and NumPy without requiring paired haze training data.',
    equationLabel: 'Scene Radiance Recovery',
    equation: 'J(x) = ( I(x) - A ) / max( t(x), t_0 ) + A',
    details: [
      { label: 'Dark Channel Patch', value: '15 × 15 min-filter erosion' },
      { label: 'Airlight Estimation A', value: 'Top 0.1% dark-channel luminance' },
      { label: 'Transmission Refinement', value: 'O(N) Guided Filter (r=24, ε=1e-3)' },
      { label: 'Lower Bound t_0', value: '0.12 (prevents noise amplification)' },
    ],
    highlights: [
      'Recovers contrast on plates attenuated below 15% atmospheric visibility.',
      'Guided filter aligns coarse transmission boundaries sharply to alphanumeric edges.',
      'Luminance-averaged top-K airlight selection resists isolated headlight spikes.',
    ],
  },
  {
    id: 'glare',
    index: '02',
    title: 'Glare & Rain Mitigation',
    modulePath: 'src/dehaze.py',
    summary:
      'Suppresses nighttime halogen/xenon veiling glare and slanted rain streaks while preserving crisp high-frequency stroke edges.',
    equationLabel: 'Luminance-Adaptive Gamma + LAB CLAHE',
    equation: 'L_out = CLAHE( L_lab( I^γ(μ) ) )  →  Bilateral( d=5, σ=35 )',
    details: [
      { label: 'Adaptive Gamma γ', value: 'Dynamic 0.60 – 2.15 from mean & highlight ratio' },
      { label: 'Local Contrast', value: 'CIE-LAB L-channel CLAHE (clip=2.4, 8×8 grid)' },
      { label: 'Streak Denoising', value: 'Edge-preserving Bilateral Filter + impulse bridge' },
      { label: 'Color Integrity', value: 'Chromaticity channels a*, b* left undistorted' },
    ],
    highlights: [
      'Compresses blown-out glare shoulders to unlock hidden 10–18 gray-level micro-contrast.',
      'Eliminates narrow bright raindrop streaks slicing across dark strokes.',
      'Preserves sharp zero-crossing boundaries required for adaptive binarization.',
    ],
  },
  {
    id: 'segment',
    index: '03',
    title: 'Plate Localization & Character Segmentation',
    modulePath: 'src/segment.py',
    summary:
      'Isolates the license plate ROI from full vehicle frames via morphological stroke density, with an automatic pre-cropped fallback and left-to-right 8-CC character extraction.',
    equationLabel: 'Morphological Localization + 8-Connected Components',
    equation: 'ROI = argmax Contour( Close( |∂x Blackhat(I)| ) )  →  Sort_x( CC_8 )',
    details: [
      { label: 'ROI Aspect Filter', value: 'Aspect ratio 1.8 – 6.5 · area 1.5% – 65%' },
      { label: 'Pre-Cropped Fallback', value: 'Automatic bypass when input aspect ≥ 2.35' },
      { label: 'Binarization', value: 'Adaptive Gaussian Threshold (block=21, C=9)' },
      { label: 'Tensor Output', value: '32 × 32 centered aspect-preserved crops' },
    ],
    highlights: [
      'Handles uneven residual glare illumination better than global Otsu thresholding.',
      'Nested-component suppression removes internal glyph holes and plate mounting bolts.',
      'Outputs normalized 32×32 batches sorted cleanly from left to right.',
    ],
  },
  {
    id: 'cnn',
    index: '04',
    title: 'Lightweight PyTorch CNN',
    modulePath: 'src/model.py',
    summary:
      'Compact 4-block Depthwise-Separable convolutional network engineered for 36-class (0–9, A–Z) inference on free Google Colab T4 GPUs or standard laptop CPUs.',
    equationLabel: 'Depthwise-Separable Backbone + GAP',
    equation: 'Stem(32) → DS(64) → DS(128) → DS(192) → GAP → FC(128) → 36',
    details: [
      { label: 'Trainable Parameters', value: '68,004 (~275 KB .pth checkpoint)' },
      { label: 'Output Vocabulary', value: '36 classes (0–9, A–Z)' },
      { label: 'Training Schedule', value: 'AdamW + CosineAnnealingLR · Label Smoothing 0.05' },
      { label: 'Colab T4 Train Time', value: '< 60 seconds (12 epochs)' },
    ],
    highlights: [
      'Depthwise 3×3 + Pointwise 1×1 convolutions cut FLOPs by >6× vs standard Conv2d.',
      'Global Average Pooling prevents spatial overfitting and shrinks dense head size.',
      'Single batched forward pass classifies an entire 10-character plate in < 4 ms.',
    ],
  },
];

const CNN_LAYER_ROWS = [
  { stage: 'Stem Conv 3×3 + BN + ReLU + Pool', shape: '32 × 16 × 16', params: '352' },
  { stage: 'Depthwise-Sep Block 2 + Pool + Drop 0.10', shape: '64 × 8 × 8', params: '2,528' },
  { stage: 'Depthwise-Sep Block 3 + Pool + Drop 0.15', shape: '128 × 4 × 4', params: '9,152' },
  { stage: 'Depthwise-Sep Block 4 + Global Avg Pool', shape: '192 × 1 × 1', params: '26,368' },
  { stage: 'FC 192→128 + BN + Drop 0.30 + FC 36', shape: '36 logits', params: '29,604' },
];

export const ArchitectureSection: React.FC = () => {
  const [activeId, setActiveId] = useState<string>(PIPELINE_STEPS[0].id);
  const activeStep = PIPELINE_STEPS.find((s) => s.id === activeId) ?? PIPELINE_STEPS[0];

  return (
    <section
      id="pipeline"
      data-section-theme="dark"
      className="min-h-screen w-full bg-[#0E1012] text-[#F4F4F0] border-b border-[#F4F4F0]/10"
    >
      <div className="w-full max-w-[1440px] mx-auto px-6 sm:px-12 xl:pr-24 py-24 sm:py-28 space-y-14">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
          className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-end pb-8 border-b border-[#F4F4F0]/12"
        >
          <div className="lg:col-span-7 space-y-3">
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#8D9399]">
              03 / 05 · System Architecture & Methodology
            </div>
            <h2 className="font-display text-3xl sm:text-5xl font-light tracking-[-0.03em] leading-[1.06] [text-wrap:balance]">
              Four deterministic stages from storm noise to verified plate string.
            </h2>
          </div>
          <div className="lg:col-span-5 text-sm text-[#8D9399] leading-relaxed">
            Select any stage to inspect its mathematical formulation, OpenCV/PyTorch module
            boundary, and computational footprint.
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
          <div
            role="tablist"
            aria-label="System architecture stages"
            className="lg:col-span-5 divide-y divide-[#F4F4F0]/12 border-t border-b border-[#F4F4F0]/12"
          >
            {PIPELINE_STEPS.map((step) => {
              const isSelected = step.id === activeStep.id;
              return (
                <button
                  key={step.id}
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  onClick={() => setActiveId(step.id)}
                  className={`w-full py-6 px-4 text-left transition-colors cursor-pointer flex items-start gap-5 ${
                    isSelected
                      ? 'bg-[#151A1E] text-[#F4F4F0]'
                      : 'bg-transparent text-[#8D9399] hover:text-[#F4F4F0] hover:bg-[#F4F4F0]/[0.03]'
                  }`}
                >
                  <span
                    className={`font-mono text-xs tracking-[0.16em] pt-1 shrink-0 ${
                      isSelected ? 'text-[#4CA8B8]' : 'text-[#6E7378]'
                    }`}
                  >
                    {step.index}
                  </span>
                  <div className="space-y-1.5 min-w-0">
                    <div className="font-display text-lg sm:text-xl font-normal tracking-[-0.02em]">
                      {step.title}
                    </div>
                    <div className="font-mono text-[11px] text-[#6E7378]">{step.modulePath}</div>
                  </div>
                </button>
              );
            })}
          </div>

          <div
            role="tabpanel"
            className="lg:col-span-7 border border-[#F4F4F0]/12 bg-[#13171A] p-6 sm:p-10 space-y-8"
          >
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] uppercase tracking-[0.16em] text-[#4CA8B8]">
                <span>Stage {activeStep.index} · {activeStep.modulePath}</span>
                <span className="text-[#8D9399]">OpenCV / PyTorch Specification</span>
              </div>
              <h3 className="font-display text-2xl sm:text-3xl font-light tracking-[-0.025em] text-[#F4F4F0]">
                {activeStep.title}
              </h3>
              <p className="text-sm sm:text-base text-[#A6ACB2] leading-relaxed">
                {activeStep.summary}
              </p>
            </div>

            <div className="border border-[#F4F4F0]/12 bg-[#0E1012] p-5 space-y-2">
              <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#6E7378]">
                {activeStep.equationLabel}
              </div>
              <div className="font-mono text-sm sm:text-base text-[#F4F4F0] overflow-x-auto whitespace-nowrap">
                {activeStep.equation}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-px bg-[#F4F4F0]/10 border border-[#F4F4F0]/10">
              {activeStep.details.map((item) => (
                <div key={item.label} className="bg-[#13171A] p-4 space-y-1">
                  <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-[#6E7378]">
                    {item.label}
                  </div>
                  <div className="text-xs sm:text-sm text-[#F4F4F0]">{item.value}</div>
                </div>
              ))}
            </div>

            {activeStep.id === 'cnn' ? (
              <div className="overflow-x-auto border-t border-[#F4F4F0]/12 pt-6">
                <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#8D9399] mb-3">
                  Layer-by-Layer Parameter Budget (68,004 Total)
                </div>
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-[#F4F4F0]/12 font-mono text-[10px] uppercase tracking-[0.12em] text-[#6E7378]">
                      <th className="py-2 pr-4 font-normal">Block</th>
                      <th className="py-2 pr-4 font-normal">Output Tensor</th>
                      <th className="py-2 text-right font-normal">Params</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#F4F4F0]/8 font-mono tabular-nums">
                    {CNN_LAYER_ROWS.map((row) => (
                      <tr key={row.stage}>
                        <td className="py-2.5 pr-4 font-sans text-[#D5D9DD]">{row.stage}</td>
                        <td className="py-2.5 pr-4 text-[#8D9399] whitespace-nowrap">{row.shape}</td>
                        <td className="py-2.5 text-right text-[#4CA8B8]">{row.params}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <ul className="space-y-2.5 pt-2 border-t border-[#F4F4F0]/12">
                {activeStep.highlights.map((pt) => (
                  <li key={pt} className="flex items-start gap-3 text-xs sm:text-sm text-[#B8BEC4]">
                    <span className="font-mono text-[#4CA8B8] mt-0.5">—</span>
                    <span>{pt}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
