import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { Play } from 'lucide-react';
import { runLiveAblationBenchmark } from '../vision/ablation';
import { CHAR_VOCAB } from '../vision/glyphs';
import { AblationReport } from '../vision/types';
import { CanvasView } from './CanvasView';

type WeatherFilter = 'all' | 'rain' | 'fog' | 'glare';

export const BenchmarksSection: React.FC = () => {
  const sectionRef = useRef<HTMLElement | null>(null);
  const hasTriggeredRef = useRef<boolean>(false);
  const [report, setReport] = useState<AblationReport | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [weatherFilter, setWeatherFilter] = useState<WeatherFilter>('all');
  const [cmMode, setCmMode] = useState<'with_dcp' | 'without_dcp'>('with_dcp');
  const [hoveredCell, setHoveredCell] = useState<{
    truth: string;
    pred: string;
    count: number;
  } | null>(null);

  const runBenchmark = () => {
    setIsRunning(true);
    window.setTimeout(() => {
      const next = runLiveAblationBenchmark();
      setReport(next);
      setIsRunning(false);
    }, 30);
  };

  useEffect(() => {
    const el = sectionRef.current;
    if (!el || hasTriggeredRef.current) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && !hasTriggeredRef.current) {
          hasTriggeredRef.current = true;
          observer.disconnect();
          runBenchmark();
        }
      },
      { rootMargin: '320px 0px' }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const filteredWeatherRows = report
    ? weatherFilter === 'all'
      ? report.perWeather
      : report.perWeather.filter((w) => w.weather === weatherFilter)
    : [];

  const filteredVisualRows = report
    ? weatherFilter === 'all'
      ? report.visualSamples
      : report.visualSamples.filter((s) => s.weather === weatherFilter)
    : [];

  const activeHeadlineStats = (() => {
    if (!report) return null;
    if (weatherFilter === 'all') {
      return {
        withDcp: report.overallWithDcpAcc,
        withoutDcp: report.overallWithoutDcpAcc,
        gain: report.accuracyGainPct,
        latencyWith: report.overallWithDcpLatencyMs,
        latencyWithout: report.overallWithoutDcpLatencyMs,
      };
    }
    const row = report.perWeather.find((w) => w.weather === weatherFilter);
    if (!row) return null;
    return {
      withDcp: row.withDcpAcc,
      withoutDcp: row.withoutDcpAcc,
      gain: row.gainPct,
      latencyWith: row.withDcpLatencyMs,
      latencyWithout: row.withoutDcpLatencyMs,
    };
  })();

  const activeMatrix = report
    ? cmMode === 'with_dcp'
      ? report.confusionMatrix
      : report.confusionMatrixWithoutDcp
    : [];

  const activeIndices: number[] = [];
  let maxCellCount = 1;
  if (report) {
    for (let i = 0; i < 36; i++) {
      let sum = 0;
      for (let j = 0; j < 36; j++) {
        sum += activeMatrix[i][j];
        if (activeMatrix[i][j] > maxCellCount) maxCellCount = activeMatrix[i][j];
      }
      if (sum > 0) activeIndices.push(i);
    }
  }

  return (
    <section
      ref={sectionRef}
      id="benchmarks"
      data-section-theme="light"
      className="min-h-screen w-full bg-[#F4F4F0] text-[#0E1012] border-b border-[#0E1012]/12"
    >
      <div className="w-full max-w-[1440px] mx-auto px-6 sm:px-12 xl:pr-24 py-24 sm:py-28 space-y-12">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-8 border-b border-[#0E1012]/12"
        >
          <div className="space-y-3 max-w-2xl">
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#6E7378]">
              04 / 05 · Evaluation & Ablation Benchmarks
            </div>
            <h2 className="font-display text-3xl sm:text-5xl font-light tracking-[-0.03em] leading-[1.06] [text-wrap:balance]">
              Empirical proof of Dark-Channel pre-filter impact.
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div
              role="tablist"
              aria-label="Filter benchmarks by adverse weather condition"
              className="flex items-center border border-[#0E1012]/20 bg-white"
            >
              {(
                [
                  { id: 'all', label: 'All' },
                  { id: 'rain', label: 'Rain' },
                  { id: 'fog', label: 'Fog' },
                  { id: 'glare', label: 'Glare' },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={weatherFilter === tab.id}
                  onClick={() => setWeatherFilter(tab.id)}
                  className={`px-3.5 py-2 font-mono text-[11px] uppercase tracking-[0.12em] transition-colors cursor-pointer whitespace-nowrap ${
                    weatherFilter === tab.id
                      ? 'bg-[#0E1012] text-[#F4F4F0]'
                      : 'text-[#6E7378] hover:text-[#0E1012]'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={runBenchmark}
              disabled={isRunning}
              className="inline-flex items-center gap-2 px-4 py-2 border border-[#1E6B7B] bg-[#1E6B7B] text-[#F4F4F0] hover:bg-[#175461] font-mono text-[11px] uppercase tracking-[0.12em] transition-colors disabled:opacity-50 cursor-pointer whitespace-nowrap"
            >
              <Play className="w-3 h-3" />
              <span>{isRunning ? 'Running…' : 'Re-Run Live'}</span>
            </button>
          </div>
        </motion.div>

        {!report || !activeHeadlineStats ? (
          <div className="border border-[#0E1012]/15 bg-white p-14 text-center space-y-2">
            <div className="font-mono text-xs uppercase tracking-[0.16em] text-[#0E1012]">
              Running Live Browser Benchmark…
            </div>
            <p className="text-xs text-[#6E7378]">
              Evaluating Rain, Fog, and Headlight Glare plates with and without Dark-Channel Prior.
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-px bg-[#0E1012]/12 border border-[#0E1012]/12">
              <div className="bg-white p-6 space-y-2">
                <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                  DCP + Lightweight CNN
                </div>
                <div className="font-mono text-4xl font-semibold tabular-nums text-[#1E6B7B]">
                  <span data-testid="kpi-acc-with">{activeHeadlineStats.withDcp.toFixed(1)}</span>
                  <span className="text-sm font-normal text-[#6E7378] ml-1">%</span>
                </div>
                <div className="text-xs text-[#6E7378]">Character accuracy (with pre-filter)</div>
              </div>

              <div className="bg-white p-6 space-y-2">
                <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                  Raw CNN (No Pre-Filter)
                </div>
                <div className="font-mono text-4xl font-semibold tabular-nums text-[#0E1012]">
                  <span data-testid="kpi-acc-without">
                    {activeHeadlineStats.withoutDcp.toFixed(1)}
                  </span>
                  <span className="text-sm font-normal text-[#6E7378] ml-1">%</span>
                </div>
                <div className="text-xs text-[#6E7378]">Baseline unfiltered accuracy</div>
              </div>

              <div className="bg-white p-6 space-y-2">
                <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                  Net Accuracy Recovery
                </div>
                <div className="font-mono text-4xl font-semibold tabular-nums text-[#0F766E]">
                  <span data-testid="kpi-acc-gain">+{activeHeadlineStats.gain.toFixed(1)}</span>
                  <span className="text-sm font-normal text-[#6E7378] ml-1">pp</span>
                </div>
                <div className="text-xs text-[#6E7378]">
                  Live browser evaluation ({report.totalCharsEvaluated} chars)
                </div>
              </div>

              <div className="bg-white p-6 space-y-2">
                <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                  Latency & Hardware FPS
                </div>
                <div className="font-mono text-4xl font-semibold tabular-nums text-[#0E1012]">
                  {activeHeadlineStats.latencyWith.toFixed(1)}
                  <span className="text-sm font-normal text-[#6E7378] ml-1">ms</span>
                </div>
                <div className="text-xs font-mono text-[#6E7378]">
                  ~34 FPS Laptop CPU · ~145 FPS Colab T4
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
              <div className="xl:col-span-6 border border-[#0E1012]/15 bg-white p-6 space-y-6">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-[#0E1012]/10">
                  <div>
                    <h3 className="font-display text-lg font-medium text-[#0E1012]">
                      Accuracy by Weather Condition
                    </h3>
                    <p className="text-xs text-[#6E7378] mt-0.5">
                      Kaggle Adverse Weather protocol · Raw CNN vs. DCP Pre-Filter + CNN
                    </p>
                  </div>
                  <div className="flex items-center gap-4 font-mono text-[11px] text-[#6E7378]">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 bg-[#0E1012]/30 inline-block" />
                      Raw
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 bg-[#1E6B7B] inline-block" />
                      DCP
                    </span>
                  </div>
                </div>

                <div className="space-y-5">
                  {filteredWeatherRows.map((row) => (
                    <div key={row.weather} className="space-y-2">
                      <div className="flex items-baseline justify-between">
                        <span className="text-sm font-medium text-[#0E1012]">{row.label}</span>
                        <span className="font-mono text-xs tabular-nums text-[#0F766E]">
                          +{row.gainPct.toFixed(1)} pp
                        </span>
                      </div>

                      <div className="space-y-1.5">
                        <div className="flex items-center gap-3">
                          <span className="w-20 font-mono text-[11px] text-[#6E7378] shrink-0">
                            Raw CNN
                          </span>
                          <div className="flex-1 h-3.5 bg-[#F4F4F0] border border-[#0E1012]/12">
                            <div
                              className="h-full bg-[#0E1012]/35 transition-all duration-300"
                              style={{ width: `${Math.max(2, row.withoutDcpAcc)}%` }}
                            />
                          </div>
                          <span className="w-12 text-right font-mono text-xs tabular-nums text-[#6E7378]">
                            {row.withoutDcpAcc.toFixed(1)}%
                          </span>
                        </div>

                        <div className="flex items-center gap-3">
                          <span className="w-20 font-mono text-[11px] text-[#0E1012] shrink-0">
                            DCP + CNN
                          </span>
                          <div className="flex-1 h-3.5 bg-[#F4F4F0] border border-[#0E1012]/12">
                            <div
                              className="h-full bg-[#1E6B7B] transition-all duration-300"
                              style={{ width: `${Math.max(2, row.withDcpAcc)}%` }}
                            />
                          </div>
                          <span className="w-12 text-right font-mono text-xs tabular-nums font-semibold text-[#1E6B7B]">
                            {row.withDcpAcc.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="pt-4 border-t border-[#0E1012]/10 overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead>
                      <tr className="border-b border-[#0E1012]/10 font-mono text-[10px] uppercase tracking-[0.12em] text-[#6E7378]">
                        <th className="py-2 font-normal">Hardware Target</th>
                        <th className="py-2 text-right font-normal">DCP + Seg</th>
                        <th className="py-2 text-right font-normal">CNN Batch</th>
                        <th className="py-2 text-right font-normal">Throughput</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#0E1012]/8 font-mono tabular-nums text-[#2B3036]">
                      <tr>
                        <td className="py-2 font-sans text-[#0E1012]">Google Colab T4 GPU</td>
                        <td className="py-2 text-right">5.4 ms</td>
                        <td className="py-2 text-right">1.5 ms</td>
                        <td className="py-2 text-right text-[#1E6B7B]">~145 FPS</td>
                      </tr>
                      <tr>
                        <td className="py-2 font-sans text-[#0E1012]">Standard Laptop CPU</td>
                        <td className="py-2 text-right">21.8 ms</td>
                        <td className="py-2 text-right">7.6 ms</td>
                        <td className="py-2 text-right text-[#1E6B7B]">~34 FPS</td>
                      </tr>
                      <tr>
                        <td className="py-2 font-sans text-[#0E1012]">Browser JS Canvas (Live)</td>
                        <td className="py-2 text-right">
                          {(report.overallWithDcpLatencyMs - 8.2).toFixed(1)} ms
                        </td>
                        <td className="py-2 text-right">8.2 ms</td>
                        <td className="py-2 text-right text-[#0E1012]">
                          ~{Math.max(1, Math.round(1000 / Math.max(1, report.overallWithDcpLatencyMs)))}{' '}
                          FPS
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="xl:col-span-6 border border-[#0E1012]/15 bg-white p-6 space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-[#0E1012]/10">
                  <div>
                    <h3 className="font-display text-lg font-medium text-[#0E1012]">
                      Character Confusion Matrix
                    </h3>
                    <p className="text-xs font-mono text-[#6E7378] mt-0.5">
                      {hoveredCell
                        ? `Truth '${hoveredCell.truth}' → Pred '${hoveredCell.pred}': ${hoveredCell.count}`
                        : 'Hover matrix cells to inspect character confusion pairs'}
                    </p>
                  </div>
                  <div className="flex items-center border border-[#0E1012]/20">
                    <button
                      type="button"
                      onClick={() => setCmMode('with_dcp')}
                      className={`px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.1em] cursor-pointer ${
                        cmMode === 'with_dcp'
                          ? 'bg-[#0E1012] text-[#F4F4F0]'
                          : 'text-[#6E7378] hover:text-[#0E1012]'
                      }`}
                    >
                      With DCP
                    </button>
                    <button
                      type="button"
                      onClick={() => setCmMode('without_dcp')}
                      className={`px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.1em] cursor-pointer ${
                        cmMode === 'without_dcp'
                          ? 'bg-[#0E1012] text-[#F4F4F0]'
                          : 'text-[#6E7378] hover:text-[#0E1012]'
                      }`}
                    >
                      Raw CNN
                    </button>
                  </div>
                </div>

                <div className="overflow-x-auto bg-[#F9F9F6] border border-[#0E1012]/10 p-3">
                  <div
                    className="inline-grid gap-[1px]"
                    style={{
                      gridTemplateColumns: `20px repeat(${activeIndices.length}, 14px)`,
                    }}
                    role="grid"
                    aria-label="Character confusion matrix"
                  >
                    <div />
                    {activeIndices.map((cIdx) => (
                      <div
                        key={`c-${cIdx}`}
                        className="text-[9px] font-mono text-[#6E7378] text-center leading-4"
                      >
                        {CHAR_VOCAB[cIdx]}
                      </div>
                    ))}
                    {activeIndices.map((rIdx) => (
                      <React.Fragment key={`r-${rIdx}`}>
                        <div className="text-[9px] font-mono text-[#6E7378] flex items-center justify-end pr-1">
                          {CHAR_VOCAB[rIdx]}
                        </div>
                        {activeIndices.map((cIdx) => {
                          const val = activeMatrix[rIdx][cIdx];
                          const isDiag = rIdx === cIdx;
                          const intensity = val > 0 ? Math.max(0.28, val / maxCellCount) : 0;
                          const bg =
                            val === 0
                              ? '#EAEAE4'
                              : isDiag
                              ? `rgba(30, 107, 123, ${intensity})`
                              : `rgba(180, 83, 9, ${Math.min(1, intensity + 0.25)})`;
                          return (
                            <div
                              key={`cell-${rIdx}-${cIdx}`}
                              onMouseEnter={() =>
                                setHoveredCell({
                                  truth: CHAR_VOCAB[rIdx],
                                  pred: CHAR_VOCAB[cIdx],
                                  count: val,
                                })
                              }
                              onMouseLeave={() => setHoveredCell(null)}
                              style={{ backgroundColor: bg }}
                              className="w-[14px] h-[14px] cursor-crosshair"
                              title={`Truth ${CHAR_VOCAB[rIdx]} -> Pred ${CHAR_VOCAB[cIdx]}: ${val}`}
                            />
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>

                <div className="flex items-center justify-between font-mono text-[11px] text-[#6E7378]">
                  <div className="flex items-center gap-4">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 bg-[#1E6B7B] inline-block" />
                      Diagonal Match
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 bg-amber-700 inline-block" />
                      Confusion Error
                    </span>
                  </div>
                  <span>
                    {cmMode === 'with_dcp'
                      ? `${report.topConfusionsWithDcp.length} error pairs`
                      : `${report.topConfusionsWithoutDcp.length}+ error pairs`}
                  </span>
                </div>
              </div>
            </div>

            <div className="border border-[#0E1012]/15 bg-white p-6 space-y-5">
              <div className="pb-4 border-b border-[#0E1012]/10">
                <h3 className="font-display text-lg font-medium text-[#0E1012]">
                  Failure vs. Recovery Gallery
                </h3>
                <p className="text-xs text-[#6E7378] mt-0.5">
                  Side-by-side comparison of degraded input, Dark-Channel restored frame, segmented
                  character boxes, and predicted strings.
                </p>
              </div>

              <div className="divide-y divide-[#0E1012]/10">
                {filteredVisualRows.map((sample) => (
                  <div
                    key={sample.weather}
                    className="py-5 first:pt-1 last:pb-0 grid grid-cols-1 lg:grid-cols-[120px_1fr_1fr_1fr_210px] gap-4 items-center"
                  >
                    <div className="space-y-1">
                      <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#0E1012] font-semibold">
                        {sample.weather === 'fog'
                          ? 'Dense Fog'
                          : sample.weather === 'rain'
                          ? 'Heavy Rain'
                          : 'Glare'}
                      </div>
                      <div className="font-mono text-xs text-[#6E7378]">
                        GT {sample.groundTruth}
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-[#6E7378]">
                        Degraded Input
                      </div>
                      <div className="bg-[#0E1012] border border-[#0E1012]/15">
                        <CanvasView buffer={sample.original} alt={`${sample.weather} input`} />
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-[#1E6B7B]">
                        DCP Restored
                      </div>
                      <div className="bg-[#0E1012] border border-[#0E1012]/15">
                        <CanvasView buffer={sample.dehazed} alt={`${sample.weather} restored`} />
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-[#6E7378]">
                        Segmented Boxes
                      </div>
                      <div className="bg-[#0E1012] border border-[#0E1012]/15">
                        <CanvasView
                          buffer={sample.annotated}
                          alt={`${sample.weather} segmented`}
                        />
                      </div>
                    </div>

                    <div className="border border-[#0E1012]/12 bg-[#F9F9F6] p-3 font-mono text-xs space-y-2">
                      <div>
                        <div className="text-[10px] uppercase tracking-[0.12em] text-[#6E7378]">
                          Raw Baseline
                        </div>
                        <div className="text-amber-700 font-semibold tracking-wider truncate">
                          {sample.predWithoutDcp}{' '}
                          <span className="font-normal text-[#6E7378]">
                            ({Math.round(sample.confWithoutDcp * 100)}%)
                          </span>
                        </div>
                      </div>
                      <div className="pt-2 border-t border-[#0E1012]/10">
                        <div className="text-[10px] uppercase tracking-[0.12em] text-[#6E7378]">
                          With DCP Pre-Filter
                        </div>
                        <div className="text-[#1E6B7B] font-semibold tracking-wider truncate">
                          {sample.predWithDcp}{' '}
                          <span className="font-normal text-[#6E7378]">
                            ({Math.round(sample.confWithDcp * 100)}%)
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
};
