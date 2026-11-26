import React, { useCallback, useEffect, useRef, useState } from 'react';
import { MoveHorizontal } from 'lucide-react';
import { RgbaBuffer } from '../vision/types';

interface BeforeAfterSliderProps {
  beforeBuffer: RgbaBuffer;
  afterBuffer: RgbaBuffer;
  beforeLabel: string;
  afterLabel: string;
  dcpEnabled: boolean;
}

function paintRgbaToCanvas(canvas: HTMLCanvasElement | null, buffer: RgbaBuffer): void {
  if (!canvas) return;
  canvas.width = buffer.width;
  canvas.height = buffer.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const imgData = ctx.createImageData(buffer.width, buffer.height);
  imgData.data.set(buffer.data);
  ctx.putImageData(imgData, 0, 0);
}

export const BeforeAfterSlider: React.FC<BeforeAfterSliderProps> = ({
  beforeBuffer,
  afterBuffer,
  beforeLabel,
  afterLabel,
  dcpEnabled,
}) => {
  const [splitPct, setSplitPct] = useState<number>(52);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const beforeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const afterCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    paintRgbaToCanvas(beforeCanvasRef.current, beforeBuffer);
  }, [beforeBuffer]);

  useEffect(() => {
    paintRgbaToCanvas(afterCanvasRef.current, afterBuffer);
  }, [afterBuffer]);

  const updateFromClientX = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSplitPct(Math.max(4, Math.min(96, pct)));
  }, []);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setIsDragging(true);
    updateFromClientX(e.clientX);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    updateFromClientX(e.clientX);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    setIsDragging(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setSplitPct((v) => Math.max(4, v - 4));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setSplitPct((v) => Math.min(96, v + 4));
    }
  };

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between text-[11px] font-mono uppercase tracking-[0.14em] text-[#6E7378]">
        <span>{beforeLabel}</span>
        <span className="text-[#1E6B7B]">
          {dcpEnabled ? afterLabel : 'DCP BYPASSED · PASS-THROUGH'}
        </span>
      </div>

      <div
        ref={containerRef}
        role="slider"
        tabIndex={0}
        aria-label="Before and after Dark-Channel de-hazing comparison slider"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(splitPct)}
        data-testid="before-after-slider"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onKeyDown={handleKeyDown}
        className="relative w-full select-none overflow-hidden border border-[#0E1012]/15 bg-[#0E1012] cursor-ew-resize focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1E6B7B]"
      >
        <canvas
          ref={afterCanvasRef}
          aria-label={afterLabel}
          className="block w-full h-auto"
        />

        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - splitPct}% 0 0)` }}
        >
          <canvas
            ref={beforeCanvasRef}
            aria-label={beforeLabel}
            className="block w-full h-auto"
          />
        </div>

        <div
          className="pointer-events-none absolute top-0 bottom-0 w-px bg-[#F4F4F0] shadow-[0_0_0_1px_rgba(14,16,18,0.35)]"
          style={{ left: `${splitPct}%` }}
        >
          <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 rounded-full bg-[#0E1012] border border-[#F4F4F0]/80 text-[#F4F4F0] flex items-center justify-center shadow-sm">
            <MoveHorizontal className="w-4 h-4" />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between text-[11px] font-mono text-[#6E7378]">
        <span>Drag divider or use arrow keys to inspect restoration boundary</span>
        <span className="tabular-nums">{Math.round(splitPct)}% / {Math.round(100 - splitPct)}%</span>
      </div>
    </div>
  );
};
