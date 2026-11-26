import React, { useEffect, useRef } from 'react';
import { GrayBuffer, RgbaBuffer } from '../vision/types';

interface CanvasViewProps {
  buffer: RgbaBuffer | GrayBuffer;
  kind?: 'rgba' | 'gray';
  alt: string;
  className?: string;
  pixelated?: boolean;
  testId?: string;
}

export const CanvasView: React.FC<CanvasViewProps> = ({
  buffer,
  kind = 'rgba',
  alt,
  className = '',
  pixelated = false,
  testId,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = buffer.width;
    canvas.height = buffer.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(buffer.width, buffer.height);
    if (kind === 'rgba') {
      imgData.data.set(buffer.data);
    } else {
      const g = buffer.data;
      const d = imgData.data;
      for (let i = 0; i < g.length; i++) {
        const v = g[i];
        const o = i * 4;
        d[o] = v;
        d[o + 1] = v;
        d[o + 2] = v;
        d[o + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [buffer, kind]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={alt}
      data-testid={testId}
      className={`block w-full h-auto ${pixelated ? '[image-rendering:pixelated]' : ''} ${className}`}
    />
  );
};
