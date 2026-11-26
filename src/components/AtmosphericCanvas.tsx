import React, { useEffect, useRef } from 'react';

interface RainStreak {
  x: number;
  y: number;
  len: number;
  speed: number;
  bright: boolean;
}

export const AtmosphericCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let rafId = 0;
    let width = 0;
    let height = 0;
    let isVisible = true;
    let lastTime = 0;
    const frameIntervalMs = 42;

    const streaks: RainStreak[] = [];

    const initParticles = (w: number, h: number) => {
      streaks.length = 0;
      const streakCount = Math.min(54, Math.max(24, Math.floor(w / 28)));
      for (let i = 0; i < streakCount; i++) {
        streaks.push({
          x: Math.random() * w,
          y: Math.random() * h,
          len: 20 + Math.random() * 40,
          speed: 3.4 + Math.random() * 4.4,
          bright: i % 3 === 0,
        });
      }
    };

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      width = parent.clientWidth;
      height = parent.clientHeight;
      canvas.width = Math.max(1, Math.floor(width));
      canvas.height = Math.max(1, Math.floor(height));
      initParticles(width, height);
    };

    const drawFrame = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.lineWidth = 1;

      ctx.strokeStyle = 'rgba(244, 244, 240, 0.065)';
      ctx.beginPath();
      for (const s of streaks) {
        if (!s.bright) {
          ctx.moveTo(s.x, s.y);
          ctx.lineTo(s.x - s.len * 0.16, s.y + s.len);
        }
      }
      ctx.stroke();

      ctx.strokeStyle = 'rgba(200, 224, 230, 0.12)';
      ctx.beginPath();
      for (const s of streaks) {
        if (s.bright) {
          ctx.moveTo(s.x, s.y);
          ctx.lineTo(s.x - s.len * 0.16, s.y + s.len);
        }
      }
      ctx.stroke();

      if (!reducedMotion) {
        for (const s of streaks) {
          s.x -= s.speed * 0.16;
          s.y += s.speed;
          if (s.y > height + s.len) {
            s.y = -s.len;
            s.x = Math.random() * (width + 120);
          }
        }
      }
    };

    const tick = (now: number) => {
      if (isVisible && now - lastTime >= frameIntervalMs) {
        lastTime = now;
        drawFrame();
      }
      if (!reducedMotion) {
        rafId = window.requestAnimationFrame(tick);
      }
    };

    const observer = new IntersectionObserver(
      (entries) => {
        isVisible = entries[0]?.isIntersecting ?? true;
      },
      { threshold: 0.05 }
    );
    observer.observe(canvas);

    resize();
    drawFrame();
    if (!reducedMotion) {
      rafId = window.requestAnimationFrame(tick);
    }

    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      observer.disconnect();
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      <div
        className="absolute inset-0 opacity-90"
        style={{
          backgroundImage:
            'radial-gradient(62% 54% at 22% 32%, rgba(30, 107, 123, 0.14) 0%, rgba(14, 16, 18, 0) 70%), radial-gradient(52% 48% at 78% 62%, rgba(210, 218, 222, 0.07) 0%, rgba(14, 16, 18, 0) 68%)',
        }}
      />
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
    </div>
  );
};
