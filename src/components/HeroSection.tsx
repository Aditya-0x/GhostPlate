import React from 'react';
import { motion } from 'motion/react';
import { ArrowDown, ArrowRight } from 'lucide-react';
import { AtmosphericCanvas } from './AtmosphericCanvas';

interface HeroSectionProps {
  onLaunchLab: () => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({ onLaunchLab }) => {
  return (
    <section
      id="overview"
      data-section-theme="dark"
      className="relative min-h-screen w-full bg-[#0E1012] text-[#F4F4F0] flex flex-col justify-between overflow-hidden border-b border-[#F4F4F0]/10"
    >
      <AtmosphericCanvas />

      <div className="relative z-10 w-full max-w-[1440px] mx-auto px-6 sm:px-12 xl:pr-24 pt-32 sm:pt-36 pb-10 flex-1 flex flex-col justify-center">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-8 items-end">
          <div className="lg:col-span-9 space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
              className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] uppercase tracking-[0.18em] text-[#8D9399]"
            >
              <span>01 / 05</span>
              <span aria-hidden="true">·</span>
              <span>Group 6 Computer Vision Research</span>
              <span aria-hidden="true">·</span>
              <span className="text-[#4CA8B8]">Adverse-Weather ALPR</span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
              className="font-display font-light tracking-[-0.035em] leading-[0.98] text-[clamp(3rem,6.8vw,6.25rem)] text-[#F4F4F0] [text-wrap:balance]"
            >
              Where clarity takes shape.
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.65, delay: 0.16, ease: [0.16, 1, 0.3, 1] }}
              className="max-w-2xl text-base sm:text-lg font-light leading-relaxed text-[#A6ACB2]"
            >
              An Automated License Plate Character Recognition pipeline engineered to recover
              legibility under heavy rain, dense atmospheric fog, and nighttime headlight glare
              through a physics-based Dark-Channel Prior de-hazing pre-filter and a compact
              68K-parameter PyTorch CNN.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.24, ease: [0.16, 1, 0.3, 1] }}
              className="pt-2 flex flex-wrap items-center gap-6"
            >
              <button
                type="button"
                onClick={onLaunchLab}
                className="group inline-flex items-center gap-3 px-7 py-3.5 rounded-full border border-[#F4F4F0]/35 hover:border-[#F4F4F0] hover:bg-[#F4F4F0] hover:text-[#0E1012] text-xs font-mono uppercase tracking-[0.16em] text-[#F4F4F0] transition-all duration-200 cursor-pointer whitespace-nowrap"
              >
                <span>Launch Vision Lab</span>
                <ArrowRight className="w-3.5 h-3.5 transition-transform duration-200 group-hover:translate-x-1" />
              </button>

              <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                Dark-Channel Prior · Guided Filter · 36-Class CNN
              </div>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.28, ease: [0.16, 1, 0.3, 1] }}
            className="lg:col-span-3 border-t lg:border-t-0 lg:border-l border-[#F4F4F0]/12 pt-6 lg:pt-0 lg:pl-8 space-y-5"
          >
            <div>
              <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                Verified DCP Recovery
              </div>
              <div className="mt-1 font-mono text-2xl font-medium tabular-nums text-[#F4F4F0]">
                +43.3 <span className="text-xs text-[#4CA8B8]">pp</span>
              </div>
              <div className="text-xs text-[#8D9399] mt-0.5">
                Mean accuracy lift across rain, fog & glare
              </div>
            </div>
            <div className="pt-4 border-t border-[#F4F4F0]/10">
              <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6E7378]">
                Model Footprint
              </div>
              <div className="mt-1 font-mono text-2xl font-medium tabular-nums text-[#F4F4F0]">
                68,004 <span className="text-xs text-[#8D9399]">params</span>
              </div>
              <div className="text-xs text-[#8D9399] mt-0.5">
                Runs on free Colab T4 GPU or laptop CPU
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      <div className="relative z-10 w-full max-w-[1440px] mx-auto px-6 sm:px-12 pb-8 pt-4 border-t border-[#F4F4F0]/10 flex items-center justify-between gap-4">
        <div className="text-xs text-[#8D9399] truncate">
          Aditya Verma · Varun Singh Rajpurohit · Damandeep Saini
        </div>
        <button
          type="button"
          onClick={onLaunchLab}
          className="inline-flex items-center gap-2.5 font-mono text-[11px] uppercase tracking-[0.18em] text-[#A6ACB2] hover:text-[#F4F4F0] transition-colors cursor-pointer whitespace-nowrap shrink-0"
        >
          <span>Scroll to Discover</span>
          <ArrowDown className="w-3.5 h-3.5" />
        </button>
      </div>
    </section>
  );
};
