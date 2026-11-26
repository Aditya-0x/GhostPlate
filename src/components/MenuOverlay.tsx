import React, { useEffect } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { ArrowUpRight, Download, X } from 'lucide-react';

export interface NavSectionItem {
  id: string;
  index: string;
  title: string;
  subtitle: string;
}

interface MenuOverlayProps {
  isOpen: boolean;
  activeSectionId: string;
  sections: NavSectionItem[];
  onClose: () => void;
  onNavigate: (sectionId: string) => void;
  onDownloadZip: () => void;
  onDownloadColab: () => void;
}

export const MenuOverlay: React.FC<MenuOverlayProps> = ({
  isOpen,
  activeSectionId,
  sections,
  onClose,
  onNavigate,
  onDownloadZip,
  onDownloadColab,
}) => {
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', handleKey);
    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', handleKey);
    };
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          role="dialog"
          aria-modal="true"
          aria-label="Site navigation menu"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
          className="fixed inset-0 z-50 bg-[#0E1012] text-[#F4F4F0] flex flex-col justify-between"
        >
          <div className="w-full max-w-[1440px] mx-auto px-6 sm:px-12 h-20 flex items-center justify-between border-b border-[#F4F4F0]/12">
            <span className="font-display text-lg font-medium tracking-[-0.02em]">theALPR</span>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-[0.16em] text-[#F4F4F0]/80 hover:text-[#F4F4F0] transition-colors cursor-pointer py-2 px-1"
            >
              <span>Close</span>
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 w-full max-w-[1440px] mx-auto px-6 sm:px-12 py-10 grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-12 items-center overflow-y-auto">
            <nav aria-label="Full screen editorial index" className="divide-y divide-[#F4F4F0]/12">
              {sections.map((section, idx) => {
                const isActive = section.id === activeSectionId;
                return (
                  <motion.button
                    key={section.id}
                    type="button"
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                      delay: 0.05 + idx * 0.045,
                      duration: 0.36,
                      ease: [0.16, 1, 0.3, 1],
                    }}
                    onClick={() => {
                      onNavigate(section.id);
                      onClose();
                    }}
                    className="group w-full py-5 sm:py-6 flex items-baseline justify-between gap-6 text-left cursor-pointer"
                  >
                    <div className="flex items-baseline gap-6 sm:gap-10 min-w-0">
                      <span
                        className={`font-mono text-xs tracking-[0.16em] shrink-0 ${
                          isActive ? 'text-[#4CA8B8]' : 'text-[#6E7378] group-hover:text-[#F4F4F0]'
                        }`}
                      >
                        {section.index}
                      </span>
                      <span
                        className={`font-display text-3xl sm:text-5xl lg:text-6xl font-light tracking-[-0.03em] transition-transform duration-300 group-hover:translate-x-2 ${
                          isActive ? 'text-[#F4F4F0]' : 'text-[#F4F4F0]/80 group-hover:text-[#F4F4F0]'
                        }`}
                      >
                        {section.title}
                      </span>
                    </div>
                    <div className="hidden md:flex items-center gap-3 text-xs text-[#8D9399] group-hover:text-[#F4F4F0] transition-colors shrink-0">
                      <span>{section.subtitle}</span>
                      <ArrowUpRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                    </div>
                  </motion.button>
                );
              })}
            </nav>

            <div className="border-t lg:border-t-0 lg:border-l border-[#F4F4F0]/12 pt-8 lg:pt-0 lg:pl-10 space-y-8">
              <div className="space-y-2">
                <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#6E7378]">
                  Research Group 06
                </div>
                <p className="text-sm text-[#B8BEC4] leading-relaxed">
                  Automated License Plate Character Recognition under Rain, Fog, and Glare.
                </p>
                <div className="pt-2 space-y-1 text-xs font-mono text-[#8D9399]">
                  <div>Aditya Verma · 12408804</div>
                  <div>Varun Singh Rajpurohit · 12409154</div>
                  <div>Damandeep Saini · 12403195</div>
                </div>
              </div>

              <div className="space-y-2.5 pt-6 border-t border-[#F4F4F0]/12">
                <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#6E7378]">
                  Open-Source Artifacts
                </div>
                <button
                  type="button"
                  onClick={onDownloadZip}
                  className="w-full flex items-center justify-between px-4 py-3 border border-[#F4F4F0]/20 hover:border-[#4CA8B8] text-xs font-mono uppercase tracking-[0.12em] text-[#F4F4F0] transition-colors cursor-pointer"
                >
                  <span>Download Python Repo (.zip)</span>
                  <Download className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  onClick={onDownloadColab}
                  className="w-full flex items-center justify-between px-4 py-3 border border-[#F4F4F0]/12 hover:border-[#F4F4F0]/40 text-xs font-mono uppercase tracking-[0.12em] text-[#B8BEC4] hover:text-[#F4F4F0] transition-colors cursor-pointer"
                >
                  <span>Google Colab Notebook (.ipynb)</span>
                  <Download className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>

          <div className="w-full max-w-[1440px] mx-auto px-6 sm:px-12 py-5 border-t border-[#F4F4F0]/12 flex items-center justify-between text-[11px] font-mono uppercase tracking-[0.14em] text-[#6E7378]">
            <span>OpenCV · PyTorch · Guided Filter DCP</span>
            <span>ESC to close</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
