import { useEffect, useState } from 'react';
import { ArchitectureSection } from './components/ArchitectureSection';
import { BenchmarksSection } from './components/BenchmarksSection';
import { HeroSection } from './components/HeroSection';
import { MenuOverlay, NavSectionItem } from './components/MenuOverlay';
import { TeamCodeSection } from './components/TeamCodeSection';
import { VisionLabSection } from './components/VisionLabSection';
import {
  buildColabNotebookJson,
  createZipBytes,
  triggerBrowserDownload,
} from './utils/exportProject';
import { FULL_EXPORT_FILES } from './utils/repoFiles';

const SECTIONS: Array<NavSectionItem & { theme: 'dark' | 'light' }> = [
  {
    id: 'overview',
    index: '01',
    title: 'Overview',
    subtitle: 'Where clarity takes shape',
    theme: 'dark',
  },
  {
    id: 'lab',
    index: '02',
    title: 'Live Lab',
    subtitle: 'Interactive Dark-Channel ALPR workbench',
    theme: 'light',
  },
  {
    id: 'pipeline',
    index: '03',
    title: 'Pipeline',
    subtitle: 'Four-stage OpenCV & PyTorch methodology',
    theme: 'dark',
  },
  {
    id: 'benchmarks',
    index: '04',
    title: 'Benchmarks',
    subtitle: 'Empirical Rain, Fog & Glare ablation study',
    theme: 'light',
  },
  {
    id: 'team',
    index: '05',
    title: 'Team & Code',
    subtitle: 'Group 6 credits & Python/Colab repository',
    theme: 'dark',
  },
];

export default function App() {
  const [activeIndex, setActiveIndex] = useState<number>(0);
  const [menuOpen, setMenuOpen] = useState<boolean>(false);

  useEffect(() => {
    const elements = SECTIONS.map((s) => document.getElementById(s.id)).filter(
      (el): el is HTMLElement => el !== null
    );
    if (elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        let bestIdx = -1;
        let bestRatio = 0;
        for (const entry of entries) {
          if (entry.isIntersecting && entry.intersectionRatio > bestRatio) {
            const idx = SECTIONS.findIndex((s) => s.id === entry.target.id);
            if (idx !== -1) {
              bestRatio = entry.intersectionRatio;
              bestIdx = idx;
            }
          }
        }
        if (bestIdx !== -1) {
          setActiveIndex(bestIdx);
        }
      },
      { threshold: [0.15, 0.35, 0.6], rootMargin: '-12% 0px -35% 0px' }
    );

    for (const el of elements) observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const scrollToSection = (sectionId: string) => {
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleDownloadZip = () => {
    const bytes = createZipBytes(FULL_EXPORT_FILES);
    triggerBrowserDownload(bytes, 'clearplate-alpr-group6.zip', 'application/zip');
  };

  const handleDownloadColab = () => {
    const nb = buildColabNotebookJson(FULL_EXPORT_FILES);
    triggerBrowserDownload(nb, 'Adverse_Weather_ALPR_Group6.ipynb', 'application/x-ipynb+json');
  };

  const activeSection = SECTIONS[activeIndex] ?? SECTIONS[0];
  const isLightHeader = activeSection.theme === 'light';

  return (
    <div className="min-h-screen w-full bg-[#0E1012] text-[#F4F4F0]">
      <header
        className={`fixed top-0 left-0 right-0 z-40 transition-colors duration-300 border-b backdrop-blur-md ${
          isLightHeader
            ? 'bg-[#F4F4F0]/90 text-[#0E1012] border-[#0E1012]/12'
            : 'bg-[#0E1012]/85 text-[#F4F4F0] border-[#F4F4F0]/12'
        }`}
      >
        <div className="w-full max-w-[1440px] mx-auto px-6 sm:px-12 h-20 flex items-center justify-between gap-8">
          <button
            type="button"
            onClick={() => scrollToSection('overview')}
            className="font-display text-lg font-medium tracking-[-0.03em] cursor-pointer whitespace-nowrap shrink-0"
          >
            theALPR
          </button>

          <div className="flex items-center gap-6 sm:gap-8 shrink-0">
            <button
              type="button"
              onClick={() => scrollToSection('overview')}
              className={`font-mono text-xs uppercase tracking-[0.16em] py-1 border-b border-transparent transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                isLightHeader
                  ? 'text-[#0E1012]/75 hover:text-[#0E1012] hover:border-[#0E1012]'
                  : 'text-[#F4F4F0]/75 hover:text-[#F4F4F0] hover:border-[#F4F4F0]'
              }`}
            >
              Home
            </button>

            <button
              type="button"
              aria-expanded={menuOpen}
              aria-label="Open navigation menu"
              onClick={() => setMenuOpen(true)}
              className={`font-mono text-xs uppercase tracking-[0.16em] py-1 border-b border-transparent transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                isLightHeader
                  ? 'text-[#0E1012] hover:border-[#0E1012]'
                  : 'text-[#F4F4F0] hover:border-[#F4F4F0]'
              }`}
            >
              Menu
            </button>
          </div>
        </div>
      </header>

      <aside
        aria-label="Section progress indicator"
        className={`hidden xl:flex fixed right-6 top-1/2 -translate-y-1/2 z-30 flex-col items-end gap-3 transition-colors duration-300 ${
          isLightHeader ? 'text-[#0E1012]' : 'text-[#F4F4F0]'
        }`}
      >
        <div
          data-testid="section-progress"
          className="font-mono text-[11px] tracking-[0.16em] tabular-nums opacity-75"
        >
          {activeIndex + 1} / {SECTIONS.length}
        </div>
        <div className="flex flex-col items-end gap-2">
          {SECTIONS.map((s, idx) => {
            const active = idx === activeIndex;
            return (
              <button
                key={s.id}
                type="button"
                aria-label={`Jump to section ${s.index} ${s.title}`}
                onClick={() => scrollToSection(s.id)}
                className="py-1 pl-3 cursor-pointer flex items-center"
              >
                <span
                  className={`block h-[1.5px] transition-all duration-300 ${
                    active
                      ? 'w-7 bg-[#1E6B7B]'
                      : isLightHeader
                      ? 'w-3 bg-[#0E1012]/25 hover:bg-[#0E1012]/60'
                      : 'w-3 bg-[#F4F4F0]/25 hover:bg-[#F4F4F0]/60'
                  }`}
                />
              </button>
            );
          })}
        </div>
      </aside>

      <MenuOverlay
        isOpen={menuOpen}
        activeSectionId={activeSection.id}
        sections={SECTIONS}
        onClose={() => setMenuOpen(false)}
        onNavigate={scrollToSection}
        onDownloadZip={handleDownloadZip}
        onDownloadColab={handleDownloadColab}
      />

      <main>
        <HeroSection onLaunchLab={() => scrollToSection('lab')} />
        <VisionLabSection />
        <ArchitectureSection />
        <BenchmarksSection />
        <TeamCodeSection />
      </main>
    </div>
  );
}
