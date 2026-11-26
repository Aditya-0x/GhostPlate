import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Check, Copy, Download } from 'lucide-react';
import {
  buildColabNotebookJson,
  createZipBytes,
  triggerBrowserDownload,
} from '../utils/exportProject';
import { FULL_EXPORT_FILES, REPOSITORY_FILES } from '../utils/repoFiles';

const TEAM_MEMBERS = [
  {
    index: '01',
    name: 'Aditya Verma',
    regId: '12408804',
    focus: 'Dark-Channel Prior De-Hazing, Guided Filter & Glare Mitigation',
  },
  {
    index: '02',
    name: 'Varun Singh Rajpurohit',
    regId: '12409154',
    focus: 'Morphological Localization, Connected-Component Segmentation & Ablation',
  },
  {
    index: '03',
    name: 'Damandeep Saini',
    regId: '12403195',
    focus: 'Lightweight Depthwise-Separable PyTorch CNN, Colab Pipeline & UI',
  },
];

export const TeamCodeSection: React.FC = () => {
  const [selectedPath, setSelectedPath] = useState<string>(REPOSITORY_FILES[0].path);
  const [copiedPath, setCopiedPath] = useState<string | null>(null);

  const activeFile =
    REPOSITORY_FILES.find((f) => f.path === selectedPath) ?? REPOSITORY_FILES[0];

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(activeFile.content);
      setCopiedPath(activeFile.path);
      window.setTimeout(() => setCopiedPath(null), 1800);
    } catch {
      setCopiedPath(null);
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

  const codeLines = activeFile.content.split('\n');

  return (
    <section
      id="team"
      data-section-theme="dark"
      className="min-h-screen w-full bg-[#0E1012] text-[#F4F4F0]"
    >
      <div className="w-full max-w-[1440px] mx-auto px-6 sm:px-12 xl:pr-24 pt-24 sm:pt-28 pb-16 space-y-14">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
          className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-end pb-8 border-b border-[#F4F4F0]/12"
        >
          <div className="lg:col-span-7 space-y-3">
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#8D9399]">
              05 / 05 · Research Team & Open-Source Codebase
            </div>
            <h2 className="font-display text-3xl sm:text-5xl font-light tracking-[-0.03em] leading-[1.06] [text-wrap:balance]">
              Group 6 · Complete OpenCV & PyTorch repository.
            </h2>
          </div>

          <div className="lg:col-span-5 flex flex-wrap lg:justify-end gap-3">
            <button
              type="button"
              onClick={handleDownloadColab}
              className="inline-flex items-center gap-2 px-4 py-2.5 border border-[#F4F4F0]/25 hover:border-[#F4F4F0] text-xs font-mono uppercase tracking-[0.12em] text-[#F4F4F0] transition-colors cursor-pointer whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Colab Notebook (.ipynb)</span>
            </button>
            <button
              type="button"
              onClick={handleDownloadZip}
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-[#1E6B7B] hover:bg-[#257f92] border border-[#1E6B7B] text-xs font-mono uppercase tracking-[0.12em] text-[#F4F4F0] transition-colors cursor-pointer whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Repo (.zip)</span>
            </button>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-[#F4F4F0]/12 border border-[#F4F4F0]/12">
          {TEAM_MEMBERS.map((member) => (
            <div key={member.regId} className="bg-[#0E1012] p-6 sm:p-8 space-y-4">
              <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.16em] text-[#6E7378]">
                <span>Investigator {member.index}</span>
                <span className="text-[#4CA8B8]">{member.regId}</span>
              </div>
              <div className="font-display text-2xl font-light tracking-[-0.02em] text-[#F4F4F0]">
                {member.name}
              </div>
              <p className="text-xs text-[#8D9399] leading-relaxed">{member.focus}</p>
            </div>
          ))}
        </div>

        <div className="border border-[#F4F4F0]/15 bg-[#121518]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#F4F4F0]/12 px-4 py-3">
            <div
              role="tablist"
              aria-label="Python codebase files"
              className="flex flex-wrap items-center gap-1"
            >
              {REPOSITORY_FILES.map((file) => {
                const isSelected = file.path === activeFile.path;
                const shortName = file.path.replace('src/', '');
                return (
                  <button
                    key={file.path}
                    type="button"
                    role="tab"
                    aria-selected={isSelected}
                    onClick={() => setSelectedPath(file.path)}
                    className={`px-3 py-1.5 font-mono text-xs transition-colors cursor-pointer whitespace-nowrap ${
                      isSelected
                        ? 'bg-[#1E6B7B] text-[#F4F4F0]'
                        : 'text-[#8D9399] hover:text-[#F4F4F0] hover:bg-[#F4F4F0]/5'
                    }`}
                  >
                    {shortName}
                  </button>
                );
              })}
            </div>

            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-[#F4F4F0]/20 hover:border-[#F4F4F0]/60 font-mono text-[11px] uppercase tracking-[0.12em] text-[#F4F4F0] transition-colors cursor-pointer whitespace-nowrap shrink-0"
            >
              {copiedPath === activeFile.path ? (
                <>
                  <Check className="w-3.5 h-3.5 text-[#4CA8B8]" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy File</span>
                </>
              )}
            </button>
          </div>

          <div className="px-4 py-2.5 border-b border-[#F4F4F0]/10 bg-[#0E1012] flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="font-mono text-[#4CA8B8]" data-testid="active-repo-file">
              {activeFile.path}
            </span>
            <span className="text-[#8D9399] truncate">{activeFile.summary}</span>
          </div>

          <div className="overflow-auto max-h-[500px] bg-[#0B0D0F] p-4 sm:p-5">
            <pre className="text-xs leading-5 font-mono text-[#D8DDE2]">
              <code>
                {codeLines.map((line, i) => (
                  <div key={i} className="flex">
                    <span className="w-10 pr-4 text-right select-none text-[#4F565E] tabular-nums shrink-0">
                      {i + 1}
                    </span>
                    <span className="whitespace-pre">{line}</span>
                  </div>
                ))}
              </code>
            </pre>
          </div>
        </div>

        <footer className="pt-8 border-t border-[#F4F4F0]/12 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-mono text-[#6E7378]">
          <div>theALPR · Automated License Plate Character Recognition under Rain, Fog, and Glare</div>
          <div>Group 6 · Open-Source OpenCV, PyTorch & Kaggle Adverse Weather Ready</div>
        </footer>
      </div>
    </section>
  );
};
