import requirementsTxt from '../../requirements.txt?raw';
import readmeMd from '../../README.md?raw';
import appPy from '../../app.py?raw';
import initPy from '../__init__.py?raw';
import dehazePy from '../dehaze.py?raw';
import segmentPy from '../segment.py?raw';
import modelPy from '../model.py?raw';
import trainPy from '../train.py?raw';
import { ProjectFileEntry } from './exportProject';

export const REPOSITORY_FILES: ProjectFileEntry[] = [
  {
    path: 'src/dehaze.py',
    language: 'python',
    summary: 'Dark-Channel Prior from scratch, fast O(N) Guided Filter, adaptive gamma, LAB CLAHE & bilateral glare suppression',
    content: dehazePy,
  },
  {
    path: 'src/segment.py',
    language: 'python',
    summary: 'Morphological blackhat + Sobel-X plate localization, pre-crop fallback, adaptive threshold & 8-CC character segmentation',
    content: segmentPy,
  },
  {
    path: 'src/model.py',
    language: 'python',
    summary: 'Lightweight 4-block Depthwise-Separable PyTorch CNN (68,004 parameters, 36 classes 0–9 & A–Z) with checkpoint I/O',
    content: modelPy,
  },
  {
    path: 'src/train.py',
    language: 'python',
    summary: 'Kaggle Adverse Weather loader, synthetic Rain/Fog/Glare generator, PyTorch training loop & WITH vs WITHOUT DCP ablation study',
    content: trainPy,
  },
  {
    path: 'app.py',
    language: 'python',
    summary: 'Dark-mode Gradio interactive demo UI with DCP toggle, intermediate maps, character strip, confidence & latency readout',
    content: appPy,
  },
  {
    path: 'requirements.txt',
    language: 'text',
    summary: '100% free, open-source Python dependencies for Colab T4 GPU or laptop CPU',
    content: requirementsTxt,
  },
  {
    path: 'README.md',
    language: 'markdown',
    summary: 'Setup guide, Mermaid system architecture diagram, and Google Colab T4 quickstart cells',
    content: readmeMd,
  },
];

export const FULL_EXPORT_FILES = [
  ...REPOSITORY_FILES.map(({ path, content }) => ({ path, content })),
  { path: 'src/__init__.py', content: initPy },
];
