export interface ProjectFileEntry {
  path: string;
  language: string;
  summary: string;
  content: string;
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Pure zero-dependency PKZIP archive builder (Store method 0, valid CRC-32 & central directory).
 * Works identically in browsers and Node.js.
 */
export function createZipBytes(files: Array<{ path: string; content: string }>): Uint8Array {
  const encoder = new TextEncoder();
  const localChunks: Uint8Array[] = [];
  const centralChunks: Uint8Array[] = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = encoder.encode(file.path);
    const dataBytes = encoder.encode(file.content);
    const checksum = crc32(dataBytes);

    // Local file header (30 bytes + filename)
    const localHeader = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(localHeader.buffer);
    lv.setUint32(0, 0x04034b50, true); // Local file header signature
    lv.setUint16(4, 20, true); // Version needed
    lv.setUint16(6, 0x0800, true); // UTF-8 flag
    lv.setUint16(8, 0, true); // Compression method: store
    lv.setUint16(10, 0, true); // Mod time
    lv.setUint16(12, 0, true); // Mod date
    lv.setUint32(14, checksum, true);
    lv.setUint32(18, dataBytes.length, true);
    lv.setUint32(22, dataBytes.length, true);
    lv.setUint16(26, nameBytes.length, true);
    lv.setUint16(28, 0, true);
    localHeader.set(nameBytes, 30);

    localChunks.push(localHeader, dataBytes);

    // Central directory header (46 bytes + filename)
    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(centralHeader.buffer);
    cv.setUint32(0, 0x02014b50, true); // Central file header signature
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, 0, true);
    cv.setUint16(14, 0, true);
    cv.setUint32(16, checksum, true);
    cv.setUint32(20, dataBytes.length, true);
    cv.setUint32(24, dataBytes.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint16(30, 0, true);
    cv.setUint16(32, 0, true);
    cv.setUint16(34, 0, true);
    cv.setUint16(36, 0, true);
    cv.setUint32(38, 0, true);
    cv.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);

    centralChunks.push(centralHeader);
    offset += localHeader.length + dataBytes.length;
  }

  const centralSize = centralChunks.reduce((sum, c) => sum + c.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true); // End of central dir signature
  ev.setUint16(4, 0, true);
  ev.setUint16(6, 0, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);
  ev.setUint16(20, 0, true);

  const totalSize = offset + centralSize + eocd.length;
  const archive = new Uint8Array(totalSize);
  let cursor = 0;
  for (const chunk of localChunks) {
    archive.set(chunk, cursor);
    cursor += chunk.length;
  }
  for (const chunk of centralChunks) {
    archive.set(chunk, cursor);
    cursor += chunk.length;
  }
  archive.set(eocd, cursor);
  return archive;
}

/**
 * Generate a self-contained Google Colab Jupyter Notebook (.ipynb) that writes
 * all project modules cleanly into the Colab workspace, installs dependencies,
 * trains the Lightweight PyTorch CNN, runs the ablation study, displays plots,
 * and launches the public dark-mode Gradio demo.
 */
export function buildColabNotebookJson(files: Array<{ path: string; content: string }>): string {
  const toLines = (text: string): string[] =>
    text.split('\n').map((line, idx, arr) => (idx < arr.length - 1 ? `${line}\n` : line));

  const cells: Array<Record<string, unknown>> = [
    {
      cell_type: 'markdown',
      metadata: {},
      source: toLines(
        '# Automated License Plate Character Recognition under Rain, Fog, and Glare\n\n' +
          '**Team:** Group 6 — Aditya Verma (12408804) · Varun Singh Rajpurohit (12409154) · Damandeep Saini (12403195)\n\n' +
          'Run all cells on a free **Google Colab T4 GPU** (or CPU) to bootstrap the full repository, train the Lightweight PyTorch CNN, execute the WITH vs. WITHOUT Dark-Channel De-hazing ablation study, and launch the Gradio dark-mode UI.'
      ),
    },
    {
      cell_type: 'code',
      execution_count: null,
      metadata: {},
      outputs: [],
      source: toLines('import os\nos.makedirs("src", exist_ok=True)\nprint("Workspace initialized.")'),
    },
  ];

  for (const file of files) {
    if (file.path === 'README.md') continue;
    cells.push({
      cell_type: 'code',
      execution_count: null,
      metadata: {},
      outputs: [],
      source: toLines(`%%writefile ${file.path}\n${file.content}`),
    });
  }

  cells.push(
    {
      cell_type: 'code',
      execution_count: null,
      metadata: {},
      outputs: [],
      source: toLines('!pip install -q -r requirements.txt'),
    },
    {
      cell_type: 'code',
      execution_count: null,
      metadata: {},
      outputs: [],
      source: toLines(
        '!python -m src.train --epochs 12 --eval-per-weather 30 --output-dir outputs'
      ),
    },
    {
      cell_type: 'code',
      execution_count: null,
      metadata: {},
      outputs: [],
      source: toLines(
        'from IPython.display import Image, display\n' +
          'display(Image("outputs/ablation_comparison.png"))\n' +
          'display(Image("outputs/confusion_matrix.png"))'
      ),
    },
    {
      cell_type: 'code',
      execution_count: null,
      metadata: {},
      outputs: [],
      source: toLines('from app import build_demo\ndemo = build_demo()\ndemo.launch(share=True)'),
    }
  );

  const notebook = {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      colab: {
        name: 'Adverse_Weather_ALPR_Group6.ipynb',
        provenance: [],
        gpuType: 'T4',
      },
      kernelspec: {
        display_name: 'Python 3',
        name: 'python3',
      },
      language_info: {
        name: 'python',
      },
    },
    cells,
  };

  return JSON.stringify(notebook, null, 2);
}

/** Trigger a client-side file download cleanly via an object URL. */
export function triggerBrowserDownload(bytes: Uint8Array | string, filename: string, mimeType: string): void {
  const blob =
    typeof bytes === 'string'
      ? new Blob([bytes], { type: mimeType })
      : new Blob([bytes.buffer as ArrayBuffer], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
