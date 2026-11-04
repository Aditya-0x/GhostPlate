"""
Interactive FastAPI Web Dashboard for Adverse Weather ALPR
Group 6: Aditya Verma, Varun Singh Rajpurohit, Appan
Run with: uvicorn app:app --port 8000
"""

import os
import io
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

from src.pipeline import AdverseWeatherALPRPipeline

app = FastAPI(title="Adverse Weather ALPR System - Group 6")

# Global pipeline instance
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        import torch
        weights = "models/char_cnn_lightweight.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        PIPELINE = AdverseWeatherALPRPipeline(
            weights_path=weights if os.path.exists(weights) else None,
            device=device
        )
    return PIPELINE


def cv2_to_base64(img: np.ndarray) -> str:
    """Encodes BGR cv2 image to base64 JPEG."""
    if img is None:
        return ""
    if img.ndim == 2:
        # Grayscale or float map
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        _, buffer = cv2.imencode(".jpg", img)
    else:
        _, buffer = cv2.imencode(".jpg", img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adverse Weather ALPR - Group 6</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-hover: #334155;
            --primary: #38bdf8;
            --accent: #818cf8;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #34d399;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
        .container { max-width: 1300px; margin: 0 auto; }
        header { margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
        h1 { font-size: 1.6rem; color: var(--primary); font-weight: 700; }
        .authors { font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }
        .grid { display: grid; grid-template-columns: 340px 1fr; gap: 24px; }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
        .card h2 { font-size: 1.1rem; margin-bottom: 16px; color: var(--accent); }
        .control-group { margin-bottom: 16px; }
        label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; color: var(--text-muted); }
        select, input[type="file"], button { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 0.9rem; }
        button { background: linear-gradient(135deg, #0284c7, #2563eb); border: none; font-weight: 600; cursor: pointer; margin-top: 10px; transition: opacity 0.2s; }
        button:hover { opacity: 0.9; }
        .checkbox-group { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 0.88rem; }
        .checkbox-group input { width: 18px; height: 18px; accent-color: var(--primary); }
        .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 16px; }
        .img-box { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }
        .img-box-header { padding: 8px 12px; font-size: 0.8rem; font-weight: 600; background: #131d31; color: var(--primary); border-bottom: 1px solid var(--border); }
        .img-box img { width: 100%; height: 210px; object-fit: contain; background: #000; }
        .ocr-banner { background: #0c4a6e; border: 1px solid #0284c7; padding: 16px; border-radius: 8px; margin-top: 16px; display: flex; justify-content: space-between; align-items: center; }
        .ocr-plate { font-size: 1.8rem; font-weight: 700; letter-spacing: 4px; color: var(--success); font-family: monospace; }
        .chars-list { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
        .char-pill { background: var(--bg); border: 1px solid var(--border); padding: 4px 8px; border-radius: 6px; font-family: monospace; font-size: 0.85rem; text-align: center; }
        .char-pill small { display: block; color: var(--text-muted); font-size: 0.65rem; }
        .badge { background: #0369a1; padding: 3px 8px; border-radius: 999px; font-size: 0.75rem; color: #e0f2fe; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Automated License Plate Character Recognition</h1>
                <div class="authors">Group 6: Aditya Verma (12408804) &bull; Varun Singh Rajpurohit (12409154) &bull; Appan (12409471)</div>
            </div>
            <span class="badge">OpenCV + PyTorch + Dark Channel Prior</span>
        </header>

        <div class="grid">
            <div class="card">
                <h2>Control Panel</h2>
                <div class="control-group">
                    <label>Benchmark Sample</label>
                    <select id="sampleSelect" onchange="loadPresetSample()">
                        <optgroup label="Real Kaggle Adverse Weather Dataset (Downloaded)">
                            <option value="data/kaggle_plates/license_plate_restoration/Foggy Images/20190615_155317_resized.jpg">Kaggle Real Foggy 01 (Dense Fog)</option>
                            <option value="data/kaggle_plates/license_plate_restoration/Foggy Images/20190712_121651_resized.jpg">Kaggle Real Foggy 02 (Atmospheric Haze)</option>
                            <option value="data/kaggle_plates/license_plate_restoration/Night Images/20190615_155317_resized.jpg">Kaggle Real Night 01 (Headlight Glare & Bloom)</option>
                            <option value="data/kaggle_plates/license_plate_restoration/Night Images/20190712_121651_resized.jpg">Kaggle Real Night 02 (Low Light + Flare)</option>
                            <option value="data/kaggle_plates/license_plate_restoration/Motion Blur Images/20190615_155317_resized.jpg">Kaggle Real Weather Blur 01</option>
                        </optgroup>
                        <optgroup label="Synthetic Adverse Weather Benchmarks">
                            <option value="data/samples/sample_01_combined.jpg">Synthetic 01: Combined (Fog + Rain + Glare)</option>
                            <option value="data/samples/sample_01_fog.jpg">Synthetic 01: Dense Fog (Koschmieder)</option>
                            <option value="data/samples/sample_01_rain.jpg">Synthetic 01: Heavy Rain Streaks</option>
                            <option value="data/samples/sample_01_glare.jpg">Synthetic 01: Headlight Glare Flare</option>
                            <option value="data/samples/sample_01_clean.jpg">Synthetic 01: Clean Baseline</option>
                        </optgroup>
                    </select>
                </div>

                <div class="control-group">
                    <label>Or Upload Custom Vehicle Image</label>
                    <input type="file" id="fileInput" accept="image/*">
                </div>

                <hr style="border: 0; border-top: 1px solid var(--border); margin: 16px 0;">

                <div class="control-group">
                    <label>Restoration Pre-Filters</label>
                    <div class="checkbox-group">
                        <input type="checkbox" id="enableDehaze" checked>
                        <span>Dark Channel Prior (DCP) De-hazing</span>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="enableGlare" checked>
                        <span>Headlight Glare Attenuation</span>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="enableDerain" checked>
                        <span>Rain Streak Filtering</span>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="enableClahe" checked>
                        <span>Adaptive Contrast (CLAHE)</span>
                    </div>
                </div>

                <button onclick="runPipeline()">Process & Recognize Plate</button>
            </div>

            <div class="card">
                <h2>Restoration & ALPR Pipeline View</h2>
                
                <div class="ocr-banner" id="ocrBanner" style="display: none;">
                    <div>
                        <div style="font-size: 0.85rem; color: #bae6fd; text-transform: uppercase; font-weight: 600;">Recognized Plate Number</div>
                        <div class="ocr-plate" id="plateText">--------</div>
                    </div>
                    <div id="plateBBoxText" style="font-size: 0.85rem; color: var(--text-muted); text-align: right;"></div>
                </div>

                <div class="chars-list" id="charsList"></div>

                <div class="results-grid">
                    <div class="img-box">
                        <div class="img-box-header">1. Raw Adverse Weather Input</div>
                        <img id="imgInput" src="" alt="Awaiting input...">
                    </div>
                    <div class="img-box">
                        <div class="img-box-header">2. Dark Channel Prior J_dark</div>
                        <img id="imgDarkCh" src="" alt="Dark Channel Prior">
                    </div>
                    <div class="img-box">
                        <div class="img-box-header">3. Guided Transmission Map t(x)</div>
                        <img id="imgTransmission" src="" alt="Transmission Map">
                    </div>
                    <div class="img-box">
                        <div class="img-box-header">4. Restored Scene Radiance</div>
                        <img id="imgEnhanced" src="" alt="Dehazed / Restored">
                    </div>
                    <div class="img-box">
                        <div class="img-box-header">5. Plate Localization & Crop</div>
                        <img id="imgCrop" src="" alt="Cropped Plate">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runPipeline() {
            const formData = new FormData();
            const fileInput = document.getElementById('fileInput');
            const sampleSelect = document.getElementById('sampleSelect');

            if (fileInput.files.length > 0) {
                formData.append('file', fileInput.files[0]);
            } else {
                formData.append('sample_path', sampleSelect.value);
            }

            formData.append('enable_dehaze', document.getElementById('enableDehaze').checked);
            formData.append('enable_glare', document.getElementById('enableGlare').checked);
            formData.append('enable_derain', document.getElementById('enableDerain').checked);
            formData.append('enable_clahe', document.getElementById('enableClahe').checked);

            document.getElementById('plateText').innerText = "Processing...";
            document.getElementById('ocrBanner').style.display = "flex";

            try {
                const res = await fetch('/api/process', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                document.getElementById('plateText').innerText = data.plate_text || "NO TEXT DETECTED";
                document.getElementById('plateBBoxText').innerText = `Bounding Box: ${JSON.stringify(data.bbox)}`;

                document.getElementById('imgInput').src = data.img_input;
                document.getElementById('imgDarkCh').src = data.img_dark_ch;
                document.getElementById('imgTransmission').src = data.img_transmission;
                document.getElementById('imgEnhanced').src = data.img_enhanced;
                document.getElementById('imgCrop').src = data.img_crop;

                const charsDiv = document.getElementById('charsList');
                charsDiv.innerHTML = "";
                (data.char_preds || []).forEach(([ch, conf]) => {
                    const pill = document.createElement('div');
                    pill.className = 'char-pill';
                    pill.innerHTML = `<strong>${ch}</strong><small>${(conf * 100).toFixed(1)}%</small>`;
                    charsDiv.appendChild(pill);
                });
            } catch (err) {
                console.error(err);
                alert("Failed to process image: " + err);
            }
        }

        function loadPresetSample() {
            runPipeline();
        }

        window.onload = () => {
            runPipeline();
        };
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


@app.post("/api/process")
async def process_image(
    file: UploadFile | None = File(None),
    sample_path: str = Form("data/samples/sample_01_combined.jpg"),
    enable_dehaze: bool = Form(True),
    enable_glare: bool = Form(True),
    enable_derain: bool = Form(True),
    enable_clahe: bool = Form(True)
):
    try:
        pipeline = get_pipeline()
        pipeline.enhancer.enable_dehaze = enable_dehaze
        pipeline.enhancer.enable_glare_suppression = enable_glare
        pipeline.enhancer.enable_derain = enable_derain
        pipeline.enhancer.enable_clahe = enable_clahe

        if file is not None and file.filename:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            if not os.path.exists(sample_path):
                from generate_sample_data import generate_benchmark_dataset
                generate_benchmark_dataset()
            img = cv2.imread(sample_path)

        if img is None:
            return JSONResponse({"error": "Failed to load image"}, status_code=400)

        # Scale down ultra-high-resolution images (e.g. 4032x3024) for real-time responsiveness
        max_dim = 1280
        ih, iw = img.shape[:2]
        if max(ih, iw) > max_dim:
            scale = max_dim / float(max(ih, iw))
            img = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)

        res = pipeline.process(img, with_enhancement=True)
        enh = res["enhancement_data"] or {}

        # Draw bbox on enhanced image for display
        disp_enhanced = res["work_image"].copy()
        if res["plate_bbox"]:
            x, y, w, h = res["plate_bbox"]
            cv2.rectangle(disp_enhanced, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return JSONResponse({
            "plate_text": res["plate_text"],
            "char_preds": res["char_preds"],
            "bbox": res["plate_bbox"],
            "img_input": cv2_to_base64(img),
            "img_enhanced": cv2_to_base64(disp_enhanced),
            "img_crop": cv2_to_base64(res["plate_crop"]),
            "img_dark_ch": cv2_to_base64(enh.get("dark_channel")),
            "img_transmission": cv2_to_base64(enh.get("transmission_map"))
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

