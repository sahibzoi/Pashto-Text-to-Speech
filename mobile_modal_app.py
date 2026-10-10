"""Mobile-friendly Pashto VITS2 demo on Modal.
Run from the root of the forked repository, AFTER downloading model/G_314000.pth:
    modal deploy mobile_modal_app.py

Produces ONE same-origin web app with GET / and POST /api/synthesize.
Uses the existing VITS2 project and does not retrain it.
"""
from pathlib import Path
import modal

APP_NAME = "pashto-voice-mobile"
ROOT = Path(__file__).resolve().parent
REMOTE = "/workspace"

image = (modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng")
    .pip_install("torch==2.1.2", "torchaudio==2.1.2", "numpy==1.24.4", "scipy==1.11.4", "librosa==0.10.1", "numba==0.58.1", "Unidecode==1.3.7", "phonemizer==3.2.1", "tensorboardX", "tqdm", "Cython", "fastapi==0.115.6", "pydantic==2.9.2")
    .add_local_dir(str(ROOT), REMOTE, ignore=[".git", ".venv", "__pycache__", "*.pyc", "samples", "filelists", "scripts", "*.zip"]))

app = modal.App(APP_NAME)

@app.cls(image=image, gpu="T4", scaledown_window=120, max_containers=1)
class SpeechEngine:
    @modal.enter()
    def init(self):
        import sys, json, torch
        sys.path.insert(0, REMOTE)
        from text import text_to_sequence
        from text.symbols import symbols
        from models import SynthesizerTrn
        import commons
        with open(f"{REMOTE}/configs/pashto.json", encoding="utf-8") as f:
            self.hps = json.load(f)
        model_path = f"{REMOTE}/model/G_314000.pth"
        if not Path(model_path).is_file():
            raise FileNotFoundError("Missing model/G_314000.pth. Download the trained checkpoint before deploying.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = SynthesizerTrn(len(symbols), 80,
            self.hps["train"]["segment_size"] // self.hps["data"]["hop_length"],
            n_speakers=self.hps["data"]["n_speakers"], **self.hps["model"]).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["model"])
        self.net.eval()
        self.text_to_sequence, self.commons, self.torch = text_to_sequence, commons, torch

    @modal.method()
    def generate(self, text: str, noise_scale: float, noise_scale_w: float, length_scale: float) -> bytes:
        import io, numpy as np
        from scipy.io.wavfile import write
        if not text.strip():
            raise ValueError("Empty text")
        seq = self.text_to_sequence(text.strip(), ["pashto_cleaners"])
        if self.hps["data"].get("add_blank"):
            seq = self.commons.intersperse(seq, 0)
        if not seq:
            raise ValueError("No supported Pashto characters found")
        torch = self.torch
        x = torch.LongTensor(seq).unsqueeze(0).to(self.device)
        x_len = torch.LongTensor([len(seq)]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        with torch.no_grad():
            audio = self.net.infer(x, x_len, sid=sid, noise_scale=noise_scale,
                                   noise_scale_w=noise_scale_w, length_scale=length_scale)
        samples = np.clip(audio[0][0].detach().cpu().float().numpy().squeeze(), -1, 1)
        buf = io.BytesIO()
        write(buf, 22050, (samples * 32767).astype(np.int16))
        return buf.getvalue()

@app.function(image=modal.Image.debian_slim(python_version="3.10").pip_install("fastapi==0.115.6"))
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response
    from pydantic import BaseModel, Field

    api = FastAPI(title="Pashto Voice Mobile Demo")
    class SpeechRequest(BaseModel):
        text: str = Field(min_length=1, max_length=400)
        noise_scale: float = Field(default=0.4, ge=0.1, le=1.0)
        noise_scale_w: float = Field(default=0.8, ge=0.1, le=1.0)
        length_scale: float = Field(default=1.0, ge=0.6, le=1.5)

    # Inline HTML ensures browser API calls stay on the same origin (avoids CORS).
    @api.get("/", response_class=HTMLResponse)
    def home():
        return HTML_PAGE

    @api.post("/api/synthesize")
    async def synthesize(request: SpeechRequest):
        try:
            audio = await SpeechEngine().generate.remote.aio(request.text,
                request.noise_scale, request.noise_scale_w, request.length_scale)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Voice generation failed: {type(exc).__name__}") from exc
        return Response(audio, media_type="audio/wav",
                        headers={"Content-Disposition": 'attachment; filename="pashto-voice.wav"'})
    return api

HTML_PAGE = r'''<!doctype html><html lang="ps" dir="rtl"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pashto AI Voice</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#1b2635;font-family:system-ui,"Noto Naskh Arabic",sans-serif}main{max-width:580px;margin:30px auto;padding:18px}article{background:white;border:1px solid #e4e9f0;border-radius:20px;padding:23px;box-shadow:0 8px 30px #0b23330d}h1{font-size:24px;margin:0 0 8px}p{color:#526175;line-height:1.9}textarea{width:100%;min-height:150px;border:1px solid #cdd7e2;border-radius:12px;padding:14px;font-size:18px;line-height:2;resize:vertical}label{display:block;margin-top:14px;font-size:14px;font-weight:600}input[type=range]{width:100%;accent-color:#183a67}button{background:#173a67;color:white;border:0;border-radius:11px;padding:13px 20px;font-size:16px;width:100%;margin-top:20px;cursor:pointer}button:disabled{opacity:.55}audio{display:block;width:100%;margin-top:15px}.link{display:block;text-align:center;padding:12px;color:#173a67}small{display:block;margin-top:12px;color:#64748b;line-height:1.7}</style></head>
<body><main><article><h1>🎙️ پښتو AI غږ</h1><p>پښتو متن ولیکئ او د منګلو لهجې غږ واورئ.</p>
<label for="t">پښتو متن</label><textarea id="t" maxlength="400">سلامونه دوستانو! د النقیب موبایل زون ته ښه راغلاست.</textarea>
<label>د غږ سرعت: <output id="speedValue">1.00</output></label><input id="speed" type="range" min="0.6" max="1.5" value="1" step="0.05">
<label>د غږ تنوع: <output id="noiseValue">0.40</output></label><input id="noise" type="range" min="0.1" max="1" value="0.4" step="0.05">
<label>د وقفو تنوع: <output id="pauseValue">0.80</output></label><input id="pause" type="range" min="0.1" max="1" value="0.8" step="0.05">
<button id="go">غږ تولید کړه</button><p id="status" role="status" aria-live="polite"></p><audio id="player" controls hidden></audio><a id="download" class="link" download="pashto-voice.wav" hidden>⬇️ غږ ډاونلوډ کړه</a>
<small>دا د یوه ویاند د VITS2 ماډل امتحاني نسخه ده. د تولید لپاره GPU او فعال Modal deployment ضروري دي. لومړی ځل ممکن لږ وخت ونیسي.</small>
</article></main><script>
let activeUrl=null;const $=id=>document.getElementById(id);for(const [id,out] of [['speed','speedValue'],['noise','noiseValue'],['pause','pauseValue']])$(id).oninput=()=>$(out).textContent=Number($(id).value).toFixed(2);
$('go').onclick=async()=>{const text=$('t').value.trim();if(!text){$('status').textContent='مهرباني وکړئ متن ولیکئ.';return} $('go').disabled=true;$('status').textContent='غږ جوړېږي، لږ انتظار وکړئ...';$('player').hidden=true;$('download').hidden=true;
try{const res=await fetch('/api/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,noise_scale:Number($('noise').value),noise_scale_w:Number($('pause').value),length_scale:Number($('speed').value)})});if(!res.ok){throw new Error('د سرور ستونزه: '+res.status)}const blob=await res.blob();if(activeUrl)URL.revokeObjectURL(activeUrl);activeUrl=URL.createObjectURL(blob);$('player').src=activeUrl;$('download').href=activeUrl;$('player').hidden=false;$('download').hidden=false;$('status').textContent='غږ جوړ شو!';}catch(err){$('status').textContent='غږ جوړ نه شو. د سرور او ماډل تنظیمات باید وکتل شي. ('+err.message+')'}finally{$('go').disabled=false}};
</script></body></html>'''
