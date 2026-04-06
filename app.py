"""
app.py  —  DeepFace Detection  |  Streamlit UI  v4.2
Connects to elite_predictor.py (EliteDetector v6.1 / fixed)
+ Video analysis via predict_video.py (v1.1 / fixed)

Changes from v4.1:
  [FIX A] overall_banner: handles NO_FACE_DETECTED label (v6.1 returns this
          string, not "NO_FACE" — was showing wrong CSS class and icon).
  [FIX B] NO_FACE block: checks for both "NO_FACE" and "NO_FACE_DETECTED"
          so the UI never silently falls through to the per-face loop.
  [FIX C] render_verdict: added NOT_A_PHOTO label mapping so sculptures /
          mannequins render a proper card rather than the default "?" fallback.
  [FIX D] render_forensic_grid: guards against missing forensic keys with
          .get(key, 0) so partial results don't crash the expander.
  [FIX E] Video analysis block: reads result["confidence"] directly
          (predict_video v1.1 now guarantees this key).

Run:  streamlit run app.py
"""

import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from elite_predictor import load_model, predict_image
from predict_video import predict_video

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeepFace Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS  — high-contrast dark terminal / forensic-lab aesthetic
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Barlow+Condensed:wght@300;500;700;900&display=swap');

:root {
    --bg:        #060810;
    --surface:   #0d1117;
    --surface2:  #131921;
    --surface3:  #1a2332;
    --border:    #1e2d42;
    --border2:   #243548;
    --accent:    #00d4ff;
    --accent-dim:#005f72;
    --real:      #00e896;
    --real-dim:  #004d33;
    --fake:      #ff2d55;
    --fake-dim:  #4d0015;
    --warn:      #ffbe00;
    --warn-dim:  #4d3800;
    --noface:    #8b5cf6;
    --noface-dim:#2d1f52;
    --text:      #cdd9e5;
    --muted:     #546e7a;
    --muted2:    #37474f;
    --mono:      'IBM Plex Mono', monospace;
    --display:   'Barlow Condensed', sans-serif;
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text);
    font-family: var(--mono);
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { font-family: var(--mono) !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1400px !important; }

.hero {
    position: relative;
    padding: 3rem 2rem 2.5rem;
    background: linear-gradient(180deg, #0a1628 0%, #060810 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 0 0 16px 16px;
    margin-bottom: 2rem;
    overflow: hidden;
    text-align: center;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 50% at 50% -10%, rgba(0,212,255,.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: var(--mono); font-size: .7rem; letter-spacing: 4px;
    color: var(--accent); text-transform: uppercase; margin-bottom: .8rem;
}
.hero-title {
    font-family: var(--display); font-size: clamp(3rem, 7vw, 6rem);
    font-weight: 900; letter-spacing: -1px; line-height: .95;
    text-transform: uppercase; color: #fff; margin-bottom: .6rem;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent) 0%, #7ee8ff 50%, var(--real) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-family: var(--mono); font-size: .82rem; color: var(--muted);
    letter-spacing: 1px; margin-bottom: 1.5rem;
}
.hero-tags { display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; }
.htag {
    font-family: var(--mono); font-size: .68rem; letter-spacing: 1px;
    padding: .22rem .7rem; border-radius: 3px; border: 1px solid; text-transform: uppercase;
}
.htag-cyan  { color: var(--accent); border-color: var(--accent-dim); background: rgba(0,212,255,.06); }
.htag-green { color: var(--real);   border-color: var(--real-dim);   background: rgba(0,232,150,.05); }
.htag-pink  { color: #ff79c6;       border-color: #4d2060;           background: rgba(255,121,198,.05); }

.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.4rem; margin-bottom: 1rem;
}
.card-title {
    font-family: var(--mono); font-size: .72rem; letter-spacing: 3px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 1rem;
    padding-bottom: .6rem; border-bottom: 1px solid var(--border);
}

.verdict {
    border-radius: 8px; padding: 1.4rem 1rem; text-align: center;
    margin-bottom: .9rem; border: 1px solid; position: relative; overflow: hidden;
}
.verdict::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,.03) 0%, transparent 100%);
    pointer-events: none;
}
.v-real      { background: var(--real-dim);   border-color: var(--real);   }
.v-fake      { background: var(--fake-dim);   border-color: var(--fake);   }
.v-unc       { background: var(--warn-dim);   border-color: var(--warn);   }
.v-noface    { background: var(--noface-dim); border-color: var(--noface); }
.v-notaphoto { background: var(--noface-dim); border-color: var(--noface); }
.verdict-icon  { font-size: 2.4rem; line-height: 1; margin-bottom: .3rem; }
.verdict-label {
    font-family: var(--display); font-size: 2.2rem; font-weight: 900;
    letter-spacing: 3px; text-transform: uppercase; line-height: 1;
}
.v-real      .verdict-label { color: var(--real);   }
.v-fake      .verdict-label { color: var(--fake);   }
.v-unc       .verdict-label { color: var(--warn);   }
.v-noface    .verdict-label { color: var(--noface); }
.v-notaphoto .verdict-label { color: var(--noface); }
.verdict-sub { font-size: .8rem; color: var(--muted); margin-top: .4rem; font-family: var(--mono); }

.pbar-wrap  { margin: 1rem 0; }
.pbar-meta  { display: flex; justify-content: space-between; font-family: var(--mono); font-size: .72rem; color: var(--muted); margin-bottom: .4rem; }
.pbar-track { height: 8px; background: var(--surface2); border-radius: 99px; overflow: hidden; border: 1px solid var(--border); }
.pbar-fill  { height: 100%; border-radius: 99px; transition: width 1.4s cubic-bezier(.16,1,.3,1); }

.overall {
    border-radius: 8px; padding: .9rem 1.2rem;
    display: flex; align-items: center; gap: .9rem;
    margin-bottom: 1.2rem; border: 1px solid;
}
.overall-real    { background: rgba(0,232,150,.06);  border-color: var(--real);   }
.overall-fake    { background: rgba(255,45,85,.06);  border-color: var(--fake);   }
.overall-unc     { background: rgba(255,190,0,.06);  border-color: var(--warn);   }
.overall-noface  { background: rgba(139,92,246,.06); border-color: var(--noface); }
.overall-icon    { font-size: 1.8rem; line-height: 1; }
.overall-lbl     { font-family: var(--display); font-size: 1.1rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.overall-desc    { font-size: .73rem; color: var(--muted); margin-top: .1rem; font-family: var(--mono); }
.overall-real    .overall-lbl { color: var(--real);   }
.overall-fake    .overall-lbl { color: var(--fake);   }
.overall-unc     .overall-lbl { color: var(--warn);   }
.overall-noface  .overall-lbl { color: var(--noface); }

.face-sep {
    font-family: var(--mono); font-size: .68rem; letter-spacing: 3px;
    text-transform: uppercase; color: var(--muted);
    padding: .4rem 0 .4rem .8rem; border-left: 2px solid var(--accent);
    margin: 1.4rem 0 .9rem;
}

.fgrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .45rem; margin-top: .7rem; }
.fcell {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 6px; padding: .6rem .5rem; text-align: center;
}
.fcell-lbl { font-size: .62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-family: var(--mono); }
.fcell-val { font-size: 1rem; font-weight: 700; color: var(--accent); margin-top: .15rem; font-family: var(--mono); }

.sbadge {
    display: inline-flex; align-items: center; gap: .4rem;
    font-family: var(--mono); font-size: .72rem; letter-spacing: 1px;
    padding: .35rem .75rem; border-radius: 4px; border: 1px solid; text-transform: uppercase;
}
.sbadge-ok  { color: var(--real);  border-color: var(--real-dim);  background: rgba(0,232,150,.08); }
.sbadge-err { color: var(--fake);  border-color: var(--fake-dim);  background: rgba(255,45,85,.08); }

.doc-pill {
    display: inline-block; background: rgba(0,212,255,.07);
    border: 1px solid var(--accent-dim); color: var(--accent);
    font-family: var(--mono); font-size: .68rem;
    padding: .2rem .6rem; border-radius: 4px;
    margin-bottom: .8rem; letter-spacing: 1px; text-transform: uppercase;
}

.video-verdict {
    border-radius: 10px; padding: 1.6rem 1.4rem;
    display: grid; grid-template-columns: auto 1fr auto;
    align-items: center; gap: 1.2rem;
    margin-bottom: 1.4rem; border: 1px solid; position: relative; overflow: hidden;
}
.video-verdict::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,.03) 0%, transparent 60%);
    pointer-events: none;
}
.vv-real { background: rgba(0,232,150,.06); border-color: var(--real); }
.vv-fake { background: rgba(255,45,85,.06); border-color: var(--fake); }
.vv-unc  { background: rgba(255,190,0,.06); border-color: var(--warn); }
.vv-icon { font-size: 3rem; line-height: 1; }
.vv-main { }
.vv-label {
    font-family: var(--display); font-size: 2rem; font-weight: 900;
    letter-spacing: 3px; text-transform: uppercase; line-height: 1; margin-bottom: .3rem;
}
.vv-real .vv-label { color: var(--real); }
.vv-fake .vv-label { color: var(--fake); }
.vv-unc  .vv-label { color: var(--warn); }
.vv-sub  { font-family: var(--mono); font-size: .75rem; color: var(--muted); }
.vv-conf {
    font-family: var(--display); font-size: 2.4rem; font-weight: 900;
    text-align: right; line-height: 1;
}
.vv-real .vv-conf { color: var(--real); }
.vv-fake .vv-conf { color: var(--fake); }
.vv-unc  .vv-conf { color: var(--warn); }
.vv-conf-lbl { font-family: var(--mono); font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; text-align: right; }

.timeline { margin: 1rem 0; }
.tl-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .35rem; font-family: var(--mono); font-size: .72rem; }
.tl-ts  { color: var(--muted); width: 4rem; flex-shrink: 0; }
.tl-bar-wrap { flex: 1; height: 6px; background: var(--surface2); border-radius: 99px; overflow: hidden; }
.tl-bar { height: 100%; border-radius: 99px; }
.tl-lbl { width: 4rem; flex-shrink: 0; text-align: right; }
.tl-real { background: var(--real); color: var(--real); }
.tl-fake { background: var(--fake); color: var(--fake); }
.tl-unc  { background: var(--warn); color: var(--warn); }

div[data-testid="metric-container"] {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; padding: .7rem .9rem !important;
}
div[data-testid="metric-container"] label {
    color: var(--muted) !important; font-family: var(--mono) !important;
    font-size: .65rem !important; letter-spacing: 1px !important; text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important; font-family: var(--mono) !important; font-size: 1.3rem !important;
}
div[data-testid="stFileUploader"] {
    background: var(--surface) !important; border: 1.5px dashed var(--border2) !important; border-radius: 10px !important;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
details { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
summary { color: var(--muted) !important; font-family: var(--mono) !important; font-size: .75rem !important; }

.sdiv { height: 1px; background: var(--border); margin: 1.4rem 0; }

.feat-item { display: flex; align-items: flex-start; gap: .6rem; font-size: .8rem; color: var(--muted); padding: .3rem 0; border-bottom: 1px solid var(--border); }
.feat-item:last-child { border-bottom: none; }
.feat-dot { color: var(--accent); font-size: .55rem; margin-top: .35rem; flex-shrink: 0; }

.footer {
    text-align: center; margin-top: 3rem; padding: 1.2rem;
    color: var(--muted2); font-family: var(--mono); font-size: .68rem;
    letter-spacing: 1px; border-top: 1px solid var(--border);
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp .5s ease-out both; }

@keyframes glowPulse {
    0%,100% { box-shadow: 0 0 0 rgba(0,232,150,0); }
    50%      { box-shadow: 0 0 24px rgba(0,232,150,.15); }
}
.v-real { animation: glowPulse 3s infinite; }

@keyframes flickerFake {
    0%,96%,100% { opacity:1; } 97% { opacity:.85; }
}
.v-fake { animation: flickerFake 4s infinite; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
MODEL_PATH  = "elite_resnet_detector.pth"
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]

# All possible NO_FACE label strings returned by v6.1
_NO_FACE_LABELS = {"NO_FACE", "NO_FACE_DETECTED"}


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


model = load_model_cached()


# ═══════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _bar_color(prob: float) -> str:
    if prob < 0.35:
        return "linear-gradient(90deg,#00e896,#00b870)"
    if prob < 0.65:
        return "linear-gradient(90deg,#ffbe00,#e6a800)"
    return "linear-gradient(90deg,#ff2d55,#cc1a3a)"


def render_prob_bar(prob: float, label: str = "FAKE PROBABILITY"):
    pct   = round(prob * 100, 1)
    color = _bar_color(prob)
    st.markdown(f"""
    <div class="pbar-wrap">
      <div class="pbar-meta"><span>{label}</span><span>{pct}%</span></div>
      <div class="pbar-track">
        <div class="pbar-fill" style="width:{pct}%;background:{color};"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_verdict(label: str, prob: float):
    # FIX C: added NOT_A_PHOTO entry
    cfg = {
        "REAL":             ("v-real",      "✅", "No manipulation detected"),
        "FAKE":             ("v-fake",      "⚠️", "AI-generated or manipulated"),
        "UNCERTAIN":        ("v-unc",       "🤔", "Inconclusive — review manually"),
        "NO_FACE":          ("v-noface",    "🚫", "No human face found in image"),
        "NO_FACE_DETECTED": ("v-noface",    "🚫", "No human face found in image"),
        "NOT_A_PHOTO":      ("v-notaphoto", "🗿", "Sculpture / mask / non-photo"),
    }
    cls, icon, sub = cfg.get(label, ("v-unc", "?", ""))
    st.markdown(f"""
    <div class="verdict {cls} fade-up">
      <div class="verdict-icon">{icon}</div>
      <div class="verdict-label">{label}</div>
      <div class="verdict-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)
    if label not in ("NO_FACE", "NO_FACE_DETECTED", "NOT_A_PHOTO"):
        render_prob_bar(prob)


def render_forensic_grid(f: dict):
    # FIX D: .get(key, 0) guards against missing keys in partial results
    cells = [
        ("Artifact",  f.get("artifact_score",  0)),
        ("Texture",   f.get("texture_score",   0)),
        ("Symmetry",  f.get("symmetry_score",  0)),
        ("Eye Ref",   f.get("eye_ref_score",   0)),
        ("Lighting",  f.get("lighting_score",  0)),
        ("FFT Ring",  f.get("fft_ring_score",  0)),
    ]
    inner = "".join(
        f'<div class="fcell">'
        f'<div class="fcell-lbl">{n}</div>'
        f'<div class="fcell-val">{v:.3f}</div>'
        f'</div>'
        for n, v in cells
    )
    fc = f.get("forensic_composite")
    if fc is not None:
        inner += (
            f'<div class="fcell" style="grid-column:span 3;background:var(--surface3);">'
            f'<div class="fcell-lbl">Forensic Composite</div>'
            f'<div class="fcell-val">{fc:.3f}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="fgrid">{inner}</div>', unsafe_allow_html=True)


def overall_banner(result: dict):
    lbl  = result["overall"]
    prob = result.get("avg_prob")

    # FIX A: handles both "NO_FACE" and "NO_FACE_DETECTED" with noface styling
    cfg = {
        "REAL":             ("overall-real",   "✅", "All face(s) assessed as authentic."),
        "FAKE":             ("overall-fake",   "⚠️", "At least one face flagged as AI-generated."),
        "UNCERTAIN":        ("overall-unc",    "🤔", "Inconclusive — manual review recommended."),
        "NO_FACE":          ("overall-noface", "🚫", "No human face detected in this image."),
        "NO_FACE_DETECTED": ("overall-noface", "🚫", "No human face detected in this image."),
        "NOT_A_PHOTO":      ("overall-noface", "🗿", "Image does not contain a real human face."),
    }
    cls, icon, desc = cfg.get(lbl, ("overall-unc", "?", ""))
    prob_txt  = f"&nbsp;·&nbsp; avg fake prob: {prob:.2%}" if prob is not None else ""
    faces_txt = f"&nbsp;·&nbsp; {result['total_faces']} face(s)" if result.get("total_faces") else ""
    st.markdown(f"""
    <div class="overall {cls} fade-up">
      <div class="overall-icon">{icon}</div>
      <div class="overall-text">
        <div class="overall-lbl">{lbl}</div>
        <div class="overall-desc">{desc}{faces_txt}{prob_txt}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_video_verdict(verdict: str, confidence: float, aggregate: dict, temporal: dict, meta: dict):
    cfg = {
        "REAL":      ("vv-real", "✅", "Video appears authentic"),
        "FAKE":      ("vv-fake", "⚠️", "Deepfake content detected"),
        "UNCERTAIN": ("vv-unc",  "🤔", "Inconclusive — manual review recommended"),
    }
    cls, icon, sub = cfg.get(verdict, ("vv-unc", "?", ""))

    st.markdown(f"""
    <div class="video-verdict {cls} fade-up">
      <div class="vv-icon">{icon}</div>
      <div class="vv-main">
        <div class="vv-label">{verdict}</div>
        <div class="vv-sub">{sub}</div>
      </div>
      <div>
        <div class="vv-conf">{confidence:.0f}%</div>
        <div class="vv-conf-lbl">confidence</div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duration",    f"{meta.get('duration_s', 0):.1f}s")
    c2.metric("Frames",      meta.get("frames_analysed", 0))
    c3.metric("Resolution",  f"{meta.get('width', 0)}×{meta.get('height', 0)}")
    c4.metric("Transitions", temporal.get("transition_count", 0))
    c5.metric("Consistent",  f"{temporal.get('consistent_pct', 0)}%")

    counts = aggregate.get("frame_counts", {})
    total  = max(sum(counts.values()), 1)
    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:var(--mono);font-size:.7rem;letter-spacing:2px;"
        "text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;'>"
        "Frame Breakdown</div>",
        unsafe_allow_html=True,
    )
    for lbl, sym, color_cls in [("REAL","✅","tl-real"),("FAKE","⚠️","tl-fake"),("UNCERTAIN","❓","tl-unc")]:
        n   = counts.get(lbl, 0)
        pct = round(n / total * 100, 1)
        st.markdown(f"""
        <div class="tl-row">
          <div class="tl-ts">{sym} {lbl}</div>
          <div class="tl-bar-wrap">
            <div class="tl-bar {color_cls}" style="width:{pct}%;"></div>
          </div>
          <div class="tl-lbl" style="color:var(--muted);">{n} frames</div>
        </div>""", unsafe_allow_html=True)


def render_frame_timeline(frame_results: list):
    if not frame_results:
        return
    st.markdown(
        "<div style='font-family:var(--mono);font-size:.7rem;letter-spacing:2px;"
        "text-transform:uppercase;color:var(--muted);margin:1.2rem 0 .6rem;'>"
        "Per-Frame Timeline</div>",
        unsafe_allow_html=True,
    )
    display   = frame_results[:30]
    color_map = {"REAL": "tl-real", "FAKE": "tl-fake", "UNCERTAIN": "tl-unc"}
    sym_map   = {"REAL": "✅", "FAKE": "⚠️", "UNCERTAIN": "❓"}
    for r in display:
        ts    = r.get("timestamp_s", 0)
        lbl   = r.get("overall", "UNCERTAIN")
        prob  = r.get("avg_prob", 0.5)
        cls   = color_map.get(lbl, "tl-unc")
        sym   = sym_map.get(lbl, "?")
        pct   = round(prob * 100, 1)
        st.markdown(f"""
        <div class="tl-row">
          <div class="tl-ts" style="width:5rem;">{sym} {ts:.1f}s</div>
          <div class="tl-bar-wrap">
            <div class="tl-bar {cls}" style="width:{pct}%;"></div>
          </div>
          <div class="tl-lbl" style="color:var(--muted);">{pct}%</div>
        </div>""", unsafe_allow_html=True)
    if len(frame_results) > 30:
        st.caption(f"… and {len(frame_results) - 30} more frames (see JSON report for full data)")


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;
                font-weight:900;letter-spacing:2px;text-transform:uppercase;
                color:#cdd9e5;margin-bottom:.2rem;">
        DEEPFACE<br><span style="color:#00d4ff;">DETECTION</span>
    </div>
    <div style="font-size:.65rem;color:#546e7a;letter-spacing:2px;
                text-transform:uppercase;margin-bottom:1rem;">
        EliteDetector v6.1
    </div>
    """, unsafe_allow_html=True)

    if model is not None:
        st.markdown('<div class="sbadge sbadge-ok">◉ &nbsp;Model active</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="sbadge sbadge-err">✕ &nbsp;Model not loaded</div>',
                    unsafe_allow_html=True)
        st.caption(f"Expected: `{MODEL_PATH}`")

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown("**⚙️ Settings**")

    use_calibration = st.checkbox(
        "Document calibration", value=True,
        help="Adaptive thresholds for passport/ID/studio photos")
    show_forensics = st.checkbox(
        "Show forensic scores", value=True,
        help="Display per-face artifact, texture, FFT scores")

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown("**🎬 Video Settings**")

    sample_fps = st.slider(
        "Sample rate (fps)", min_value=0.5, max_value=5.0,
        value=1.0, step=0.5,
        help="How many frames per second to analyse")
    max_frames = st.slider(
        "Max frames", min_value=30, max_value=500,
        value=150, step=30,
        help="Hard cap on frames to analyse per video")

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown("**📊 Model info**")
    st.metric("Architecture", "Dual ResNet")
    st.metric("RGB branch",   "ResNet-50")
    st.metric("FFT branch",   "ResNet-18")
    st.metric("TTA views",    "5-view")
    import torch
    st.metric("Device", "CUDA" if torch.cuda.is_available() else "CPU")

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown("**💡 Tips**")
    st.info(
        "• Images: JPG, JPEG, PNG, **WebP**\n"
        "• Videos: MP4, MOV, AVI, MKV, WebM\n"
        "• Face must be clearly visible\n"
        "• Passport/ID photos auto-detected\n"
        "• Group photos: each face independent\n"
        "• Longer videos → increase sample fps\n"
        "  for finer temporal analysis"
    )
    st.caption("© 2024 DeepFace Detection  |  v6.1 fixed")


# ═══════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">◈ &nbsp; Forensic AI Analysis System &nbsp; ◈</div>
  <div class="hero-title">DEEP<span>FACE</span><br>DETECTION</div>
  <div class="hero-sub">
    dual-branch cnn &nbsp;·&nbsp; 5-view tta &nbsp;·&nbsp; per-face independent inference
    &nbsp;·&nbsp; forensic composite scoring &nbsp;·&nbsp; video frame analysis
  </div>
  <div class="hero-tags">
    <span class="htag htag-cyan">FFT + RGB fusion</span>
    <span class="htag htag-green">MediaPipe face gate</span>
    <span class="htag htag-pink">PRNU noise residual</span>
    <span class="htag htag-cyan">Chrominance analysis</span>
    <span class="htag htag-green">Document adaptive</span>
    <span class="htag htag-pink">Video temporal analysis</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# MODE SELECTOR
# ═══════════════════════════════════════════════════════════════════════
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Image"

col_m1, col_m2, _ = st.columns([1, 1, 5])
with col_m1:
    if st.button(
        "🖼  Image",
        use_container_width=True,
        type="primary" if st.session_state.input_mode == "Image" else "secondary",
    ):
        st.session_state.input_mode = "Image"
        st.rerun()
with col_m2:
    if st.button(
        "🎬  Video",
        use_container_width=True,
        type="primary" if st.session_state.input_mode == "Video" else "secondary",
    ):
        st.session_state.input_mode = "Video"
        st.rerun()

mode = st.session_state.input_mode
st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# UPLOAD + FEATURE PANEL
# ═══════════════════════════════════════════════════════════════════════
col_up, col_feat = st.columns([1.6, 1], gap="large")

with col_up:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if mode == "Image":
        st.markdown('<div class="card-title">◈ Upload image</div>', unsafe_allow_html=True)
        st.caption("Portrait · Group · Passport / ID · Any face photo — JPG, PNG, WebP")
        uploaded = st.file_uploader(
            "Upload image", type=IMAGE_TYPES, label_visibility="collapsed"
        )
        uploaded_video = None
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, use_container_width=True)

    else:
        st.markdown('<div class="card-title">◈ Upload video</div>', unsafe_allow_html=True)
        st.caption("MP4 · MOV · AVI · MKV · WebM — frame-by-frame deepfake analysis")
        uploaded_video = st.file_uploader(
            "Upload video", type=VIDEO_TYPES, label_visibility="collapsed"
        )
        uploaded = None
        if uploaded_video:
            _suffix = "." + uploaded_video.name.rsplit(".", 1)[-1].lower()
            _preview_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix)
            _preview_tmp.write(uploaded_video.read())
            _preview_tmp.flush()
            _preview_tmp.close()
            uploaded_video.seek(0)

            st.video(_preview_tmp.name)

            _cap = cv2.VideoCapture(_preview_tmp.name)
            if _cap.isOpened():
                _fps = _cap.get(cv2.CAP_PROP_FPS) or 0
                _nf  = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                _w   = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                _h   = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                _dur = _nf / _fps if _fps > 0 else 0
                _cap.release()
                st.markdown(
                    f"<div style='font-family:var(--mono);font-size:.72rem;"
                    f"color:var(--muted);margin-top:.6rem;display:flex;gap:1.5rem;'>"
                    f"<span>📐 {_w}×{_h}</span>"
                    f"<span>⏱ {_dur:.1f}s</span>"
                    f"<span>🎞 {_fps:.1f} fps</span>"
                    f"<span>🖼 {_nf} frames</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.session_state["_video_preview_path"] = _preview_tmp.name
        else:
            st.session_state.pop("_video_preview_path", None)

    st.markdown('</div>', unsafe_allow_html=True)

with col_feat:
    if mode == "Image":
        feats = [
            ("Dual-branch CNN",   "RGB spatial + FFT frequency domains"),
            ("5-view TTA",        "Original · flip · ±3° rotation · centre-crop"),
            ("Wider social thr.", "real<0.35 / fake>0.65 — less UNCERTAIN"),
            ("Forensic override", "3-signal threshold (was 4) — resolves borderlines"),
            ("Cal. floor",        "min 0.15 prob floor — stops factor compounding"),
            ("PRNU residual",     "Noise fingerprint for GAN artifact detection"),
            ("Chrominance",       "Cb/Cr channel inconsistency detection"),
            ("Doc adaptive",      "Passport/ID: relaxed thresholds + calibration"),
            ("Eye reflection",    "L/R highlight consistency scoring"),
            ("FFT ring",          "Radial GAN periodic spike detection"),
            ("WebP support",      "WebP images decoded via PIL before inference"),
        ]
    else:
        feats = [
            ("Frame sampling",    "Configurable FPS-based frame extraction"),
            ("Per-frame predict", "Each frame independently classified"),
            ("Temporal analysis", "Rolling window consistency check"),
            ("Weighted voting",   "Confidence-weighted majority verdict"),
            ("Transition detect", "Flags sudden real/fake label switches"),
            ("PRNU residual",     "Noise fingerprint for GAN artifact detection"),
            ("FFT ring",          "Radial GAN periodic spike detection"),
            ("Dual-branch CNN",   "RGB spatial + FFT frequency domains"),
            ("Face gate",         "Per-frame NO_FACE fallback to full frame"),
            ("JSON report",       "Full frame-level results exportable as JSON"),
            ("Annotated video",   "Optional output video with bbox overlays"),
        ]
    inner = "".join(
        f'<div class="feat-item">'
        f'<span class="feat-dot">▸</span>'
        f'<span><strong style="color:#cdd9e5;">{t}</strong>'
        f'<br><span style="font-size:.72rem;">{d}</span></span>'
        f'</div>'
        for t, d in feats
    )
    st.markdown(
        f'<div class="card"><div class="card-title">◈ Pipeline</div>{inner}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# IMAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
if mode == "Image" and uploaded and model is not None:
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;
                font-weight:700;letter-spacing:2px;text-transform:uppercase;
                margin-bottom:1rem;">◈ Analysis Results</div>
    """, unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img_pil.save(tmp, format="JPEG", quality=95)
        tmp_path = tmp.name

    with st.spinner("Running forensic analysis …"):
        t0      = time.time()
        result  = predict_image(model, tmp_path, use_calibration=use_calibration)
        elapsed = time.time() - t0

    os.unlink(tmp_path)

    if "error" in result:
        st.error(f"❌  {result['error']}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verdict",       result["overall"])
    m2.metric("Faces",         result["total_faces"])
    avg_str = f"{result['avg_prob']:.2%}" if result["avg_prob"] is not None else "N/A"
    m3.metric("Avg fake prob", avg_str)
    m4.metric("Analysis time", f"{elapsed:.2f}s")

    st.markdown("<br>", unsafe_allow_html=True)
    overall_banner(result)

    # FIX B: check both NO_FACE label variants
    if result["overall"] in _NO_FACE_LABELS:
        st.markdown(f"""
        <div class="card" style="border-color:var(--noface);">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;
                      font-weight:700;color:var(--noface);margin-bottom:.5rem;">
            🚫 No Human Face Detected
          </div>
          <div style="color:var(--muted);font-size:.82rem;line-height:1.7;">
            {result.get('note', '')}
          </div>
          <div style="margin-top:1rem;font-size:.75rem;color:var(--muted2);">
            The deepfake detector only analyses images containing human faces.<br>
            Please upload a photo with a clearly visible face.
          </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    if result.get("doc_photo"):
        st.markdown(
            '<div class="doc-pill">📄 Document / passport / ID photo '
            '— adaptive thresholds applied</div>',
            unsafe_allow_html=True,
        )

    img_np = np.array(img_pil)

    for f in result["per_face"]:
        fid   = f["face_id"]
        label = f["label"]
        prob  = f["probability"]
        conf  = f.get("confidence", round(
            (1 - prob if label == "REAL" else prob) * 100, 1))
        total = result["total_faces"]

        st.markdown(
            f'<div class="face-sep">'
            f'face {fid + 1} / {total} &nbsp;·&nbsp; {label}'
            f' &nbsp;·&nbsp; conf {conf:.1f}%'
            f'</div>',
            unsafe_allow_html=True,
        )

        fc_col, res_col = st.columns([1, 1.5], gap="large")

        with fc_col:
            bbox = f.get("bbox")
            if bbox:
                x, y, w, h = bbox
                pad  = int(max(w, h) * 0.18)
                H, W = img_np.shape[:2]
                crop = img_np[
                    max(0, y - pad): min(H, y + h + pad),
                    max(0, x - pad): min(W, x + w + pad),
                ]
                st.image(crop, use_container_width=True, caption=f"Face {fid + 1}")
            else:
                st.image(img_np, use_container_width=True, caption="Full image fallback")
            if f.get("note"):
                st.caption(f"ℹ️  {f['note']}")

        with res_col:
            render_verdict(label, prob)

            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{conf:.1f}%")
            c2.metric("Fake prob",  f"{prob:.2%}")
            c3.metric("Forensic",   f"{f.get('forensic_composite', 0):.3f}")

            if show_forensics:
                with st.expander("🔬 Forensic breakdown", expanded=True):
                    render_forensic_grid(f)
                    st.markdown("""
                    <div style="font-size:.62rem;color:var(--muted);
                                margin-top:.6rem;line-height:1.8;font-family:var(--mono);">
                    artifact = edge + HF + texture + PRNU + chrominance &nbsp;|&nbsp;
                    texture = skin micro-detail &nbsp;|&nbsp; symmetry = L/R pixel diff &nbsp;|&nbsp;
                    eye_ref = highlight consistency &nbsp;|&nbsp; lighting = L/R luminance &nbsp;|&nbsp;
                    fft_ring = radial GAN spike &nbsp;|&nbsp;
                    forensic = weighted composite (0 = real · 1 = fake)
                    </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# VIDEO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif mode == "Video" and uploaded_video and model is not None:
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)

    tmp_video_path = st.session_state.get("_video_preview_path")

    if not tmp_video_path or not os.path.exists(tmp_video_path):
        st.warning("⚠️  Video preview not ready — please re-upload the file.")
    else:
        analyse_clicked = st.button(
            "🔍  Analyse Video for Deepfakes",
            type="primary",
            use_container_width=False,
        )

        if analyse_clicked:
            st.markdown("""
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;
                        font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        margin-bottom:1rem;">◈ Video Analysis Results</div>
            """, unsafe_allow_html=True)

            report_tmp = tmp_video_path + "_report.json"

            with st.spinner("🎬 Extracting frames and running forensic analysis … (this may take a minute)"):
                t0 = time.time()
                video_result = predict_video(
                    model,
                    tmp_video_path,
                    sample_fps      = sample_fps,
                    max_frames      = max_frames,
                    use_calibration = use_calibration,
                    report_path     = report_tmp,
                )
                elapsed = time.time() - t0

            if "error" in video_result:
                st.error(f"❌  {video_result['error']}")
            else:
                verdict    = video_result["verdict"]
                # FIX E: reads "confidence" key (guaranteed by predict_video v1.1)
                confidence = video_result["confidence"]
                aggregate  = video_result["aggregate"]
                temporal   = video_result["temporal"]
                meta       = video_result["meta"]
                frames_    = video_result.get("per_frame", [])

                render_video_verdict(verdict, confidence, aggregate, temporal, meta)
                st.caption(f"⏱  Analysis completed in {elapsed:.1f}s")

                with st.expander("📊 Per-frame timeline", expanded=True):
                    render_frame_timeline(frames_)

                if os.path.exists(report_tmp):
                    with open(report_tmp, "rb") as f:
                        st.download_button(
                            label="📄 Download JSON report",
                            data=f.read(),
                            file_name=f"{uploaded_video.name}_report.json",
                            mime="application/json",
                        )
                    os.unlink(report_tmp)
        else:
            st.markdown("""
            <div style="background:var(--surface);border:1px solid var(--border2);
                        border-left:3px solid var(--accent);border-radius:8px;
                        padding:1rem 1.2rem;font-family:var(--mono);font-size:.82rem;
                        color:var(--muted);margin-top:.5rem;">
              ✅ &nbsp; Video loaded successfully. Press <strong style="color:var(--accent);">
              Analyse Video for Deepfakes</strong> above to begin frame-level analysis.
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# MODEL NOT LOADED
# ═══════════════════════════════════════════════════════════════════════
elif (uploaded or uploaded_video) and model is None:
    st.markdown(f"""
    <div class="card" style="border-color:var(--fake);text-align:center;padding:2rem;">
      <div style="font-size:2.5rem;">❌</div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;
                  font-weight:700;color:var(--fake);margin:.5rem 0;">Model Not Loaded</div>
      <div style="color:var(--muted);font-size:.82rem;">
        Expected weights at: <code>{MODEL_PATH}</code><br>
        Update <code>MODEL_PATH</code> in app.py to point to your .pth file.
      </div>
    </div>""", unsafe_allow_html=True)

else:
    icon    = "🖼️" if mode == "Image" else "🎬"
    label   = "image" if mode == "Image" else "video"
    formats = "JPG · PNG · WebP" if mode == "Image" else "MP4 · MOV · AVI · MKV · WebM"
    note    = (
        "Portraits · group photos · passport / ID photos<br>"
        "Images without a human face return a <strong style='color:var(--noface);'>NO_FACE</strong> verdict<br>"
        "Each face in a group photo is judged independently"
        if mode == "Image" else
        "Each frame is independently analysed for deepfake content<br>"
        "Temporal consistency across frames improves accuracy<br>"
        "Use the sidebar to tune sample rate and max frames"
    )
    st.markdown(f"""
    <div class="card" style="text-align:center;padding:3.5rem 2rem;border-style:dashed;">
      <div style="font-size:3rem;margin-bottom:1rem;">{icon}</div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;
                  font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:.6rem;">
        Upload a {label} to begin
      </div>
      <div style="color:var(--muted);font-size:.82rem;line-height:1.8;margin-bottom:.8rem;">
        {note}
      </div>
      <div style="color:var(--muted2);font-size:.72rem;letter-spacing:1px;font-family:var(--mono);">
        Supported formats: {formats}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  DEEPFACE DETECTION &nbsp;·&nbsp; EliteDetector v6.1 (fixed) &nbsp;·&nbsp;
  Dual-branch ResNet (RGB + FFT) &nbsp;·&nbsp; 5-view TTA &nbsp;·&nbsp;
  per-face forensic analysis &nbsp;·&nbsp; video frame-level deepfake detection
</div>
""", unsafe_allow_html=True)