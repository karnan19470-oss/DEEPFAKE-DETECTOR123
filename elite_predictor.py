"""
elite_predictor.py  (v8.0 — Uncertain-Zone Fix)
════════════════════════════════════════════════════════════════════════════════
ROOT CAUSE ANALYSIS — why v7.0 predicts nearly everything as UNCERTAIN
════════════════════════════════════════════════════════════════════════════════

PROBLEM 1 — Four suppressor stages still multiply in series
─────────────────────────────────────────────────────────────
  v7.0 claimed to fix the multiplier chain, but four independent downward
  corrections still fire in sequence on every clean-background face:

    Stage 3: calibrate_skintone()  ×0.76 (medium-dark, lum 120, prob just < 0.65)
    Stage 5: nudge                 ×0.75 (dark-skin-guard, doc domain)
    Stage 6: _boost_confidence()   ×0.65 (hidden: fires when art<0.30 AND prob in uncertain band)
    Stage 7: _calibration_factor() ×0.58 (art<0.20, tex>0.08, document domain)

    product = 0.76 × 0.75 × 0.65 × 0.58 = 0.215

  Starting from prob=0.70, that gives 0.70 × 0.215 = 0.150, rescued to the
  FIX D floor of 0.32 — still inside the UNCERTAIN zone [0.28, 0.74].

PROBLEM 2 — Uncertain band is enormous: [0.28, 0.74] = 46% of [0,1]
─────────────────────────────────────────────────────────────────────
  The four suppressors reliably push any prob that starts <0.85 into this zone.
  And FIX D then floors it at 0.32, keeping it in the zone.
  Result: REAL requires reaching prob ≤ 0.28, FAKE requires ≥ 0.74.
  After four suppressors and a floor at 0.32, neither is reachable for
  normal inputs.

PROBLEM 3 — _boost_confidence() contains a hidden third suppressor
────────────────────────────────────────────────────────────────────
  Inside _boost_confidence(), for DOCUMENT/STUDIO/PHONE domains, there is:

      if art < 0.30 and tex > 0.05 and low <= prob <= high:
          prob *= 0.65

  This fires on virtually every clean face image (low artifacts, visible
  texture) after the nudge already reduced prob. It was meant to help
  "real-looking" images, but instead it compresses the probability further
  into uncertain territory on top of stages 3 and 5.

PROBLEM 4 — FIX G cutoff at 0.65 creates a cliff
──────────────────────────────────────────────────
  Skin-tone correction is skipped only if prob ≥ 0.65. But after TTA the
  model often outputs values like 0.62–0.64 for genuine fakes — just below
  the cutoff. Those values still receive the full 0.76× suppression,
  dropping them to ~0.47, then nudges push them to ~0.32 (floor territory).

FIXES IN v8.0
════════════════════════════════════════════════════════════════════════════════

  [FIX 1] Remove the hidden _boost_confidence() suppressor entirely.
           The ×0.65 for clean images inside the uncertain band is removed.
           The forensic REAL_PUSH/FAKE_PUSH boosts are retained.

  [FIX 2] Stage 5 nudge strength halved again for document/studio.
           Nudge was 0.75 for dark skin in v7.0. Reduced to 0.87 (13% pull
           toward real). The calibration in stage 7 still handles this; a
           strong nudge here just pre-compresses the signal.

  [FIX 3] Stage 3 FIX G threshold: raised from 0.65 → 0.55.
           Model values ≥ 0.55 (moderately fake) should not be suppressed
           by skin-tone correction. The original intent was to protect strong
           fake signals; 0.65 was set too high given the surrounding suppressors.

  [FIX 4] _calibration_factor() minimum raised from 0.52 → 0.70.
           The remaining combined multiplier (stage3 × stage5 × stage7) must
           not be less than ~0.65 for any single clean image. Setting the
           floor of each individual factor to 0.70 achieves this.

  [FIX 5] Threshold narrowing: widen REAL band, narrow uncertain zone.
           DOCUMENT: real_thr 0.28→0.38, fake_thr 0.74→0.70.
           This means the uncertain band shrinks from [0.28,0.74] (46%)
           to [0.38, 0.70] (32%). Crucially the floor (0.32) now sits
           BELOW real_thr (0.38), so a calibrated-down genuine REAL
           image can actually reach the REAL verdict.

  [FIX 6] FIX D floor lowered for DOCUMENT: 0.32 → 0.22.
           The purpose of the floor is to prevent a clearly-fake model
           output from being calibrated down to zero. 0.22 achieves this
           (well below any fake signal) while no longer anchoring images
           inside the uncertain zone. The floor is a safety net, not a default.

  [FIX 7] Combined multiplier guard: after stages 3+5+7, if the total
           reduction exceeds 60% of the original prob, the result is
           clamped to 0.40× of the original (maximum 60% reduction across
           the whole pipeline). This is the definitive anti-compounding fix.
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

# ── Optional: MediaPipe ───────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

DOMAIN_DOCUMENT = "document"
DOMAIN_STUDIO   = "studio"
DOMAIN_OUTDOOR  = "outdoor"
DOMAIN_PHONE    = "phone"
DOMAIN_SOCIAL   = "social"

# ─────────────────────────────────────────────────────────────────────────────
# FIX 5: Threshold narrowing.
#   DOCUMENT: real_thr 0.28→0.38, fake_thr 0.74→0.70.
#   Uncertain band shrinks from [0.28,0.74] to [0.38,0.70].
#   The FIX 6 floor of 0.22 now sits BELOW real_thr=0.38, so calibrated
#   real images can reach REAL without hitting the floor first.
#   Other domains adjusted proportionally to avoid the same cliff.
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_THRESHOLDS = {
    DOMAIN_DOCUMENT: dict(real_thr=0.38, fake_thr=0.70, unc_low=0.38, unc_high=0.70),  # FIX 5
    DOMAIN_STUDIO:   dict(real_thr=0.36, fake_thr=0.72, unc_low=0.36, unc_high=0.72),  # FIX 5
    DOMAIN_OUTDOOR:  dict(real_thr=0.38, fake_thr=0.72, unc_low=0.38, unc_high=0.72),  # FIX 5
    DOMAIN_PHONE:    dict(real_thr=0.36, fake_thr=0.70, unc_low=0.36, unc_high=0.70),  # FIX 5
    DOMAIN_SOCIAL:   dict(real_thr=0.40, fake_thr=0.65, unc_low=0.40, unc_high=0.65),
}

FORENSIC_OVERRIDE_COUNT = 3

# ─────────────────────────────────────────────────────────────────────────────
# FIX 7: Maximum allowed combined pipeline reduction.
#   After all calibration stages, if total multiplicative reduction > 60%,
#   the final prob is clamped to max(final_prob, raw_prob * 0.40).
#   0.40 means at most a 60% total reduction from the raw model output.
# ─────────────────────────────────────────────────────────────────────────────
MAX_PIPELINE_REDUCTION = 0.40   # prob cannot drop below raw_prob * this factor


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class EliteDetector(nn.Module):
    """
    Dual-branch deepfake detector:
      • RGB branch  — ResNet-50 for spatial / texture / color features
      • FFT branch  — ResNet-18 with 1-channel input for frequency artifacts
    Combined via a 3-layer MLP classifier head.
    """
    def __init__(self):
        super().__init__()
        self.rgb_net    = models.resnet50(weights=None)
        self.rgb_net.fc = nn.Identity()

        self.fft_net           = models.resnet18(weights=None)
        self.fft_net.conv1     = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.fft_net.fc        = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, rgb: torch.Tensor, fft: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            torch.cat([self.rgb_net(rgb), self.fft_net(fft)], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def load_model(path: str) -> EliteDetector:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights not found: {path}\n"
            f"Please ensure 'elite_resnet_detector.pth' is in the current directory.")

    model = EliteDetector()
    ckpt  = torch.load(path, map_location=DEVICE)

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    for p in model.parameters():
        p.data = p.data.float()

    model.to(DEVICE).eval()
    print(f"[EliteDetector v8.0] Model loaded — device={DEVICE}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TRANSFORMS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def _compute_fft(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    spec = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    return np.log(np.abs(spec) + 1.0)


def _fft_to_tensor(mag: np.ndarray) -> torch.Tensor:
    r = cv2.resize(mag, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(r).unsqueeze(0)
    return (t - t.mean()) / (t.std() + 1e-7)


# ══════════════════════════════════════════════════════════════════════════════
# FACE DETECTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_dnn_net = None


def _haar_detect(rgb: np.ndarray) -> list:
    gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    return [] if not len(faces) else [(int(x), int(y), int(w), int(h))
                                       for x, y, w, h in faces]


def _dnn_detect(rgb: np.ndarray) -> list:
    global _dnn_net
    if _dnn_net is None:
        proto   = "deploy.prototxt"
        weights = "res10_300x300_ssd_iter_140000.caffemodel"
        if os.path.exists(proto) and os.path.exists(weights):
            _dnn_net = cv2.dnn.readNetFromCaffe(proto, weights)
        else:
            _dnn_net = "haar"

    if _dnn_net == "haar":
        return _haar_detect(rgb)

    h, w = rgb.shape[:2]
    bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blob  = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)),
                                   1.0, (300, 300), (104, 177, 123))
    _dnn_net.setInput(blob)
    dets = _dnn_net.forward()
    out  = []
    for i in range(dets.shape[2]):
        if dets[0, 0, i, 2] < 0.45:
            continue
        b = (dets[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        fw, fh = b[2] - b[0], b[3] - b[1]
        if fw >= 30 and fh >= 30:
            out.append((b[0], b[1], fw, fh))
    return out


def _mp_detect(rgb: np.ndarray) -> list:
    with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.25) as det:
        res = det.process(rgb)
        if not res.detections:
            return []
        H, W = rgb.shape[:2]
        out = []
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            x  = max(0, int(bb.xmin * W))
            y  = max(0, int(bb.ymin * H))
            fw = int(bb.width  * W)
            fh = int(bb.height * H)
            if fw >= 30 and fh >= 30:
                out.append((x, y, fw, fh))
        return out


def detect_faces(rgb: np.ndarray) -> list:
    if _MP_AVAILABLE:
        faces = _mp_detect(rgb)
        if faces:
            return faces
    faces = _dnn_detect(rgb)
    if faces:
        return faces
    return _haar_detect(rgb)


def crop_padded(rgb: np.ndarray, x: int, y: int, w: int, h: int,
                pad: float = 0.25) -> np.ndarray:
    H, W = rgb.shape[:2]
    x1 = max(0, int(x - pad * w))
    y1 = max(0, int(y - pad * h))
    x2 = min(W, int(x + w + pad * w))
    y2 = min(H, int(y + h + pad * h))
    return rgb[y1:y2, x1:x2]


def good_quality(img: np.ndarray) -> bool:
    if img.shape[0] < 48 or img.shape[1] < 48:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 12.0:
        return False
    if np.std(img) < 8.0:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# NON-PHOTO PRE-FILTER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _skin_saturation(img_rgb: np.ndarray) -> float:
    h, w = img_rgb.shape[:2]
    crop = img_rgb[h//4 : 3*h//4, w//4 : 3*w//4]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    return float(np.mean(hsv[:, :, 1]) / 255.0)


def _hue_diversity(img_rgb: np.ndarray) -> float:
    h, w = img_rgb.shape[:2]
    crop = img_rgb[h//4 : 3*h//4, w//4 : 3*w//4]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    return float(np.std(hsv[:, :, 0]))


def _specular_blob_score(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blobs  = cv2.erode(bright, kernel, iterations=1)
    return float(np.sum(blobs > 0) / max(gray.size, 1))


def _texture_isotropy(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ex   = float(np.mean(gx ** 2)) + 1e-7
    ey   = float(np.mean(gy ** 2)) + 1e-7
    return float(abs(ex / ey - 1.0))


def _rgb_channel_correlation(img_rgb: np.ndarray) -> float:
    h, w = img_rgb.shape[:2]
    crop = img_rgb[h//4 : 3*h//4, w//4 : 3*w//4].astype(np.float32)
    r, g, b = crop[:,:,0].ravel(), crop[:,:,1].ravel(), crop[:,:,2].ravel()
    def _corr(a, b):
        return float(np.mean((a - a.mean()) * (b - b.mean())) /
                     (np.std(a) * np.std(b) + 1e-7))
    return (_corr(r, g) + _corr(g, b)) / 2.0


def is_non_photo_face(img_rgb: np.ndarray) -> tuple:
    sat    = _skin_saturation(img_rgb)
    hue_d  = _hue_diversity(img_rgb)
    spec   = _specular_blob_score(img_rgb)
    aniso  = _texture_isotropy(img_rgb)
    ch_cor = _rgb_channel_correlation(img_rgb)

    n1 = not (0.15 <= sat <= 0.72)
    n2 = hue_d < 6.5
    n3 = spec  > 0.060
    n4 = aniso > 0.60
    n5 = ch_cor < 0.70

    count = sum([n1, n2, n3, n4, n5])
    return count >= 3, dict(sat=round(sat,3), hue_div=round(hue_d,2),
                             spec=round(spec,4), aniso=round(aniso,3),
                             ch_cor=round(ch_cor,3), signals=f"{count}/5")


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN DETECTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _peripheral_bg_std(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    m    = max(10, int(w * 0.08))
    patches = [
        img_bgr[0          : int(h*0.08), 0   : m],
        img_bgr[0          : int(h*0.08), w-m : w],
        img_bgr[int(h*0.08): int(h*0.50), 0   : m],
        img_bgr[int(h*0.08): int(h*0.50), w-m : w],
        img_bgr[int(h*0.50): int(h*0.85), 0   : m],
        img_bgr[int(h*0.50): int(h*0.85), w-m : w],
    ]
    stds = [np.std(p) for p in patches if p.size > 0]
    return float(np.mean(stds)) if stds else 999.0


def _bg_hsv_saturation(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    m    = max(10, int(w * 0.08))
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    patches = [
        hsv[0          : int(h*0.08), 0   : m, 1],
        hsv[0          : int(h*0.08), w-m : w, 1],
        hsv[int(h*0.08): int(h*0.50), 0   : m, 1],
        hsv[int(h*0.08): int(h*0.50), w-m : w, 1],
    ]
    vals = np.concatenate([p.ravel() for p in patches if p.size > 0])
    return float(np.mean(vals)) / 255.0


def _outdoor_complexity(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    top  = img_bgr[: int(h * 0.20), :]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.Canny(gray, 30, 100) > 0))


def _centered_face_score(img_bgr: np.ndarray) -> float:
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detect_faces(rgb)
    if not faces:
        return 0.0
    w = img_bgr.shape[1]
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return float(1.0 - abs((x + fw / 2) - w / 2) / (w / 2 + 1e-7))


def _jpeg_compression_level(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    if h > 16:
        a = gray[7:h:8, :]
        b = gray[8:h:8, :]
        rows = min(a.shape[0], b.shape[0])
        h_diff = np.abs(a[:rows] - b[:rows])
    else:
        h_diff = np.array([0.0])

    if w > 16:
        a = gray[:, 7:w:8]
        b = gray[:, 8:w:8]
        cols = min(a.shape[1], b.shape[1])
        v_diff = np.abs(a[:, :cols] - b[:, :cols])
    else:
        v_diff = np.array([0.0])

    return float((np.mean(h_diff) + np.mean(v_diff)) / 2) / 255.0


def _depth_of_field_score(img_bgr: np.ndarray) -> float:
    h, w  = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face  = gray[int(h*0.15):int(h*0.60), int(w*0.20):int(w*0.80)]
    bg    = gray[int(h*0.80):, :]
    if face.size == 0 or bg.size == 0:
        return 0.0
    fs = cv2.Laplacian(face, cv2.CV_64F).var()
    bs = cv2.Laplacian(bg,   cv2.CV_64F).var()
    return float(np.clip(fs / (bs + 1e-6) / 50, 0, 1))


def detect_domain(img_bgr: np.ndarray) -> str:
    h, w       = img_bgr.shape[:2]
    aspect     = h / max(w, 1)
    is_portrait = 1.0 <= aspect <= 1.95

    outdoor_cpx = _outdoor_complexity(img_bgr)
    periph_std  = _peripheral_bg_std(img_bgr)
    hsv_sat     = _bg_hsv_saturation(img_bgr)

    if outdoor_cpx > 0.10:
        return DOMAIN_OUTDOOR

    if is_portrait:
        gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edge_full = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
        centred   = _centered_face_score(img_bgr)
        s1 = periph_std < 50
        s2 = hsv_sat    < 0.30
        s3 = edge_full  < 0.07
        s4 = centred    > 0.60
        if sum([s1, s2, s3, s4]) >= 2:
            return DOMAIN_DOCUMENT

    jpeg_block = _jpeg_compression_level(img_bgr)
    dof        = _depth_of_field_score(img_bgr)
    if sum([jpeg_block > 0.003, dof > 0.30, is_portrait, periph_std > 20]) >= 3:
        return DOMAIN_PHONE

    if periph_std < 55 and hsv_sat < 0.35:
        return DOMAIN_STUDIO

    return DOMAIN_SOCIAL


# ══════════════════════════════════════════════════════════════════════════════
# FORENSIC SIGNALS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def artifact_score(img: np.ndarray) -> float:
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    mag   = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
    h, w  = mag.shape
    hf    = max(0.0, float(mag.mean() - mag[h//4:3*h//4, w//4:3*w//4].mean()))
    tex   = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500)
    return float(np.clip(0.4*edges + 0.4*min(1.0, hf/10) + 0.2*tex, 0, 1))


def fft_ring_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
    h, w = mag.shape
    y, x = np.indices((h, w))
    r    = np.sqrt((x - w//2)**2 + (y - h//2)**2).astype(np.int32)
    cnt  = np.bincount(r.ravel())
    rad  = np.bincount(r.ravel(), mag.ravel()) / np.maximum(cnt, 1)
    return float(np.clip(np.max(np.abs(np.diff(rad))) / 10, 0, 1))


def skin_texture_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F))) / 100)


def face_symmetry_score(img: np.ndarray) -> float:
    h, w  = img.shape[:2]
    left  = img[:, :w//2]
    right = cv2.flip(img[:, w//2:], 1)
    mw    = min(left.shape[1], right.shape[1])
    return float(np.mean(np.abs(
        left[:, :mw].astype(np.float32) -
        right[:, :mw].astype(np.float32))) / 255)


def illumination_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, w = gray.shape
    return float(abs(np.mean(gray[:, :w//2]) - np.mean(gray[:, w//2:])) / 255)


def eye_reflection_score(img: np.ndarray) -> float:
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w   = gray.shape
    y1, y2 = int(h * .25), int(h * .45)
    le     = gray[y1:y2, int(w*.15):int(w*.40)]
    re     = gray[y1:y2, int(w*.60):int(w*.85)]
    if le.size == 0 or re.size == 0:
        return 0.0
    bd = abs(float(np.max(le)) - float(np.max(re))) / 255
    lx = np.unravel_index(np.argmax(le), le.shape)[1]
    rx = np.unravel_index(np.argmax(re), re.shape)[1]
    pd = abs(lx - (re.shape[1] - rx)) / (w + 1e-7)
    return float(np.clip(bd + pd, 0, 1))


def noise_consistency_score(img: np.ndarray) -> float:
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    h, w  = noise.shape
    quads = [noise[:h//2,:w//2], noise[:h//2,w//2:],
             noise[h//2:,:w//2], noise[h//2:,w//2:]]
    stds  = [np.std(q) for q in quads if q.size > 0]
    return float(np.clip(np.std(stds) / (np.mean(stds) + 1e-7), 0, 1))


def color_coherence_score(img: np.ndarray) -> float:
    h, w   = img.shape[:2]
    cy, cx = h // 2, w // 2
    r      = min(h, w) // 4
    Y, X   = np.ogrid[:h, :w]
    inner  = (X - cx)**2 + (Y - cy)**2 <= r**2
    outer  = ~inner
    if not inner.any() or not outer.any():
        return 0.0
    return float(np.clip(
        np.linalg.norm(img[inner].reshape(-1,3).mean(0) -
                       img[outer].reshape(-1,3).mean(0)) / 255, 0, 1))


def chrominance_noise_score(img: np.ndarray) -> float:
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    def hp(ch): return ch - cv2.GaussianBlur(ch, (7, 7), 0)
    return float(np.clip(
        (np.std(hp(ycc[:,:,2])) + np.std(hp(ycc[:,:,1]))) /
        (2 * np.std(hp(ycc[:,:,0])) + 1e-6), 0, 1))


def jpeg_grid_artifact_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft  = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = fft.shape
    cy, cx = h // 2, w // 2
    gf = []
    for k in range(1, 5):
        gf.append(fft[min(cy + k*h//8, h-1), cx])
        gf.append(fft[cy, min(cx + k*w//8, w-1)])
    return float(np.clip((np.mean(gf) / (fft.mean() + 1e-6) - 1.0) / 5.0, 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# RESOLUTION-ADAPTIVE CORRECTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _resolution_tier(img_rgb: np.ndarray) -> tuple:
    px = img_rgb.shape[0] * img_rgb.shape[1]
    if   px >= 500_000: return "high",   1.00, 1.00, 0.00, 0.00
    elif px >= 150_000: return "medium", 0.85, 0.80, 0.00, 0.00
    elif px >= 50_000:  return "low",    0.60, 0.55, 0.04, 0.02
    else:               return "tiny",   0.45, 0.40, 0.06, 0.03


# ══════════════════════════════════════════════════════════════════════════════
# OUTDOOR SCENE DETECTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _scene_complexity(img_rgb: np.ndarray) -> float:
    h, w  = img_rgb.shape[:2]
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    mask  = np.ones_like(edges, dtype=bool)
    mask[int(h*0.20):int(h*0.80), int(w*0.25):int(w*0.75)] = False
    return float(np.sum(edges[mask] > 0) / max(np.sum(mask), 1))


def _bg_asymmetry(img_rgb: np.ndarray) -> float:
    h, w = img_rgb.shape[:2]
    vals = []
    for band in [img_rgb[:int(h*0.20), :], img_rgb[int(h*0.80):, :]]:
        if band.shape[1] < 2:
            continue
        bw    = band.shape[1]
        left  = band[:, :bw//2].astype(np.float32)
        right = cv2.flip(band[:, bw//2:], 1).astype(np.float32)
        mw2   = min(left.shape[1], right.shape[1])
        vals.append(float(np.mean(np.abs(left[:,:mw2] - right[:,:mw2])) / 255))
    return float(np.mean(vals)) if vals else 0.0


def is_outdoor_scene(img_rgb: np.ndarray) -> bool:
    hsv   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    o1    = _scene_complexity(img_rgb) > 0.08
    o2    = _bg_asymmetry(img_rgb)     > 0.06
    o3    = float(np.std(hsv[:, :, 0])) > 60.0
    return sum([o1, o2, o3]) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# SKIN-TONE CALIBRATION
#
# FIX 3: FIX G threshold raised 0.65 → 0.55.
#   The model outputs prob=0.60 for genuine fakes with dark skin.
#   Under v7.0 that value was suppressed by ×0.76, then further by nudge,
#   making it unreachable as FAKE.
#   Now: any prob ≥ 0.55 is considered a strong-enough model signal — skip
#   the skin-tone correction entirely.
# ══════════════════════════════════════════════════════════════════════════════

def _mean_face_luminance(img_rgb: np.ndarray) -> float:
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray[h//4 : 3*h//4, w//4 : 3*w//4]))


def _skin_rg_ratio(img_rgb: np.ndarray) -> float:
    h, w = img_rgb.shape[:2]
    crop = img_rgb[h//4 : 3*h//4, w//4 : 3*w//4].astype(np.float32)
    return float(np.mean(crop[:,:,0] / (crop[:,:,1] + 1.0)))


def calibrate_skintone(prob: float, img_rgb: np.ndarray,
                        domain: str = DOMAIN_SOCIAL) -> tuple:
    """
    Corrects for known skin-tone bias in GAN/deepfake detectors.

    FIX 3: Skip correction if prob >= 0.55 (raised from 0.65).
            Model values in the moderate-fake range should not be suppressed.
    Domain-aware (document/studio: lighter correction).
    """
    lum    = _mean_face_luminance(img_rgb)
    rg_mod = max(0.0, (_skin_rg_ratio(img_rgb) - 1.1) * 0.10)

    if   lum > 170: base_factor, tier = 1.00,          "light"
    elif lum > 140: base_factor, tier = 0.92 - rg_mod, "medium"         # raised from 0.88
    elif lum > 95:  base_factor, tier = 0.84 - rg_mod, "medium-dark"    # raised from 0.76
    else:           base_factor, tier = 0.76 - rg_mod, "dark"           # raised from 0.66

    # Domain-aware relaxation (from v7.0 FIX A — retained)
    if domain in (DOMAIN_DOCUMENT, DOMAIN_STUDIO):
        base_factor = max(base_factor, 0.94)   # very light touch for ID photos
    elif domain == DOMAIN_PHONE:
        base_factor = max(base_factor, 0.90)

    # FIX 3: raised threshold — skip correction for moderately-fake signals too
    if prob >= 0.55:
        return float(np.clip(prob, 0.01, 0.99)), tier

    return float(np.clip(prob * base_factor, 0.01, 0.99)), tier


# ══════════════════════════════════════════════════════════════════════════════
# FORENSIC ENSEMBLE VOTE  (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

def _forensic_vote(art, tex, sym, eye, ring, noise, color,
                   chroma, jgrid, domain,
                   art_f=1.0, ring_f=1.0, color_tol=0.0, sym_tol=0.0):
    art_thr  = 0.55 if domain == DOMAIN_OUTDOOR  else 0.45
    ring_thr = 0.45 if domain == DOMAIN_OUTDOOR  else 0.35
    eye_thr  = 0.30 if domain in (DOMAIN_DOCUMENT, DOMAIN_STUDIO) else 0.22
    jg_thr   = 0.55 if domain == DOMAIN_PHONE    else 0.35

    fake_votes = sum([art    > art_thr,
                      ring   > ring_thr,
                      sym    > 0.22,
                      eye    > eye_thr,
                      noise  > 0.55,
                      color  > 0.30,
                      chroma > 0.55,
                      jgrid  > jg_thr])

    art_c  = art  * art_f
    ring_c = ring * ring_f
    real_votes = sum([art_c  < 0.20,
                      ring_c < 0.15,
                      sym    < (0.10 + sym_tol),
                      tex    > 0.08,
                      noise  < 0.28,
                      color  < (0.12 + color_tol),
                      chroma < 0.25,
                      jgrid  < 0.10])

    if fake_votes >= FORENSIC_OVERRIDE_COUNT:
        return "FAKE_PUSH", real_votes, fake_votes
    if real_votes >= FORENSIC_OVERRIDE_COUNT:
        return "REAL_PUSH", real_votes, fake_votes
    return None, real_votes, fake_votes


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN CALIBRATION
#
# FIX 4: Minimum calibration factor raised to 0.70 (was 0.52 in v7.0).
#   Combined multiplier budget:
#     Stage 3 (skintone) × Stage 5 (nudge) × Stage 7 (calibration)
#   must not push the probability below raw_prob × 0.40.
#   With Stage 3 max ×0.84, Stage 5 max ×0.87, this factor must be ≥ 0.55.
#   We use 0.70 for safety margin.
# ══════════════════════════════════════════════════════════════════════════════

def _calibration_factor(art: float, tex: float, domain: str) -> float:
    # FIX 4: floor of 0.70 on all branches (was 0.52 in v7.0)
    if domain in (DOMAIN_DOCUMENT, DOMAIN_STUDIO):
        if   art < 0.12 and tex > 0.10: return 0.70   # was 0.52
        elif art < 0.20 and tex > 0.08: return 0.74   # was 0.58
        elif art < 0.25 and tex > 0.05: return 0.80   # was 0.65
        elif art < 0.30:                return 0.85   # was 0.72
        elif art < 0.40:                return 0.92   # was 0.82
        elif art > 0.75:                return 1.25   # unchanged (fake signal)
        elif art > 0.60:                return 1.10   # unchanged
        else:                           return 0.95   # was 0.88
    elif domain == DOMAIN_OUTDOOR:
        if   art < 0.20:  return 0.78   # was 0.68
        elif art < 0.35:  return 0.88   # was 0.80
        elif art > 0.80:  return 1.15   # unchanged
        elif art > 0.65:  return 1.05   # unchanged
        else:             return 0.94   # was 0.90
    elif domain == DOMAIN_PHONE:
        if   art < 0.20 and tex > 0.06: return 0.74  # was 0.58
        elif art < 0.30:                return 0.82  # was 0.68
        elif art < 0.40:                return 0.90  # was 0.80
        elif art > 0.75:                return 1.18  # unchanged
        elif art > 0.60:                return 1.05  # unchanged
        else:                           return 0.93  # was 0.88
    return 1.00


def calibrate_domain(prob: float, img: np.ndarray, domain: str) -> float:
    art    = artifact_score(img)
    tex    = skin_texture_score(img)
    factor = _calibration_factor(art, tex, domain)
    return float(np.clip(prob * factor, 0.01, 0.99))


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE BOOSTING
#
# FIX 1: Remove the hidden ×0.65 suppressor for clean images.
#   v7.0 contained this block inside _boost_confidence():
#
#       if domain in (DOCUMENT, STUDIO, PHONE):
#           if art < 0.30 and tex > 0.05 and low <= prob <= high:
#               prob *= 0.65
#
#   This multiplier fires on virtually every clean face that ends up in the
#   uncertain band — which after stages 3 and 5 is almost everything.
#   A fourth suppressor on top of three others is pure compounding noise.
#   REMOVED entirely. The forensic REAL_PUSH and FAKE_PUSH boosts are retained.
# ══════════════════════════════════════════════════════════════════════════════

def _boost_confidence(prob, override, domain, art, tex, rv=0, fv=0) -> float:
    thr  = DOMAIN_THRESHOLDS[domain]
    low  = thr["unc_low"]
    high = thr["unc_high"]

    # Forensic-vote-driven boosts (retained from v7.0)
    if override == "REAL_PUSH" and prob > low:
        strength = 0.82 + 0.025 * max(0, rv - 4)
        prob     = prob + strength * (0.32 - prob)
    elif override == "FAKE_PUSH" and prob < high:
        strength = 0.82 + 0.025 * max(0, fv - 4)
        prob     = prob + strength * (0.68 - prob)

    # FIX 1: The ×0.65 block is intentionally removed here.
    # (It was: if art<0.30 and tex>0.05 and low<=prob<=high: prob *= 0.65)
    # That multiplier was the 4th compounding suppressor; see module docstring.

    return float(np.clip(prob, 0.01, 0.99))


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE + TTA  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _infer_once(model: EliteDetector, img: np.ndarray) -> float:
    rgb_t = _rgb_transform(img).unsqueeze(0).to(DEVICE).float()
    fft_t = _fft_to_tensor(_compute_fft(img)).unsqueeze(0).to(DEVICE).float()
    with torch.no_grad():
        return F.softmax(model(rgb_t, fft_t), dim=1)[0][1].item()


def tta_predict(model: EliteDetector, img: np.ndarray) -> float:
    h, w = img.shape[:2]
    p1   = _infer_once(model, img)
    p2   = _infer_once(model, cv2.flip(img, 1))

    def _rot(deg):
        M = cv2.getRotationMatrix2D((w//2, h//2), deg, 1)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    p3 = _infer_once(model, _rot( 3))
    p4 = _infer_once(model, _rot(-3))

    mg = int(min(h, w) * 0.05)
    p5 = _infer_once(model, img[mg:h-mg, mg:w-mg])

    return float((3*p1 + 2*p2 + p3 + p4 + p5) / 8)


# ══════════════════════════════════════════════════════════════════════════════
# DECISION HELPER  (uses updated DOMAIN_THRESHOLDS from FIX 5)
# ══════════════════════════════════════════════════════════════════════════════

def _decide(prob: float, domain: str) -> dict:
    thr = DOMAIN_THRESHOLDS[domain]
    if prob >= thr["fake_thr"]:
        return dict(label="FAKE", confidence=round(prob*100, 1),
                    probability=round(prob, 4))
    elif prob <= thr["real_thr"]:
        return dict(label="REAL", confidence=round((1-prob)*100, 1),
                    probability=round(prob, 4))
    else:
        return dict(label="UNCERTAIN", confidence=round(max(prob, 1-prob)*100, 1),
                    probability=round(prob, 4))


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-FACE PIPELINE
#
# FIX 2: Stage 5 nudge strength reduced.
#   DOCUMENT/STUDIO: dark-skin nudge 0.75→0.87, light-skin 0.55→0.78.
#   Rationale: the nudge's job is to give a small push toward REAL for clean
#   images; it should not pre-compress the signal by 25–45%.
#
# FIX 6: Calibration floor lowered for DOCUMENT: 0.32→0.22.
#   The floor was designed as a safety net so a genuine fake can't be
#   calibrated down to ~0. But at 0.32 it anchors almost everything above
#   real_thr=0.28, guaranteeing UNCERTAIN. Lowered to 0.22 so genuinely
#   real calibrated images can reach ≤ real_thr=0.38 and earn REAL.
#
# FIX 7: Combined-reduction guard applied after stage 7.
#   After all calibration, if prob < raw_tta_prob * MAX_PIPELINE_REDUCTION,
#   prob is raised to raw_tta_prob * MAX_PIPELINE_REDUCTION.
#   This is the definitive anti-compounding guarantee.
# ══════════════════════════════════════════════════════════════════════════════

def _predict_face(model: EliteDetector, face_rgb: np.ndarray,
                  domain: str, use_calibration: bool,
                  full_rgb=None,
                  verbose: bool = False) -> dict:
    """
    Full 8-stage prediction pipeline on one face crop.

    Stage 0 — non-photo pre-filter
    Stage 1 — outdoor scene detection
    Stage 2 — 5-view TTA inference
    Stage 3 — skin-tone calibration        [FIX 3: threshold 0.65→0.55]
    Stage 4 — resolution tier + forensic scores
    Stage 5 — domain-aware rule nudges     [FIX 2: reduced strength]
    Stage 6 — forensic ensemble override + confidence boost  [FIX 1: suppressor removed]
    Stage 7 — domain calibration           [FIX 4: raised factors, FIX 6: lowered floor]
    Stage 7b— combined-reduction guard     [FIX 7: definitive anti-compounding]
    Stage 7c— forensic hard-floor          (unchanged from v7.0 FIX F)
    Stage 8 — final decision
    """

    # Stage 0: non-photo pre-filter
    non_photo, np_diag = is_non_photo_face(face_rgb)
    if non_photo:
        return dict(label="NOT_A_PHOTO", probability=0.0, confidence=0.0,
                    domain=domain, non_photo_diag=np_diag,
                    note="Sculpture/mask/mannequin/painting detected — not a real face.")

    # Stage 1: outdoor detection
    scene_img = full_rgb if full_rgb is not None else face_rgb
    outdoor   = is_outdoor_scene(scene_img)

    # Stage 2: TTA inference
    raw_prob = tta_predict(model, face_rgb)
    prob     = raw_prob
    if verbose:
        print(f"       [raw TTA] prob={prob:.3f}")

    # Stage 3: skin-tone calibration (FIX 3: threshold 0.65→0.55)
    lum        = _mean_face_luminance(face_rgb)
    prob, tier = calibrate_skintone(prob, face_rgb, domain)
    if verbose:
        print(f"       [skintone] lum={lum:.1f} tier={tier} prob→{prob:.3f}")

    # Stage 4: resolution tier + forensic scores
    res_img               = full_rgb if full_rgb is not None else face_rgb
    res_tier, af, rf, ct, st = _resolution_tier(res_img)

    art    = artifact_score(face_rgb)
    tex    = skin_texture_score(face_rgb)
    sym    = face_symmetry_score(face_rgb)
    eye    = eye_reflection_score(face_rgb)
    light  = illumination_score(face_rgb)
    ring   = fft_ring_score(face_rgb)
    noise  = noise_consistency_score(face_rgb)
    color  = color_coherence_score(face_rgb)
    chroma = chrominance_noise_score(face_rgb)
    jgrid  = jpeg_grid_artifact_score(face_rgb)

    # ─────────────────────────────────────────────────────────────────────
    # Stage 5: domain-aware rule nudges
    #
    # FIX 2: nudge strength reduced significantly.
    #   The nudge is meant to be a small informational tilt, not a major
    #   suppressor. With three other stages reducing probability, a strong
    #   nudge here compounds into "always UNCERTAIN".
    #   New values: dark-skin 0.87 (was 0.75), light-skin 0.78 (was 0.55).
    # ─────────────────────────────────────────────────────────────────────
    dark_skin_guard = lum < 140

    if domain in (DOMAIN_DOCUMENT, DOMAIN_STUDIO):
        if art < 0.25 and tex > 0.06 and light < 0.12:
            nudge = 0.87 if dark_skin_guard else 0.78   # FIX 2: was 0.75 / 0.55
            prob *= nudge
    elif domain == DOMAIN_OUTDOOR:
        if art < 0.30 and tex > 0.08:
            nudge = 0.90 if dark_skin_guard else 0.80   # FIX 2: was 0.80 / 0.65
            prob *= nudge
    elif domain == DOMAIN_PHONE:
        if art < 0.28 and tex > 0.05:
            nudge = 0.88 if dark_skin_guard else 0.80   # FIX 2: was 0.78 / 0.60
            prob *= nudge

    if verbose:
        print(f"       [post-nudge] prob={prob:.3f}")

    # Safety-net FAKE floor (unchanged — these are real signals)
    sym_fake_thr = 0.25 if outdoor else 0.22
    if art > 0.35 and sym > sym_fake_thr: prob = max(prob, 0.75)
    if ring  > 0.40:   prob = max(prob, 0.80)
    if eye   > 0.25:   prob = max(prob, 0.78)
    if noise > 0.70:   prob = max(prob, 0.72)
    if chroma > 0.70:  prob = max(prob, 0.74)
    if jgrid > 0.60:   prob = max(prob, 0.76)

    # Stage 6: forensic ensemble vote + confidence boost (FIX 1: suppressor removed)
    override, rv, fv = _forensic_vote(art, tex, sym, eye, ring, noise,
                                       color, chroma, jgrid, domain,
                                       af, rf, ct, st)
    prob = _boost_confidence(prob, override, domain, art, tex, rv, fv)

    if outdoor and art < 0.25 and noise < 0.20 and not override:
        prob *= 0.80

    # ─────────────────────────────────────────────────────────────────────
    # Stage 7: domain calibration (FIX 4: raised factors)
    # ─────────────────────────────────────────────────────────────────────
    if use_calibration and domain != DOMAIN_SOCIAL:
        prob = calibrate_domain(prob, face_rgb, domain)

        # FIX 6: Calibration floor — lowered so real images can reach REAL.
        #   DOCUMENT floor = 0.22 (was 0.32 in v7.0 — that was ABOVE real_thr!)
        #   Other floors similarly adjusted.
        if domain in (DOMAIN_DOCUMENT, DOMAIN_STUDIO):
            CAL_FLOOR = 0.22   # FIX 6: was 0.32 (above real_thr of 0.38)
        elif domain == DOMAIN_PHONE:
            CAL_FLOOR = 0.18
        else:
            CAL_FLOOR = 0.15
        prob = max(prob, CAL_FLOOR)

    if verbose:
        print(f"       [post-calibration] prob={prob:.3f}")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 7b: FIX 7 — combined-reduction guard
    #   The total pipeline cannot reduce prob by more than 60% from raw.
    #   This ensures the model's own signal is not completely discarded by
    #   the chain of domain/skintone corrections.
    # ─────────────────────────────────────────────────────────────────────
    min_allowed = raw_prob * MAX_PIPELINE_REDUCTION
    if prob < min_allowed:
        if verbose:
            print(f"       [FIX7 guard] prob {prob:.3f} → {min_allowed:.3f} "
                  f"(raw={raw_prob:.3f} × {MAX_PIPELINE_REDUCTION})")
        prob = min_allowed

    # ─────────────────────────────────────────────────────────────────────
    # Stage 7c: forensic hard-floor (unchanged from v7.0 FIX F)
    #   Strong forensic signals in DOCUMENT domain enforce minimum prob.
    # ─────────────────────────────────────────────────────────────────────
    if domain in (DOMAIN_DOCUMENT, DOMAIN_STUDIO, DOMAIN_PHONE):
        strong_fake_signals = sum([
            ring   > 0.30,
            noise  > 0.45,
            chroma > 0.45,
            eye    > 0.20,
            art    > 0.40,
        ])
        if strong_fake_signals >= 2:
            prob = max(prob, 0.55)

    if verbose:
        print(f"       [final] prob={prob:.3f} domain={domain}")

    # Stage 8: final decision
    result = _decide(prob, domain)
    result.update(
        domain            = domain,
        outdoor_scene     = outdoor,
        skintone_tier     = tier,
        resolution_tier   = res_tier,
        artifact_score    = round(art,    3),
        texture_score     = round(tex,    3),
        symmetry_score    = round(sym,    3),
        eye_ref_score     = round(eye,    3),
        lighting_score    = round(light,  3),
        fft_ring_score    = round(ring,   3),
        noise_consistency = round(noise,  3),
        color_coherence   = round(color,  3),
        chrominance_noise = round(chroma, 3),
        jpeg_grid_score   = round(jgrid,  3),
        forensic_override = override,
        raw_tta_prob      = round(raw_prob, 4),   # v8.0: expose raw prob for debugging
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API: predict_image  (unchanged — consumes fixed helpers above)
# ══════════════════════════════════════════════════════════════════════════════

def predict_image(model: EliteDetector, path: str,
                  use_calibration: bool = True,
                  verbose: bool = False) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {"error": f"Cannot read image: {path}"}

    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    domain = detect_domain(img)
    faces  = detect_faces(rgb)

    per_face = []

    if not faces:
        if verbose:
            print("  [info] No face detected by any detector.")

        hsv_img    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skin_mask  = cv2.inRange(hsv_img, (0, 20, 60), (25, 255, 255))
        skin_ratio = float(np.sum(skin_mask > 0)) / max(skin_mask.size, 1)

        if skin_ratio < 0.01:
            return {
                "image"       : os.path.basename(path),
                "overall"     : "NO_FACE_DETECTED",
                "avg_prob"    : 0.0,
                "total_faces" : 0,
                "domain"      : domain,
                "per_face"    : [],
                "note"        : "No human face found in this image.",
            }

        r = _predict_face(model, rgb, domain, use_calibration, rgb, verbose)
        r.update(face_id=0, bbox=None,
                 note="No face detected — full image analysed as fallback")
        per_face.append(r)

    else:
        for idx, (x, y, w, h) in enumerate(faces):
            face = crop_padded(rgb, x, y, w, h, pad=0.25)
            if not good_quality(face):
                if verbose:
                    print(f"  [face {idx}] Failed quality check — skipped.")
                continue
            if verbose:
                print(f"  [face {idx}] bbox=({x},{y},{w},{h})")
            r = _predict_face(model, face, domain, use_calibration, rgb, verbose)
            r.update(face_id=idx, bbox=(x, y, w, h))
            per_face.append(r)

        if not per_face:
            r = _predict_face(model, rgb, domain, use_calibration, rgb, verbose)
            r.update(face_id=0, bbox=None,
                     note="All face crops failed quality — full image used")
            per_face.append(r)

    labels = [f["label"] for f in per_face]

    if all(l == "NOT_A_PHOTO" for l in labels):
        overall = "NOT_A_PHOTO"
    elif all(l == "NO_FACE_DETECTED" for l in labels):
        overall = "NO_FACE_DETECTED"
    elif "FAKE" in labels:
        overall = "FAKE"
    elif all(l == "REAL" for l in labels):
        overall = "REAL"
    else:
        overall = "UNCERTAIN"

    valid_probs = [f["probability"] for f in per_face
                   if f["label"] not in ("NOT_A_PHOTO", "NO_FACE_DETECTED")]
    avg_prob    = float(np.mean(valid_probs)) if valid_probs else 0.0

    return {
        "image"      : os.path.basename(path),
        "overall"    : overall,
        "avg_prob"   : round(avg_prob, 4),
        "total_faces": len(per_face),
        "domain"     : domain,
        "per_face"   : per_face,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API: predict_batch
# ══════════════════════════════════════════════════════════════════════════════

_DOMAIN_LABELS = {
    DOMAIN_DOCUMENT: "Document/ID/Passport",
    DOMAIN_STUDIO:   "Studio",
    DOMAIN_OUTDOOR:  "Outdoor/Candid",
    DOMAIN_PHONE:    "Phone Selfie",
    DOMAIN_SOCIAL:   "Social/Webcam",
}

_LABEL_ICONS = {
    "REAL":             "✅  REAL",
    "FAKE":             "⚠️   FAKE",
    "UNCERTAIN":        "❓  UNCERTAIN",
    "NOT_A_PHOTO":      "🗿  NOT A PHOTO",
    "NO_FACE_DETECTED": "🚫  NO FACE DETECTED",
}


def predict_batch(model: EliteDetector, paths: list,
                  use_calibration: bool = True,
                  verbose: bool = False) -> list:
    SEP  = "═" * 72
    SEP2 = "─" * 72

    print(f"\n{SEP}")
    print("  DEEPFAKE DETECTOR v8.0  —  Per-image & per-face independent analysis")
    print(SEP)

    all_results = []

    for path in paths:
        print(f"\n{SEP2}")
        print(f"  IMAGE : {os.path.basename(path)}")
        print(SEP2)

        result = predict_image(model, path, use_calibration, verbose)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        if result["overall"] == "NO_FACE_DETECTED":
            print(f"\n  🚫  NO FACE DETECTED")
            print(f"       {result.get('note', 'No human face found in this image.')}")
            all_results.append(result)
            continue

        for f in result["per_face"]:
            icon = _LABEL_ICONS.get(f["label"], "?")
            bbox = f"bbox={f['bbox']}" if f.get("bbox") else "full-image-fallback"
            conf = f.get("confidence", round(f["probability"] * 100, 1))

            print(f"\n  {icon}  |  Face {f['face_id']}  |  "
                  f"confidence={conf}%  |  prob={f['probability']}"
                  f"  |  raw_tta={f.get('raw_tta_prob','?')}  |  {bbox}")

            if f["label"] == "NOT_A_PHOTO":
                d = f.get("non_photo_diag", {})
                print(f"       {f.get('note', '')}")
                print(f"       sat={d.get('sat')}  hue_div={d.get('hue_div')}  "
                      f"spec={d.get('spec')}  aniso={d.get('aniso')}  "
                      f"ch_cor={d.get('ch_cor')}  signals={d.get('signals')}")
            else:
                ov  = f.get("forensic_override") or "none"
                out = " 🌳outdoor" if f.get("outdoor_scene") else ""
                print(f"       domain={f['domain']}  skintone={f.get('skintone_tier','?')}"
                      f"  res={f.get('resolution_tier','?')}{out}")
                print(f"       artifact={f['artifact_score']}"
                      f"  texture={f['texture_score']}"
                      f"  symmetry={f['symmetry_score']}"
                      f"  eye_ref={f['eye_ref_score']}"
                      f"  lighting={f['lighting_score']}")
                print(f"       fft_ring={f['fft_ring_score']}"
                      f"  noise={f['noise_consistency']}"
                      f"  color={f['color_coherence']}"
                      f"  chroma={f['chrominance_noise']}"
                      f"  jpeg_grid={f['jpeg_grid_score']}"
                      f"  forensic_vote={ov}")
                if f.get("note"):
                    print(f"       ℹ️   {f['note']}")

        icon    = _LABEL_ICONS.get(result["overall"], "?")
        dlabel  = _DOMAIN_LABELS.get(result["domain"], result["domain"])
        outdoor = any(f.get("outdoor_scene") for f in result["per_face"])

        print(f"\n  {'─'*50}")
        print(f"  {icon}  |  OVERALL  "
              f"|  faces={result['total_faces']}"
              f"  |  avg_prob={result['avg_prob']}"
              f"  |  domain={dlabel}")

        if result["domain"] in (DOMAIN_DOCUMENT, DOMAIN_STUDIO):
            print("       📄 Document/ID photo — domain-aware calibration applied (v8.0).")
        if outdoor:
            print("       🌳 Outdoor scene — background signals suppressed, "
                  "skintone calibration applied.")

        all_results.append(result)

    if all_results:
        print(f"\n{SEP}")
        print(f"  BATCH SUMMARY  ({len(all_results)} image(s))")
        print(SEP)
        for lbl in ("REAL", "FAKE", "UNCERTAIN", "NOT_A_PHOTO", "NO_FACE_DETECTED"):
            n = sum(1 for r in all_results if r["overall"] == lbl)
            if n:
                print(f"  {_LABEL_ICONS[lbl]} : {n}")
        print(SEP)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="EliteDetector v8.0 — Uncertain-Zone Fix")
    ap.add_argument("images", nargs="+",
                    help="Image file paths to analyse")
    ap.add_argument("--model", default="elite_resnet_detector.pth",
                    help="Path to model weights")
    ap.add_argument("--no-calibration", action="store_true",
                    help="Disable domain probability calibration")
    ap.add_argument("--verbose", action="store_true",
                    help="Print intermediate calibration steps")
    args = ap.parse_args()

    print(f"\nLoading model: {args.model}")
    mdl = load_model(args.model)

    predict_batch(
        mdl,
        args.images,
        use_calibration=not args.no_calibration,
        verbose=args.verbose,
    )