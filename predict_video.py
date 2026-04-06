"""
predict_video.py  (fixed v1.1)
════════════════════════════════════════════════════════════════
Video Deepfake Detector — built on top of EliteDetector v6.1

Changes from v1.0:
  [FIX A] Import guard: is_document_photo removed (no longer exists in
          elite_predictor v6.1 — domain detection is internal).
  [FIX B] confidence field: predict_video() now returns "confidence" as
          a plain float (not "confidence_pct") so app.py render_video_verdict()
          receives the right key without a KeyError.
  [FIX C] per_frame results now include "overall" and "avg_prob" keys
          guaranteed, even when predict_image() returns NO_FACE_DETECTED,
          so the timeline renderer never crashes on missing keys.
════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
import json
import os
import time
import argparse
import tempfile
from collections import Counter
from typing import Optional

# ── import from fixed elite_predictor ────────────────────────────────────
from elite_predictor import (
    EliteDetector,
    load_model,
    predict_image,
    detect_faces,
    crop_padded,
    good_quality,
    _predict_face,
)
# NOTE: is_document_photo was removed in v6.1 — domain detection is now
# handled internally by detect_domain() inside predict_image().


# ─────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "REAL":      (34,  197,  94),
    "FAKE":      (239,  68,  68),
    "UNCERTAIN": (234, 179,   8),
}

DEFAULT_SAMPLE_FPS     = 1
DEFAULT_MAX_FRAMES     = 300
TEMPORAL_WINDOW        = 5
CONSISTENCY_THRESHOLD  = 0.60


# ─────────────────────────────────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────

def _progress(current: int, total: int, start_time: float,
              label: str = "UNKNOWN", extra: str = "") -> None:
    frac    = current / max(total, 1)
    bar_w   = 30
    filled  = int(bar_w * frac)
    bar     = "█" * filled + "░" * (bar_w - filled)
    elapsed = time.time() - start_time
    eta     = (elapsed / max(current, 1)) * (total - current)
    print(f"\r  [{bar}]  {current}/{total}  "
          f"ETA {int(eta)}s  last={label}  {extra}",
          end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str,
                   sample_fps: float = DEFAULT_SAMPLE_FPS,
                   max_frames: int   = DEFAULT_MAX_FRAMES
                   ) -> tuple:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / source_fps

    interval = max(1, int(source_fps / sample_fps))
    expected = min(max_frames, total_frames // interval)

    meta = {
        "source_fps"    : round(source_fps, 2),
        "sample_fps"    : sample_fps,
        "total_frames"  : total_frames,
        "duration_s"    : round(duration_s, 2),
        "width"         : width,
        "height"        : height,
        "frame_interval": interval,
        "frames_analysed": expected,
    }

    print(f"\n  📹  Video info:")
    print(f"      Resolution : {width}×{height}")
    print(f"      Duration   : {duration_s:.1f}s  ({source_fps:.1f} fps)")
    print(f"      Sampling   : every {interval} frames  "
          f"→ ~{expected} frames to analyse")

    frames    = []
    frame_idx = 0
    sampled   = 0

    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sampled += 1
        frame_idx += 1

    cap.release()
    meta["frames_analysed"] = len(frames)
    print(f"      Extracted  : {len(frames)} frames")
    return frames, meta


# ─────────────────────────────────────────────────────────────────────────
# ANNOTATED VIDEO WRITER
# ─────────────────────────────────────────────────────────────────────────

def _draw_overlay(frame_rgb: np.ndarray,
                  frame_result: dict,
                  frame_no: int,
                  timestamp: float) -> np.ndarray:
    out = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = out.shape[:2]

    for f in frame_result.get("per_face", []):
        label  = f.get("label", "UNCERTAIN")
        color  = LABEL_COLORS.get(label, (234, 179, 8))[::-1]
        conf   = f.get("confidence", round(f.get("probability", 0.5) * 100, 1))
        bbox   = f.get("bbox")

        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            text = f"{label} {conf:.0f}%"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out,
                          (x, y - th - 8), (x + tw + 6, y), color, -1)
            cv2.putText(out, text, (x + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    overall = frame_result.get("overall", "UNCERTAIN")
    banner_color = LABEL_COLORS.get(overall, (234, 179, 8))[::-1]
    banner_text  = f"Frame {frame_no}  {timestamp:.1f}s  →  {overall}"
    cv2.rectangle(out, (0, 0), (w, 32), banner_color, -1)
    cv2.putText(out, banner_text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    return out


def write_annotated_video(video_path: str,
                          frames: list,
                          frame_results: list,
                          frame_indices: list,
                          source_fps: float,
                          output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠️  Cannot re-open {video_path} for annotation writing.")
        return

    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS) or source_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    idx_to_result = {
        fi: (frame_results[i], frames[i])
        for i, fi in enumerate(frame_indices)
    }

    frame_no = 0
    print(f"\n  ✍️  Writing annotated video → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no in idx_to_result:
            result, rgb_frame = idx_to_result[frame_no]
            ts = frame_no / max(fps, 1)
            annotated = _draw_overlay(rgb_frame, result, frame_no, ts)
            writer.write(annotated)
        else:
            writer.write(frame)

        frame_no += 1

    cap.release()
    writer.release()
    print(f"  ✅  Annotated video saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────
# TEMPORAL CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────

def analyse_temporal_consistency(frame_results: list) -> dict:
    labels      = [r.get("overall", "UNCERTAIN") for r in frame_results]
    transitions = [i for i in range(1, len(labels))
                   if labels[i] != labels[i-1]]

    consistent_frames = 0
    for i in range(len(labels)):
        window = labels[max(0, i - TEMPORAL_WINDOW) : i + 1]
        cnt    = Counter(window)
        if cnt.most_common(1)[0][1] / len(window) >= CONSISTENCY_THRESHOLD:
            consistent_frames += 1

    dominant = Counter(labels).most_common(1)[0][0] if labels else "UNCERTAIN"

    return {
        "dominant_label"  : dominant,
        "transitions"     : transitions,
        "transition_count": len(transitions),
        "consistent_pct"  : round(
            100 * consistent_frames / max(len(labels), 1), 1),
    }


# ─────────────────────────────────────────────────────────────────────────
# WEIGHTED MAJORITY VOTING
# ─────────────────────────────────────────────────────────────────────────

def _aggregate_verdict(frame_results: list) -> dict:
    fake_score = 0.0
    real_score = 0.0
    unc_score  = 0.0

    for r in frame_results:
        p     = r.get("avg_prob", 0.5)
        label = r.get("overall", "UNCERTAIN")

        if label == "FAKE":
            fake_score += p
        elif label == "REAL":
            real_score += (1.0 - p)
        else:
            unc_score  += 0.5

    total   = fake_score + real_score + unc_score + 1e-9
    f_frac  = fake_score / total
    r_frac  = real_score / total

    counts  = Counter(r.get("overall", "UNCERTAIN") for r in frame_results)

    if f_frac >= 0.45:
        verdict    = "FAKE"
        confidence = round(f_frac * 100, 1)
    elif r_frac >= 0.55:
        verdict    = "REAL"
        confidence = round(r_frac * 100, 1)
    else:
        verdict    = "UNCERTAIN"
        confidence = round(max(f_frac, r_frac) * 100, 1)

    return {
        "verdict"          : verdict,
        "confidence_pct"   : confidence,
        "fake_score_frac"  : round(f_frac, 4),
        "real_score_frac"  : round(r_frac, 4),
        "frame_counts"     : dict(counts),
        "total_frames_used": len(frame_results),
    }


# ─────────────────────────────────────────────────────────────────────────
# MAIN VIDEO PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────

def predict_video(model: EliteDetector,
                  video_path: str,
                  sample_fps: float           = DEFAULT_SAMPLE_FPS,
                  max_frames: int             = DEFAULT_MAX_FRAMES,
                  use_calibration: bool       = True,
                  output_video: Optional[str] = None,
                  report_path: Optional[str]  = None) -> dict:
    """
    Full video deepfake prediction pipeline.

    Returns a dict with keys:
      video, verdict, confidence, aggregate, temporal, meta, per_frame
    """
    SEP  = "=" * 68
    SEP2 = "─" * 68

    print(f"\n{SEP}")
    print("  VIDEO DEEPFAKE DETECTOR  v1.1")
    print(f"  File : {os.path.basename(video_path)}")
    print(SEP)

    if not os.path.isfile(video_path):
        return {"error": f"File not found: {video_path}"}

    # 1. Extract frames
    frames, meta = extract_frames(video_path, sample_fps, max_frames)
    if not frames:
        return {"error": "No frames could be extracted from video."}

    source_fps  = meta["source_fps"]
    interval    = meta["frame_interval"]
    frame_indices = [i * interval for i in range(len(frames))]

    # 2. Per-frame prediction
    print(f"\n  🔍  Analysing {len(frames)} frames...")
    print(SEP2)

    frame_results = []
    start_time    = time.time()

    for idx, (frame_rgb, orig_idx) in enumerate(zip(frames, frame_indices)):
        timestamp = orig_idx / max(source_fps, 1)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        cv2.imwrite(tmp_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        result = predict_image(model, tmp_path, use_calibration)
        os.unlink(tmp_path)

        if "error" not in result:
            result["frame_index"] = orig_idx
            result["timestamp_s"] = round(timestamp, 2)

            # FIX C: guarantee overall and avg_prob keys even for edge cases
            result.setdefault("overall",   "UNCERTAIN")
            result.setdefault("avg_prob",  0.5)

            frame_results.append(result)
            label = result["overall"]
        else:
            label = "ERROR"

        _progress(idx + 1, len(frames), start_time,
                  label=label, extra=f"t={timestamp:.1f}s")

    print()

    if not frame_results:
        return {"error": "All frames failed prediction."}

    # 3. Aggregate
    aggregate = _aggregate_verdict(frame_results)
    temporal  = analyse_temporal_consistency(frame_results)

    # 4. Per-frame summary
    print(f"\n{SEP2}")
    print("  PER-FRAME RESULTS (summary)")
    print(SEP2)

    for r in frame_results:
        sym   = {"REAL":"✅","FAKE":"⚠️","UNCERTAIN":"❓"}.get(r["overall"], "?")
        faces = r.get("total_faces", 0)
        print(f"  {sym}  t={r['timestamp_s']:6.1f}s  "
              f"frame={r['frame_index']:5d}  "
              f"→  {r['overall']:9s}  "
              f"prob={r['avg_prob']:.3f}  "
              f"faces={faces}")

    # 5. Final verdict
    verdict  = aggregate["verdict"]
    # FIX B: expose as "confidence" (plain float) not "confidence_pct"
    # app.py render_video_verdict() reads result["confidence"]
    confidence = aggregate["confidence_pct"]

    sym      = {"REAL":"✅","FAKE":"⚠️","UNCERTAIN":"❓"}.get(verdict, "?")
    dom      = temporal["dominant_label"]
    trans    = temporal["transition_count"]
    cons_pct = temporal["consistent_pct"]
    fcounts  = aggregate["frame_counts"]

    print(f"\n{SEP}")
    print("  FINAL VIDEO VERDICT")
    print(SEP)
    print(f"\n  {sym}  VERDICT     : {verdict}  ({confidence}% confidence)")
    print(f"\n  Frame breakdown:")
    for lbl, sym2 in [("REAL","✅"),("FAKE","⚠️"),("UNCERTAIN","❓")]:
        n = fcounts.get(lbl, 0)
        bar = "█" * n + " " * max(0, 30 - n)
        print(f"    {sym2}  {lbl:9s} : {n:4d} frames  {bar}")

    print(f"\n  Temporal analysis:")
    print(f"    Dominant label      : {dom}")
    print(f"    Label transitions   : {trans}")
    print(f"    Consistent frames   : {cons_pct}%")

    print(f"\n  Video metadata:")
    print(f"    Duration            : {meta['duration_s']}s")
    print(f"    Resolution          : {meta['width']}×{meta['height']}")
    print(f"    Source FPS          : {meta['source_fps']}")
    print(f"    Frames analysed     : {meta['frames_analysed']}")
    print(SEP)

    # 6. Annotated video (optional)
    if output_video:
        write_annotated_video(
            video_path, frames, frame_results,
            frame_indices, source_fps, output_video)

    # 7. Build return dict
    # FIX B: "confidence" key (not "confidence_pct") for app.py compatibility
    full_result = {
        "video"      : os.path.basename(video_path),
        "verdict"    : verdict,
        "confidence" : confidence,   # FIX B: was confidence_pct in aggregate
        "aggregate"  : aggregate,
        "temporal"   : temporal,
        "meta"       : meta,
        "per_frame"  : frame_results,
    }

    if report_path:
        with open(report_path, "w") as f:
            json.dump(full_result, f, indent=2, default=str)
        print(f"\n  📄  JSON report saved: {report_path}")

    return full_result


# ─────────────────────────────────────────────────────────────────────────
# BATCH VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────

def predict_video_batch(model: EliteDetector,
                        video_paths: list,
                        sample_fps: float     = DEFAULT_SAMPLE_FPS,
                        max_frames: int       = DEFAULT_MAX_FRAMES,
                        use_calibration: bool = True,
                        output_dir: Optional[str] = None) -> list:
    SEP = "=" * 68
    all_results = []

    for vpath in video_paths:
        basename    = os.path.splitext(os.path.basename(vpath))[0]
        out_video   = None
        report_path = None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_video   = os.path.join(output_dir, f"{basename}_annotated.mp4")
            report_path = os.path.join(output_dir, f"{basename}_report.json")

        r = predict_video(
            model, vpath,
            sample_fps      = sample_fps,
            max_frames      = max_frames,
            use_calibration = use_calibration,
            output_video    = out_video,
            report_path     = report_path,
        )
        all_results.append(r)

    print(f"\n{SEP}")
    print(f"  BATCH VIDEO SUMMARY  ({len(all_results)} videos)")
    print(SEP)
    for lbl, sym in [("REAL","✅"),("FAKE","⚠️"),("UNCERTAIN","❓")]:
        n = sum(1 for r in all_results if r.get("verdict") == lbl)
        print(f"  {sym}  {lbl} : {n} video(s)")
    print(SEP)

    return all_results


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Video deepfake detector v1.1 — frame-level + video-level verdict")

    ap.add_argument("videos", nargs="+",
                    help="One or more video file paths to analyse")
    ap.add_argument("--model", default="elite_resnet_detector.pth",
                    help="Path to EliteDetector weights (.pth)")
    ap.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS,
                    help=f"Frames per second to sample (default {DEFAULT_SAMPLE_FPS})")
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                    help=f"Max frames to analyse per video (default {DEFAULT_MAX_FRAMES})")
    ap.add_argument("--no-calibration", action="store_true",
                    help="Disable document-photo probability calibration")
    ap.add_argument("--output-video", default=None,
                    help="Path for annotated output video (single input only)")
    ap.add_argument("--report", default=None,
                    help="Path to save JSON report (single input only)")
    ap.add_argument("--output-dir", default=None,
                    help="Directory for annotated videos + reports (batch mode)")

    args = ap.parse_args()

    print(f"\n🚀  Loading model from: {args.model}")
    model = load_model(args.model)

    if len(args.videos) == 1:
        predict_video(
            model,
            args.videos[0],
            sample_fps      = args.sample_fps,
            max_frames      = args.max_frames,
            use_calibration = not args.no_calibration,
            output_video    = args.output_video,
            report_path     = args.report,
        )
    else:
        predict_video_batch(
            model,
            args.videos,
            sample_fps      = args.sample_fps,
            max_frames      = args.max_frames,
            use_calibration = not args.no_calibration,
            output_dir      = args.output_dir,
        )