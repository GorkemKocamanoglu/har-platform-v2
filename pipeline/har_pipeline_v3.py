"""
har_pipeline.py — the labeling pipeline as importable functions.
Same logic as HAR_Pipeline_v2.ipynb, refactored for batch processing.
"""
import os, json, time, glob
import numpy as np
import pandas as pd

# ----------------------------- tunables (same defaults as the notebook) ----
WINDOW_MS       = 2000
STEP_MS         = 1000
ACC_STD_MIN     = 0.07
FREQ_MATCH_HZ   = 0.65
RUN_FREQ_HZ     = 2.4
RUN_ACC_STD     = 0.60
MIN_SEGMENT_S   = 5.0
MAX_VLM_CHUNK_S = 25.0
MODEL_NAME      = "gemini-2.5-flash"

# ----------------------------- sync ---------------------------------------
def get_video_duration_ms(video_path):
    from pymediainfo import MediaInfo
    for tr in MediaInfo.parse(video_path).tracks:
        if tr.track_type == "General" and tr.duration:
            return int(tr.duration)
    return None

def get_video_meta_start_ms(video_path):
    """Start time claimed by the video metadata (often WRONG by many seconds)."""
    import datetime, calendar
    from pymediainfo import MediaInfo
    for tr in MediaInfo.parse(video_path).tracks:
        if tr.track_type == "General" and tr.encoded_date:
            clean = tr.encoded_date.replace("UTC", "").strip()
            dt = datetime.datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")
            end_ms = calendar.timegm(dt.timetuple()) * 1000
            dur = int(tr.duration) if tr.duration else 0
            return end_ms - dur
    return None

def synchronize(sensor_csv, video_path, out_csv, log=print):
    """align_start sync: video start pinned to sensor start."""
    v_dur = get_video_duration_ms(video_path)
    if v_dur is None:
        raise ValueError("no duration in video metadata")
    df = pd.read_csv(sensor_csv)
    s_start, s_end = df.Timestamp_ms.iloc[0], df.Timestamp_ms.iloc[-1]
    v_start, v_end = s_start, s_start + v_dur
    log(f"  video {v_dur/1000:.1f}s, sensors {(s_end-s_start)/1000:.1f}s, "
        f"trim tail {max(0,(s_end-v_end)/1000):.1f}s")
    out = df[(df.Timestamp_ms >= v_start) & (df.Timestamp_ms <= v_end)].copy()
    if out.empty:
        raise ValueError("no overlap after trim")
    out.to_csv(out_csv, index=False)
    return v_start

# ----------------------------- features ------------------------------------
def dominant_freq(sig, ts_ms):
    if len(sig) < 8: return 0.0, 0.0
    t = (ts_ms - ts_ms[0]) / 1000.0
    fs = len(t) / max(t[-1], 1e-6)
    x = sig - sig.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    band = (freqs >= 0.8) & (freqs <= 5.0)
    if not band.any(): return 0.0, 0.0
    bi = np.where(band)[0]
    i = bi[np.argmax(spec[bi])]
    if 0 < i < len(spec) - 1 and spec[i] > 0:      # parabolic sub-bin interpolation
        a, b, c = spec[i-1], spec[i], spec[i+1]
        denom = a - 2*b + c
        delta = 0.5*(a-c)/denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    f_peak = float(freqs[i] + delta*(freqs[1]-freqs[0]))
    return f_peak, float(spec[i] / (spec[1:].sum() + 1e-9))

def device_tag(name):
    n = str(name).lower()
    if "left" in n:  return "L"
    if "right" in n: return "R"
    raise ValueError(f"device '{name}' has neither Left nor Right in its name")

def build_windows(synced_csv, log=print):
    df = pd.read_csv(synced_csv)
    df["Device_Name"] = df["Device_Name"].astype(str).str.strip()
    tags = {d: device_tag(d) for d in df.Device_Name.unique()}
    streams = {tags[d]: g.reset_index(drop=True) for d, g in df.groupby("Device_Name")}
    active = sorted(streams)
    if len(active) == 1:
        log(f"  WARNING: single wrist only ({list(tags)})")
    t0, t1 = df.Timestamp_ms.iloc[0], df.Timestamp_ms.iloc[-1]
    rows, ws, wid = [], t0, 0
    while ws + WINDOW_MS <= t1:
        r = {"Window_ID": wid, "Start_ms": int(ws), "End_ms": int(ws + WINDOW_MS)}
        ok = True
        for tag, s in streams.items():
            w = s[(s.Timestamp_ms >= ws) & (s.Timestamp_ms < ws + WINDOW_MS)]
            if len(w) < 8: ok = False; break
            acc  = np.sqrt(w.AccX**2 + w.AccY**2 + w.AccZ**2).values
            gyro = np.sqrt(w.GyroX**2 + w.GyroY**2 + w.GyroZ**2).values
            f, p = dominant_freq(acc, w.Timestamp_ms.values)
            r.update({f"{tag}_acc_std": acc.std(), f"{tag}_gyro_mean": gyro.mean(),
                      f"{tag}_dom_freq": f, f"{tag}_peak_power": p})
        if ok:
            rows.append(r); wid += 1
        ws += STEP_MS
    return pd.DataFrame(rows), streams, active

# ----------------------------- motion classification -----------------------
def classify_windows(W, active_tags):
    def freqs_agree(f1, f2, tol=FREQ_MATCH_HZ):
        if abs(f1 - f2) < tol: return True
        return abs(2*f1 - f2) < 1.5*tol or abs(f1 - 2*f2) < 1.5*tol

    def one(r):
        stds  = [r[f"{t}_acc_std"]  for t in active_tags]
        freqs = [r[f"{t}_dom_freq"] for t in active_tags]
        energetic = all(s > ACC_STD_MIN for s in stds)
        in_band   = all(f > 0.9 for f in freqs)
        if len(active_tags) == 2:
            cad = in_band and freqs_agree(freqs[0], freqs[1])
        else:
            cad = in_band and r[f"{active_tags[0]}_peak_power"] > 0.10
        if energetic and cad:
            if len(freqs) == 2 and abs(freqs[0]-freqs[1]) >= FREQ_MATCH_HZ:
                f = max(freqs)
            else:
                f = sum(freqs)/len(freqs)
            e = max(stds)
            return "Running" if (f >= RUN_FREQ_HZ or e >= RUN_ACC_STD) else "Walking"
        return "Idle"

    W = W.copy()
    W["Motion"] = W.apply(one, axis=1)
    # 3-window mode smoothing
    lab = W["Motion"].tolist()
    for i in range(1, len(lab)-1):
        seg = lab[i-1:i+2]
        lab[i] = max(set(seg), key=seg.count)
    W["Motion"] = lab
    return W

# ----------------------------- min-duration + segments ---------------------
def _runs(labels):
    out, s = [], 0
    for i in range(1, len(labels)+1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append([s, i-1, labels[s]]); s = i
    return out

def enforce_min_duration(W, min_s=MIN_SEGMENT_S):
    labels = W["Motion"].tolist()
    dur = lambda r: (W.End_ms.iloc[r[1]] - W.Start_ms.iloc[r[0]]) / 1000.0
    while True:
        rr = _runs(labels)
        if len(rr) <= 1: break
        short = [r for r in rr if dur(r) < min_s]
        if not short: break
        r = min(short, key=dur); i = rr.index(r)
        prev = rr[i-1] if i > 0 else None
        nxt  = rr[i+1] if i < len(rr)-1 else None
        if prev and nxt and prev[2] == nxt[2]: new = prev[2]
        elif prev and (nxt is None or dur(prev) >= dur(nxt)): new = prev[2]
        else: new = nxt[2]
        for k in range(r[0], r[1]+1): labels[k] = new
    W = W.copy(); W["Motion"] = labels
    return W

def build_segments(W):
    segs, cur = [], None
    for _, r in W.iterrows():
        if cur is None or r.Motion != cur["Motion"]:
            if cur: segs.append(cur)
            cur = dict(Motion=r.Motion, Start_ms=int(r.Start_ms), End_ms=int(r.End_ms),
                       first_w=int(r.Window_ID), last_w=int(r.Window_ID))
        else:
            cur["End_ms"] = int(r.End_ms); cur["last_w"] = int(r.Window_ID)
    if cur: segs.append(cur)
    return pd.DataFrame(segs)

def build_chunks(seg_df):
    chunks = []
    for si, seg in seg_df.iterrows():
        dur = seg.End_ms - seg.Start_ms
        n = max(1, int(np.ceil(dur / (MAX_VLM_CHUNK_S*1000))))
        edges = np.linspace(seg.Start_ms, seg.End_ms, n+1)
        for k in range(n):
            chunks.append(dict(seg_idx=si, Motion=seg.Motion,
                               Start_ms=int(edges[k]), End_ms=int(edges[k+1])))
    return pd.DataFrame(chunks)

# ----------------------------- frames ---------------------------------------
def extract_chunk_frames(chunk_df, video_path, video_start_ms, frames_dir, log=print):
    import cv2
    os.makedirs(frames_dir, exist_ok=True)
    jobs = []
    for ci, ch in chunk_df.iterrows():
        dur = ch.End_ms - ch.Start_ms
        fracs = [0.5] if dur < 3000 else [0.2, 0.5, 0.8]
        for fi, frac in enumerate(fracs):
            jobs.append(((ch.Start_ms + frac*dur) - video_start_ms, ci, fi))
    jobs.sort()
    cap = cv2.VideoCapture(video_path)
    chunk_frames = {ci: [] for ci in chunk_df.index}
    for t_ms, ci, fi in jobs:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t_ms))
        ok, frame = cap.read()
        if ok:
            name = f"chunk_{ci:03d}_f{fi}.jpg"
            cv2.imwrite(os.path.join(frames_dir, name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            chunk_frames[ci].append(name)
    cap.release()
    n = sum(len(v) for v in chunk_frames.values())
    log(f"  {n} frames for {len(chunk_df)} chunks")
    return chunk_frames

# ----------------------------- VLM ------------------------------------------
MOTION_HINT = {
    "Idle":    "Motion sensors show the wearer is STATIONARY (not walking or running).",
    "Walking": "Motion sensors show the wearer is WALKING.",
    "Running": "Motion sensors show the wearer is RUNNING.",
}

def _build_prompt(motion):
    return f"""These photos span one activity segment from a FIRST-PERSON wearable camera (smart glasses).

{MOTION_HINT[motion]}

Describe what the CAMERA WEARER is doing. Ignore all other people visible in the
photos — never describe their actions, only the wearer's.

Return JSON with exactly these fields:
- "context": the wearer's location/scenario, 1-2 lowercase words
  (examples: kitchen, office, train, bus, street, supermarket, living room, stairs, car)
- "action": what the wearer is doing, a short lowercase verb phrase of 1-4 words
  (examples: chopping vegetables, typing on laptop, scrolling phone, washing dishes,
   carrying bag, looking around)"""

def label_chunks_vlm(chunk_df, chunk_frames, frames_dir, api_key, log=print,
                     max_workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google import genai
    from google.genai import types
    from PIL import Image

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema={"type": "OBJECT",
                         "properties": {"context": {"type": "STRING"},
                                        "action":  {"type": "STRING"}},
                         "required": ["context", "action"]})

    def one(ci, motion, files):
        imgs = [Image.open(os.path.join(frames_dir, f)) for f in files]
        delay = 2.0
        for _ in range(5):
            try:
                resp = client.models.generate_content(
                    model=MODEL_NAME, contents=[_build_prompt(motion), *imgs], config=cfg)
                text = (resp.text or "").strip()
                if text:
                    d = json.loads(text)
                    return ci, d.get("context","unknown").strip().lower(), \
                               d.get("action","unknown").strip().lower()
                time.sleep(delay); delay *= 2
            except json.JSONDecodeError:
                time.sleep(delay); delay *= 2
            except Exception as e:
                if any(k in str(e) for k in ("429", "RESOURCE_EXHAUSTED", "503")):
                    time.sleep(delay); delay *= 2
                else:
                    log(f"  chunk {ci}: {e}"); return ci, "unknown", "unknown"
        return ci, "unknown", "unknown"

    results = {}
    todo = [(ci, ch.Motion, chunk_frames[ci]) for ci, ch in chunk_df.iterrows()
            if chunk_frames.get(ci)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(one, *t) for t in todo]
        for fut in as_completed(futs):
            ci, ctx, act = fut.result()
            results[ci] = (ctx, act)
    chunk_df = chunk_df.copy()
    chunk_df["Context"] = chunk_df.index.map(lambda i: results.get(i, ("unknown","unknown"))[0])
    chunk_df["Action"]  = chunk_df.index.map(lambda i: results.get(i, ("unknown","unknown"))[1])
    return chunk_df

# ----------------------------- assembly + taxonomy --------------------------
def assemble_windows(W, chunk_df):
    final = W[["Window_ID","Start_ms","End_ms","Motion"]].copy()
    final["Context"] = "unknown"; final["Action"] = "unknown"
    mid = (final.Start_ms + final.End_ms) / 2
    for ci, ch in chunk_df.iterrows():
        m = (mid >= ch.Start_ms) & (mid < ch.End_ms)
        final.loc[m, "Context"] = ch.Context
        final.loc[m, "Action"]  = ch.Action
    known = chunk_df[chunk_df.Context != "unknown"]
    for ci, ch in chunk_df[chunk_df.Context == "unknown"].iterrows():
        cand = known[known.Motion == ch.Motion]
        if cand.empty: cand = known
        if cand.empty: break
        nearest = cand.iloc[(cand.Start_ms - ch.Start_ms).abs().argmin()]
        m = (mid >= ch.Start_ms) & (mid < ch.End_ms)
        final.loc[m, ["Context","Action"]] = [nearest.Context, nearest.Action]
    return final

DEFAULT_TAXONOMY = [
    ("Running",     lambda r: r.Motion == "Running"),
    ("Walking",     lambda r: r.Motion == "Walking"),
    ("Cooking",     lambda r: "kitchen" in r.Context or
                              any(w in r.Action for w in ["cook","chop","stir","fry","wash dish"])),
    ("Using train", lambda r: any(w in r.Context for w in ["train","metro","subway","tram"])),
    ("Using bus",   lambda r: "bus" in r.Context),
    ("Cycling",     lambda r: any(w in r.Action for w in ["cycl","bik"]) or "bike" in r.Context),
    ("Desk work",   lambda r: any(w in r.Action for w in ["typ","comput","writ","laptop"])),
    ("Phone use",   lambda r: "phone" in r.Action),
    ("Stationary",  lambda r: r.Motion == "Idle"),
]

def apply_taxonomy(final, taxonomy=None):
    taxonomy = taxonomy or DEFAULT_TAXONOMY
    def one(r):
        for name, rule in taxonomy:
            try:
                if rule(r): return name
            except Exception:
                pass
        return "Other"
    final = final.copy()
    final["Label"] = final.apply(one, axis=1)
    return final

# ----------------------------- one call does a whole session ----------------
def process_session(session_dir, api_key, taxonomy=None, force=False, log=print):
    """Run the full pipeline on one session folder containing sensors.csv + video.
    Writes synced_data.csv, windows_structured.csv, windows_labeled_final.csv there.
    Skips if outputs already exist (unless force=True)."""
    out_final = os.path.join(session_dir, "windows_labeled_final.csv")
    if os.path.exists(out_final) and not force:
        log(f"  already processed — skipping (force=True to redo)")
        return None

    sensor = next((p for n in ["sensors.csv", "sensor_data.csv"]
                   if os.path.exists(p := os.path.join(session_dir, n))), None)
    videos = [p for p in glob.glob(os.path.join(session_dir, "*"))
              if p.lower().endswith((".mp4", ".mov"))]
    if sensor is None or not videos:
        raise FileNotFoundError(f"need a sensor csv and a video in {session_dir}")
    video = videos[0]

    synced = os.path.join(session_dir, "synced_data.csv")
    v_start = synchronize(sensor, video, synced, log=log)

    W, streams, active = build_windows(synced, log=log)
    W = classify_windows(W, active)
    W = enforce_min_duration(W)
    seg_df = build_segments(W)
    chunk_df = build_chunks(seg_df)
    log(f"  {len(W)} windows -> {len(seg_df)} segments -> {len(chunk_df)} VLM chunks")

    frames_dir = os.path.join(session_dir, "frames")
    chunk_frames = extract_chunk_frames(chunk_df, video, v_start, frames_dir, log=log)
    chunk_df = label_chunks_vlm(chunk_df, chunk_frames, frames_dir, api_key, log=log)

    final = assemble_windows(W, chunk_df)
    final.to_csv(os.path.join(session_dir, "windows_structured.csv"), index=False)
    final = apply_taxonomy(final, taxonomy)
    final.to_csv(out_final, index=False)
    log("  labels: " + ", ".join(f"{k}:{v}" for k, v in
                                 final.Label.value_counts().items()))
    return final
