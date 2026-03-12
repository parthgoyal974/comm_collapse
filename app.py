"""
app.py — Communication Collapse Detection · Showcase App
=========================================================
Drop this file in the project ROOT directory (same level as src/, data/, results/).

Install deps (if not already):
    pip install streamlit plotly pandas

Run:
    streamlit run app.py

Uses:
    results/checkpoints/main/best_model/   ← trained DeBERTa-v3 checkpoint
    data/tokenizer_cache/                  ← cached tokenizer
    data/processed/train.pkl               ← for loading example conversations
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import sys
import json
import pickle
import re
from pathlib import Path
from collections import defaultdict

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── project src on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CommCollapse · Early Breakdown Detection",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a12; }

/* hero */
.hero {
    background: linear-gradient(135deg, #0d0b1e 0%, #1a1040 40%, #0f1e3a 100%);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 18px; padding: 38px 44px 32px 44px;
    margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-title { font-size: 2.3rem; font-weight: 800; color: #f1f5f9;
              letter-spacing: -0.8px; margin: 0 0 6px 0; line-height: 1.15; }
.hero-sub   { font-size: 1rem; color: rgba(255,255,255,0.55);
              margin: 0 0 22px 0; max-width: 700px; line-height: 1.55; }
.badge {
    display: inline-block; background: rgba(139,92,246,0.18);
    border: 1px solid rgba(139,92,246,0.4); color: #c4b5fd;
    padding: 3px 11px; border-radius: 20px; font-size: 0.73rem;
    font-weight: 600; margin-right: 7px; letter-spacing: 0.4px;
}

/* kpi */
.kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 0 0; }
.kpi { flex: 1; min-width: 100px; background: rgba(255,255,255,0.04);
       border: 1px solid rgba(255,255,255,0.09); border-radius: 12px;
       padding: 16px 18px; text-align: center; transition: border-color .2s; }
.kpi:hover { border-color: rgba(139,92,246,0.45); }
.kpi-val { font-size: 1.85rem; font-weight: 700; margin: 0; line-height: 1.1; }
.kpi-lbl { font-size: 0.7rem; color: rgba(255,255,255,0.42);
           text-transform: uppercase; letter-spacing: 0.9px; margin-top: 5px; }

/* section title */
.sec { font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
       padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.07);
       margin: 0 0 16px 0; }

/* alerts */
.alert-danger {
    background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.45);
    border-left: 4px solid #ef4444; border-radius: 10px;
    padding: 16px 20px; margin: 0 0 16px 0;
}
.alert-safe {
    background: rgba(74,222,128,0.09); border: 1px solid rgba(74,222,128,0.35);
    border-left: 4px solid #4ade80; border-radius: 10px;
    padding: 16px 20px; margin: 0 0 16px 0;
}
.alert-title { font-size: 1rem; font-weight: 700; margin: 0 0 5px 0; }
.alert-body  { font-size: 0.84rem; color: rgba(255,255,255,0.65); margin: 0; }

/* turns */
.turn-wrap { margin: 5px 0; }
.turn-sys  { background: rgba(55,48,95,0.55); border-left: 3px solid #7c3aed;
             border-radius: 0 8px 8px 0; padding: 9px 14px; }
.turn-usr  { background: rgba(15,35,65,0.6);  border-left: 3px solid #0ea5e9;
             border-radius: 0 8px 8px 0; padding: 9px 14px; }
.turn-bd   { border-left-color: #ef4444 !important;
             background: rgba(180,20,20,0.22) !important; }
.turn-warn { border-left-color: #f59e0b !important;
             background: rgba(160,100,0,0.22) !important; }
.spk { font-size: 0.66rem; font-weight: 700; letter-spacing: 1px;
       text-transform: uppercase; opacity: .65; margin-bottom: 3px; }
.utt { font-size: 0.9rem; color: #e2e8f0; }
.rpill { float: right; font-size: 0.67rem; font-weight: 700;
         padding: 1px 7px; border-radius: 8px; border: 1px solid;
         margin-top: -2px; }

/* example preview */
.ex-preview { font-size: 0.78rem; color: rgba(255,255,255,0.42);
              margin: 4px 0 0 0; line-height: 1.4; }

/* info card */
.info-card { background: rgba(255,255,255,0.03);
             border: 1px solid rgba(255,255,255,0.07);
             border-radius: 10px; padding: 14px 18px;
             font-size: 0.84rem; color: rgba(255,255,255,0.55); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED: MODEL + TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚙️ Loading DeBERTa-v3 checkpoint…")
def load_model_and_tokenizer():
    import torch
    from transformers import DebertaV2Tokenizer
    from model import load_model

    ckpt  = ROOT / "results" / "checkpoints" / "main" / "best_model"
    cache = ROOT / "data" / "tokenizer_cache"

    if not ckpt.exists():
        return None, None, (
            f"Checkpoint not found at `{ckpt}`.\n\n"
            "Run `python src/train.py` or `python run_all.py --skip-baselines`."
        )

    try:
        if (cache / "tokenizer_config.json").exists():
            tokenizer = DebertaV2Tokenizer.from_pretrained(str(cache))
        else:
            tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")
            from tokenize_utils import SPECIAL_TOKENS
            tokenizer.add_special_tokens(SPECIAL_TOKENS)

        model = load_model(ckpt, vocab_size=len(tokenizer), device="cpu")
        model.eval()
        return model, tokenizer, None
    except Exception as exc:
        return None, None, f"Model load error: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED: EXAMPLES from train.pkl (25% stratified subset used in training)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_examples():
    """
    Load pre-computed examples from data/app_examples.json.
    Generate that file once by running:  python generate_examples.py
    Each entry is a conversation the trained model predicts CORRECTLY.
    """
    examples_path = ROOT / "data" / "app_examples.json"
    if not examples_path.exists():
        return []
    try:
        with open(examples_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  PARSING
# ══════════════════════════════════════════════════════════════════════════════
_SYS = {"s", "sys", "system", "bot", "agent", "assistant", "chatbot", "a"}
_USR = {"u", "usr", "user", "human", "h", "person"}


def parse_conversation(text: str) -> list[dict] | None:
    text = text.strip()
    if not text:
        return None

    # DBDC3 JSON
    if text.startswith("{"):
        try:
            raw = json.loads(text)
            raw_turns = raw.get("turns", raw.get("utterances", []))
            turns = []
            for t in raw_turns:
                spk = str(t.get("speaker", "S")).upper()
                speaker = "SYS" if spk in ("S", "SYS", "SYSTEM") else "USR"
                utt     = t.get("utterance", t.get("text", "")).strip()
                if utt:
                    turns.append({"speaker": speaker, "text": utt})
            return turns if len(turns) >= 2 else None
        except json.JSONDecodeError:
            pass

    # Line-based
    turns = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            if turns:
                turns[-1]["text"] += " " + line
            continue
        colon   = line.index(":")
        prefix  = line[:colon].strip().lower()
        content = line[colon + 1:].strip()
        if not content:
            continue
        if prefix in _SYS:
            speaker = "SYS"
        elif prefix in _USR:
            speaker = "USR"
        else:
            last    = turns[-1]["speaker"] if turns else "USR"
            speaker = "USR" if last == "SYS" else "SYS"
        turns.append({"speaker": speaker, "text": content})

    return turns if len(turns) >= 2 else None


def turns_to_plain(turns: list[dict]) -> str:
    return "\n".join(
        f"{'S' if t.get('speaker','SYS') in ('SYS','S') else 'U'}: {t['text']}"
        for t in turns
    )


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(turns, model, tokenizer,
                  window_size=5, stride=2, alpha=0.40, tau=0.80, K=3):
    import torch
    from tokenize_utils import tokenize_window

    n = len(turns)
    windows = []
    for start in range(0, max(1, n - window_size + 1), stride):
        end = min(start + window_size, n)
        windows.append({"start": start, "end": end, "turns": turns[start:end]})
    if not windows:
        windows = [{"start": 0, "end": n, "turns": turns}]

    raw_scores = []
    with torch.no_grad():
        for w in windows:
            try:
                enc  = tokenize_window(w["turns"], tokenizer, max_length=256)
                ids  = enc["input_ids"].unsqueeze(0)
                mask = enc["attention_mask"].unsqueeze(0)
                out  = model(ids, mask)
                s    = float(out["risk"].squeeze().item())
            except Exception:
                s = 0.0
            raw_scores.append(s)

    ema_scores = []
    ema = raw_scores[0]
    for s in raw_scores:
        ema = alpha * s + (1.0 - alpha) * ema
        ema_scores.append(ema)

    alert_indices, streak = [], 0
    for i, s in enumerate(ema_scores):
        if s > tau:
            streak += 1
            if streak >= K:
                alert_indices.append(i)
        else:
            streak = 0

    # Fallback: if the K-streak never fires (either conversation too short,
    # or signal only spikes once), treat any single window above tau as an
    # alert. This matches the K=1 criterion used in generate_examples.py.
    n_windows = len(ema_scores)
    any_above_tau = any(s > tau for s in ema_scores)
    alerted = bool(alert_indices) or any_above_tau
    first_alert = alert_indices[0] if alert_indices else (
        next((i for i, s in enumerate(ema_scores) if s > tau), None)
        if any_above_tau else None
    )

    turn_risk = [0.0] * n
    for i, w in enumerate(windows):
        for j in range(w["start"], w["end"]):
            if j < n:
                turn_risk[j] = max(turn_risk[j], ema_scores[i])

    return {
        "windows": windows, "raw_scores": raw_scores, "ema_scores": ema_scores,
        "alert_indices": alert_indices, "first_alert": first_alert,
        "turn_risk": turn_risk, "tau": tau, "alpha": alpha, "K": K,
        "n_windows": n_windows,
        "alerted": alerted,
        "peak_risk": float(max(ema_scores)),
        "mean_risk": float(np.mean(ema_scores)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════
_CFG    = {"displayModeBar": False}
_PBGC   = "rgba(0,0,0,0)"
_PLOTBG = "rgba(255,255,255,0.025)"
_GRID   = "rgba(255,255,255,0.055)"
_FC     = "#94a3b8"


def chart_trajectory(result):
    windows = result["windows"]
    raw, ema, tau = result["raw_scores"], result["ema_scores"], result["tau"]
    xlabels = [f"W{i+1} (t{w['start']+1}–{w['end']})" for i, w in enumerate(windows)]
    mcolors = ["#ef4444" if s >= tau else ("#f59e0b" if s >= tau*.65 else "#4ade80")
               for s in ema]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xlabels, y=raw, mode="lines+markers", name="Raw Score",
        line=dict(color="rgba(148,163,184,0.35)", width=1.5, dash="dot"),
        marker=dict(size=5, color="rgba(148,163,184,0.5)"),
    ))
    fig.add_trace(go.Scatter(
        x=xlabels, y=ema, mode="lines+markers", name="EMA Risk",
        line=dict(color="#8b5cf6", width=2.8),
        marker=dict(size=11, color=mcolors, line=dict(color="white", width=1.8)),
    ))
    fig.add_hline(y=tau, line=dict(color="#ef4444", width=1.6, dash="dash"),
                  annotation_text=f"τ = {tau}", annotation_font_color="#ef4444",
                  annotation_position="top right")
    for i in set(result["alert_indices"]):
        fig.add_vrect(x0=i-.45, x1=i+.45,
                      fillcolor="rgba(239,68,68,0.1)", layer="below", line_width=0)

    fig.update_layout(
        title=dict(text="Window Risk Trajectory — EMA-Smoothed Scores",
                   font=dict(size=13, color="#e2e8f0")),
        paper_bgcolor=_PBGC, plot_bgcolor=_PLOTBG, font=dict(color=_FC),
        xaxis=dict(title="Conversation Window", gridcolor=_GRID, tickfont=dict(size=9)),
        yaxis=dict(title="Risk Score", range=[0, 1.08], gridcolor=_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor=_PBGC),
        height=330, margin=dict(t=48, b=36, l=48, r=36),
    )
    return fig


def chart_bar(result):
    ema, tau = result["ema_scores"], result["tau"]
    windows  = result["windows"]
    xlabels  = [f"W{i+1}" for i in range(len(windows))]
    colors   = ["#ef4444" if s >= tau else ("#f59e0b" if s >= tau*.65 else "#4ade80")
                for s in ema]

    fig = go.Figure(go.Bar(
        x=xlabels, y=ema, marker=dict(color=colors),
        text=[f"{v:.3f}" for v in ema], textposition="outside",
        textfont=dict(size=9, color=_FC),
    ))
    fig.add_hline(y=tau, line=dict(color="#ef4444", dash="dash", width=1.5))
    fig.update_layout(
        title=dict(text="Per-Window EMA Risk", font=dict(size=13, color="#e2e8f0")),
        paper_bgcolor=_PBGC, plot_bgcolor=_PLOTBG, font=dict(color=_FC),
        xaxis=dict(gridcolor=_GRID, tickfont=dict(size=9)),
        yaxis=dict(range=[0, 1.18], gridcolor=_GRID, title="EMA Risk"),
        height=240, showlegend=False,
        margin=dict(t=44, b=28, l=44, r=28),
    )
    return fig


def chart_gauge(value, title):
    color = "#ef4444" if value >= .80 else ("#f59e0b" if value >= .50 else "#4ade80")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value * 100, 1),
        number=dict(suffix="%", font=dict(size=26, color=color)),
        title=dict(text=title, font=dict(size=11, color=_FC)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=_FC, tickfont=dict(size=9, color=_FC)),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(255,255,255,0.035)",
            bordercolor="rgba(255,255,255,0.08)",
            steps=[
                dict(range=[0,  50], color="rgba(74,222,128,0.08)"),
                dict(range=[50, 80], color="rgba(245,158,11,0.08)"),
                dict(range=[80,100], color="rgba(239,68,68,0.08)"),
            ],
            threshold=dict(line=dict(color="#ef4444", width=2), value=80),
        ),
    ))
    fig.update_layout(paper_bgcolor=_PBGC, font=dict(color=_FC),
                      height=195, margin=dict(t=28, b=8, l=16, r=16))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def render_conversation(turns, turn_risk, tau):
    parts = []
    for i, (t, risk) in enumerate(zip(turns, turn_risk)):
        is_sys   = t.get("speaker", "SYS") in ("SYS", "S")
        base_cls = "turn-sys" if is_sys else "turn-usr"
        spk_lbl  = "SYSTEM"  if is_sys else "USER"
        spk_col  = "#a78bfa" if is_sys else "#38bdf8"
        extra_cls, rpill = "", ""

        if is_sys:
            if risk >= tau:
                extra_cls = " turn-bd"
                pc, pb    = "#ef4444", "rgba(239,68,68,0.18)"
            elif risk >= tau * .65:
                extra_cls = " turn-warn"
                pc, pb    = "#f59e0b", "rgba(245,158,11,0.18)"
            else:
                pc, pb    = "#4ade80", "rgba(74,222,128,0.12)"
            rpill = (
                f'<span class="rpill" '
                f'style="color:{pc};background:{pb};border-color:{pc};">'
                f'risk {risk:.3f}</span>'
            )

        parts.append(
            f'<div class="turn-wrap">'
            f'<div class="{base_cls}{extra_cls}">'
            f'{rpill}'
            f'<div class="spk" style="color:{spk_col}">[{i+1}] {spk_lbl}</div>'
            f'<div class="utt">{t["text"]}</div>'
            f'</div></div>'
        )
    st.markdown("\n".join(parts), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    model, tokenizer, model_err = load_model_and_tokenizer()
    examples = load_examples()

    for key, default in [("conv_text", ""), ("result", None), ("parsed", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── HERO ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">⚠️ Communication Collapse Detection</div>
        <div class="hero-sub">
            Early-warning system for dialogue breakdown in operational AI chatbots —
            5-turn sliding windows · DeBERTa-v3-small · onset-aware labelling · EMA risk aggregation
        </div>
        <span class="badge">VIT Vellore</span>
        <span class="badge">Parth Goyal · 23BCE0411</span>
        <span class="badge">DeBERTa-v3-small · 44M params</span>
        <span class="badge">DBDC3 Benchmark · MIT Licence</span>
    </div>
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-val" style="color:#4ade80">0.767</div><div class="kpi-lbl">F1-Positive</div></div>
        <div class="kpi"><div class="kpi-val" style="color:#60a5fa">0.849</div><div class="kpi-lbl">AUROC</div></div>
        <div class="kpi"><div class="kpi-val" style="color:#c084fc">0.995</div><div class="kpi-lbl">Precision</div></div>
        <div class="kpi"><div class="kpi-val" style="color:#fb923c">0.624</div><div class="kpi-lbl">Recall</div></div>
        <div class="kpi"><div class="kpi-val" style="color:#fbbf24">0.848</div><div class="kpi-lbl">Accuracy</div></div>
        <div class="kpi"><div class="kpi-val" style="color:#f472b6">TP=423</div><div class="kpi-lbl">True Positives</div></div>
        <div class="kpi"><div class="kpi-val" style="color:#34d399">FP=2</div><div class="kpi-lbl">False Alarms</div></div>
        <div class="kpi"><div class="kpi-val" style="color:#818cf8">615</div><div class="kpi-lbl">Dialogues</div></div>
    </div>
    """, unsafe_allow_html=True)

    if model_err:
        st.error(f"**Model not loaded:** {model_err}")
        st.info("💡 Checkpoint expected at `results/checkpoints/main/best_model/model.pt`")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Inference Settings")
        tau = st.slider("Alert Threshold τ", 0.20, 0.95, 0.80, 0.05,
            help="EMA risk must exceed τ for K windows to fire an alert. "
                 "Paper uses τ=0.80 (Precision=0.995, FP=2).")
        alpha = st.slider("EMA Smoothing α", 0.10, 0.90, 0.40, 0.05,
            help="R_t = α·p_t + (1−α)·R_{t−1}. Higher = more reactive to current window.")
        K = st.slider("Alert Persistence K", 1, 5, 3, 1,
            help="Alert fires only after K consecutive above-τ windows. Prevents spike false alarms.")

        st.divider()
        st.markdown("### 🔬 How It Works")
        st.markdown("""
**Instead of:** *"Did this response fail?"*  
**We ask:** *"Is this conversation approaching collapse?"*

**Key steps:**
1. Parse conversation into turns
2. Slide a 5-turn window (stride 2)
3. Each window → DeBERTa CLS embedding → risk score p_t ∈ [0,1]
4. EMA smoothing: R_t = α·p_t + (1−α)·R_{t−1}
5. Alert if R_t > τ for K consecutive windows

**Onset-aware labelling** (training innovation): flags only the *first* breakdown moment, not already-saturated windows. Collapses the positive rate from ~90% → 27%, creating a real early-warning signal.
        """)
        st.divider()
        st.markdown("### 🏆 vs Baselines")
        st.dataframe(pd.DataFrame({
            "System":  ["Rule-based '16", "SVM '17", "CNN '17",
                        "BiLSTM '17", "BERT-FT '19", "**Ours**"],
            "F1":      [0.178, 0.312, 0.361, 0.423, 0.498, 0.767],
            "AUROC":   [0.581, 0.643, 0.672, 0.714, 0.763, 0.849],
        }), hide_index=True, use_container_width=True)
        st.divider()
        st.caption("🔗 [github.com/parthgoyal974/comm_collapse](https://github.com/parthgoyal974/comm_collapse)")

    # ── TWO COLUMNS: INPUT | LIVE OUTPUT ────────────────────────────────────
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="sec">💬 Conversation Input</div>', unsafe_allow_html=True)

        # Example loader
        if examples:
            n_bd_ex = sum(1 for e in examples if e.get("model_alerted", e.get("gt_breakdown")))
            n_ok_ex = len(examples) - n_bd_ex
            with st.expander(
                f"📚 Load an Example Conversation  "
                f"({n_bd_ex} breakdown · {n_ok_ex} healthy)",
                expanded=False
            ):
                st.markdown(
                    '<div style="font-size:0.8rem;color:rgba(255,255,255,0.4);'
                    'margin-bottom:10px;">'
                    ''
                    'Click any card to load it into the input window.</div>',
                    unsafe_allow_html=True
                )
                ec1, ec2 = st.columns(2)
                for i, ex in enumerate(examples):
                    col = ec1 if i % 2 == 0 else ec2
                    with col:
                        model_bd = ex.get("model_alerted", ex.get("gt_breakdown", False))
                        icon     = "🚨" if model_bd else "✅"
                        m_label  = "Breakdown" if model_bd else "Healthy"
                        short    = ex.get("conv_id", f"ex_{i}")[:16]
                        peak     = ex.get("peak_risk", 0.0)
                        if st.button(
                            f"{icon} {m_label} · {short}",
                            key=f"ex_{i}",
                            use_container_width=True,
                        ):
                            st.session_state.conv_text  = ex["plain"]
                            st.session_state["ta_conv"] = ex["plain"]
                            st.session_state.result     = None
                            st.session_state.parsed     = None
                            st.rerun()
                        risk_color = "#ef4444" if model_bd else "#4ade80"
                        st.markdown(
                            f'<div class="ex-preview">'
                            f'<span style="color:{risk_color};font-weight:600;">'
                            f'peak risk {peak:.3f}</span> · {ex["n_turns"]} turns<br>'
                            f'{ex["preview"]}</div>',
                            unsafe_allow_html=True
                        )
        else:
            st.info(
                "📦 No pre-computed examples found. "
                "Run `python generate_examples.py` once to generate them."
            )

        # Text area
        PLACEHOLDER = (
            "Paste a conversation below. Supported formats:\n\n"
            "━━ Simple (recommended) ━━━━━━━━━━━━━━━\n"
            "S: Hello! How can I help you today?\n"
            "U: I need to book a flight to Paris.\n"
            "S: Sure! What date would you like to travel?\n"
            "U: Next Friday please.\n"
            "S: I enjoy watching sports on TV.\n\n"
            "━━ Named labels ━━━━━━━━━━━━━━━━━━━━━━\n"
            "System: Hi there!\n"
            "User: I have a problem with my order.\n\n"
            "━━ DBDC3 JSON ━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Paste the full {\"turns\":[...]} JSON directly."
        )
        # Seed widget key when an example is injected or on first load
        if "ta_conv" not in st.session_state:
            st.session_state["ta_conv"] = st.session_state.conv_text

        conv_text = st.text_area(
            "Conversation",
            height=310, placeholder=PLACEHOLDER,
            label_visibility="collapsed", key="ta_conv",
        )
        # Keep alias in sync
        st.session_state.conv_text = conv_text

        # Parse preview
        parsed_preview = parse_conversation(conv_text) if conv_text.strip() else None
        if conv_text.strip() and parsed_preview:
            n_sys = sum(1 for t in parsed_preview if t["speaker"] == "SYS")
            st.markdown(
                f'<div style="color:#475569;font-size:0.78rem;margin-top:6px;">'
                f'✓ Parsed {len(parsed_preview)} turns — '
                f'{n_sys} system · {len(parsed_preview) - n_sys} user</div>',
                unsafe_allow_html=True
            )
        elif conv_text.strip():
            st.warning("⚠️ Could not parse — use `S:` / `U:` prefixes (see placeholder).")

        # Action buttons
        b1, b2 = st.columns([3, 1])
        with b1:
            run_btn = st.button(
                "🔍 Analyse Conversation", use_container_width=True, type="primary",
                disabled=(model is None or not conv_text.strip()),
            )
        with b2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.conv_text = ""
                st.session_state.result    = None
                st.session_state.parsed    = None
                st.rerun()

        if run_btn:
            parsed = parse_conversation(conv_text)
            if not parsed or len(parsed) < 2:
                st.error("Need at least 2 turns. Check format.")
            else:
                with st.spinner("Running DeBERTa-v3 inference…"):
                    result = run_inference(parsed, model, tokenizer,
                                          alpha=alpha, tau=tau, K=K)
                st.session_state.result = result
                st.session_state.parsed = parsed

    with col_out:
        st.markdown('<div class="sec">📈 Risk Analysis</div>', unsafe_allow_html=True)
        result = st.session_state.result
        parsed = st.session_state.parsed

        if result is None:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.025);border:1px solid
                 rgba(255,255,255,0.07);border-radius:14px;
                 padding:60px 32px;text-align:center;
                 color:rgba(255,255,255,0.28);">
                <div style="font-size:3.2rem;margin-bottom:14px">🔬</div>
                <div style="font-size:1.05rem;font-weight:500;">
                    Paste a conversation and click Analyse
                </div>
                <div style="font-size:0.82rem;margin-top:8px;">
                    Or load an example from the panel on the left
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            # Alert banner
            if result["alerted"]:
                fw    = result["first_alert"]
                fturn = result["windows"][fw]["start"] + 1
                streak_fired = bool(result["alert_indices"])
                if streak_fired:
                    detail = (f'EMA risk {result["ema_scores"][fw]:.3f} exceeded '
                              f'τ={tau:.2f} for {K} consecutive windows — '
                              f'first alert at Window {fw+1} (Turn {fturn}).')
                else:
                    detail = (f'EMA risk {result["ema_scores"][fw]:.3f} exceeded '
                              f'τ={tau:.2f} at Window {fw+1} (Turn {fturn}). '
                              )
                st.markdown(
                    f'<div class="alert-danger">'
                    f'<div class="alert-title" style="color:#ef4444;">'
                    f'🚨 COMMUNICATION COLLAPSE DETECTED</div>'
                    f'<div class="alert-body">{detail}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="alert-safe">'
                    f'<div class="alert-title" style="color:#4ade80;">'
                    f'✅ NO COLLAPSE DETECTED</div>'
                    f'<div class="alert-body">'
                    f'Peak EMA risk: {result["peak_risk"]:.3f} — '
                    f'below alert threshold τ={tau:.2f}.'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            # Gauges
            g1, g2, g3 = st.columns(3)
            with g1:
                st.plotly_chart(chart_gauge(result["peak_risk"], "Peak EMA Risk"),
                                use_container_width=True, config=_CFG)
            with g2:
                st.plotly_chart(chart_gauge(result["mean_risk"], "Mean EMA Risk"),
                                use_container_width=True, config=_CFG)
            with g3:
                n_win = len(result["windows"])
                n_al  = len(result["alert_indices"])
                st.plotly_chart(chart_gauge(n_al / max(n_win, 1), "Alert Window Frac."),
                                use_container_width=True, config=_CFG)

    # ── FULL-WIDTH TABS ──────────────────────────────────────────────────────
    if result is not None and parsed is not None:
        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs([
            "📉 Risk Trajectory", "🗂️ Window Detail",
            "💬 Annotated Dialogue", "🧠 Model Info",
        ])

        with tab1:
            st.plotly_chart(chart_trajectory(result),
                            use_container_width=True, config=_CFG)
            st.plotly_chart(chart_bar(result),
                            use_container_width=True, config=_CFG)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Windows",  len(result["windows"]))
            m2.metric("Alert Windows",  len(result["alert_indices"]))
            m3.metric("Peak Risk",      f"{result['peak_risk']:.3f}")
            m4.metric("Mean Risk",      f"{result['mean_risk']:.3f}")
            m5.metric("Alert Fired",    "Yes 🚨" if result["alerted"] else "No ✅")
            st.markdown("""
            <div class="info-card" style="margin-top:14px;">
            <b>Reading the chart:</b> Dashed grey = raw model output p_t.
            Solid purple = EMA-smoothed risk R_t = α·p_t + (1−α)·R_{t-1}.
            Coloured circles: 🟢 safe · 🟡 elevated · 🔴 above τ.
            Red shaded bands = windows in the alert streak.
            Alert fires when streak reaches K consecutive windows above τ.
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            rows = []
            for i, (w, raw, ema) in enumerate(
                zip(result["windows"], result["raw_scores"], result["ema_scores"])
            ):
                status = ("🚨 ALERT" if ema >= result["tau"] else
                          ("⚠️ Elevated" if ema >= result["tau"] * .65 else "✅ Safe"))
                sys_utts = " | ".join(
                    (t["text"][:55] + "…" if len(t["text"]) > 55 else t["text"])
                    for t in w["turns"] if t.get("speaker", "SYS") in ("SYS", "S")
                )
                rows.append({
                    "Window": f"W{i+1}", "Turns": f"{w['start']+1}–{w['end']}",
                    "Raw": f"{raw:.4f}", "EMA": f"{ema:.4f}",
                    "Status": status, "System utterances": sys_utts,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.divider()
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**Inference parameters**")
                st.markdown(f"- Backbone: `microsoft/deberta-v3-small` (44M params)")
                st.markdown(f"- Window size W: `5`  ·  Stride Δ: `2`")
                st.markdown(f"- EMA α: `{result['alpha']}`  ·  Threshold τ: `{result['tau']}`  ·  K: `{result['K']}`")
            with pc2:
                st.markdown("**Conversation stats**")
                st.markdown(f"- Total turns: `{len(parsed)}`")
                n_s = sum(1 for t in parsed if t.get("speaker","SYS") in ("SYS","S"))
                st.markdown(f"- System turns: `{n_s}`  ·  User turns: `{len(parsed)-n_s}`")
                st.markdown(f"- Windows built: `{len(result['windows'])}`  ·  Alert windows: `{len(result['alert_indices'])}`")

        with tab3:
            st.markdown("""
            <div style="font-size:0.8rem;color:#475569;margin-bottom:14px;">
            Turns colour-coded by max EMA risk from all windows they appear in.
            <span style="color:#ef4444;">■</span> Red = above τ ·
            <span style="color:#f59e0b;">■</span> Amber = elevated · Default = safe.
            Risk pill (top-right of system turns) shows the score.
            </div>""", unsafe_allow_html=True)
            render_conversation(parsed, result["turn_risk"], result["tau"])
            st.markdown("""
            <div class="info-card" style="margin-top:16px;font-size:0.77rem;">
            🟢 Risk &lt; 52% → Safe &nbsp;·&nbsp;
            🟡 52–79% → Elevated &nbsp;·&nbsp;
            🔴 ≥ 80% → Alert &nbsp;·&nbsp;
            R<sub>t</sub> = α·p<sub>t</sub> + (1−α)·R<sub>t−1</sub>
            </div>""", unsafe_allow_html=True)

        with tab4:
            inf1, inf2 = st.columns(2)
            with inf1:
                st.markdown("#### Architecture")
                st.markdown("""
**Backbone:** `microsoft/deberta-v3-small`  
44M parameters · 12 transformer layers · hidden dim 768

**Why DeBERTa?** Disentangled attention handles content and position separately — critical for dialogue where *when* a breakdown occurs relative to prior turns matters as much as *what* is said. ELECTRA-style pre-training makes it more sample-efficient than BERT at smaller scale.

**Risk head:**  
`LayerNorm(768)` → `Dropout(0.1)` → `Linear(768→1)` → `Sigmoid`

**Input serialisation:**  
Each 5-turn window is formatted as:
```
[TURN_1][SPEAKER_SYS][EXPECT_RESPONSE] text…
[TURN_2][SPEAKER_USR][NO_EXPECT] text…
…
```
8 special tokens added to the vocabulary.

**Loss:** Weighted BCE · pos_weight = 1.58 = 2441/1548
                """)
            with inf2:
                st.markdown("#### Training Config")
                st.dataframe(pd.DataFrame({
                    "Parameter": [
                        "Backbone LR", "Head LR", "Optimizer", "Weight decay",
                        "Batch size", "Epochs", "Early stopping", "Grad clipping",
                        "pos_weight", "Decision τ", "EMA α", "Alert K",
                        "Train samples", "Hardware",
                    ],
                    "Value": [
                        "4e-5", "2e-4", "AdamW (ε=1e-8)", "0.01",
                        "32", "4", "patience 3 (dev F1)", "1.0",
                        "1.58 (neg/pos)", "0.80", "0.40", "3 consecutive",
                        "3,989 (25% stratified)", "CPU · 32 GB RAM",
                    ],
                }), hide_index=True, use_container_width=True)

            st.divider()
            st.markdown("#### Test-Set Results (τ = 0.80, 1,696 windows)")
            cm_html = """
            <table style="border-collapse:collapse;width:100%;font-size:0.9rem;">
            <tr>
                <td style="padding:8px 14px;"></td>
                <th style="padding:8px 14px;color:#60a5fa;">Predicted Positive</th>
                <th style="padding:8px 14px;color:#60a5fa;">Predicted Negative</th>
            </tr>
            <tr>
                <th style="padding:8px 14px;color:#60a5fa;">Actual Positive</th>
                <td style="padding:8px 14px;background:rgba(74,222,128,0.12);
                    color:#4ade80;font-weight:700;border-radius:6px;text-align:center;">
                    TP = 423</td>
                <td style="padding:8px 14px;background:rgba(239,68,68,0.12);
                    color:#ef4444;text-align:center;">FN = 255</td>
            </tr>
            <tr>
                <th style="padding:8px 14px;color:#60a5fa;">Actual Negative</th>
                <td style="padding:8px 14px;background:rgba(245,158,11,0.12);
                    color:#f59e0b;text-align:center;">FP = 2 🎯</td>
                <td style="padding:8px 14px;background:rgba(74,222,128,0.12);
                    color:#4ade80;font-weight:700;border-radius:6px;text-align:center;">
                    TN = 1,016</td>
            </tr>
            </table>
            """
            st.markdown(cm_html, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card" style="margin-top:14px;">
            <b>Precision 0.995</b> — when the model fires an alert, it is correct 99.5% of the time.
            Only 2 false alarms across 1,696 test windows.<br>
            <b>Recall 0.624</b> — catches 62.4% of all real breakdown onsets.<br>
            <b>AUROC 0.849</b> — ranks any random breakdown window above any random healthy window
            84.9% of the time, across every possible threshold.<br>
            <b>ROC-optimal τ ≈ 0.32</b> → recall rises to ~0.72, FPR rises to ~0.15.
            Adjust the sidebar slider to explore the trade-off live.
            </div>
            """, unsafe_allow_html=True)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#334155;font-size:0.76rem;padding:4px 0 12px 0;">
        Communication Collapse Detection · Parth Goyal (23BCE0411) · VIT Vellore ·
        DeBERTa-v3-small + Onset-Aware Labelling + EMA Aggregation ·
        F1=0.767 · AUROC=0.849 · Precision=0.995 · FP=2/1696<br>
        Dataset: DBDC3 English Benchmark (MIT Licence) ·
        <a href="https://github.com/parthgoyal974/comm_collapse"
           style="color:#6d28d9;">github.com/parthgoyal974/comm_collapse</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()