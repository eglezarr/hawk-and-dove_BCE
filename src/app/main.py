"""Hawk & Dove · interfaz Streamlit (bloque 3).

Solo lee los resultados de data/ y llama a las funciones del chat en vivo; no
ejecuta modelos de procesado. Arranque: run.sh / run.bat. La interfaz está en
inglés, como las salidas del producto.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo, por si no se instaló con pip -e

import altair as alt
import pandas as pd
import streamlit as st

from src.app import data, live

st.set_page_config(page_title="Hawk & Dove", page_icon="🦅", layout="wide")

STANCE_COLORS = {"hawkish": "#c0392b", "neutral": "#7f8c8d", "dovish": "#2471a3"}


# ---------- estado: segundo del vídeo al que saltar ----------

def jump_to(seconds: float) -> None:
    st.session_state["t"] = int(seconds)


st.session_state.setdefault("t", 0)
st.session_state.setdefault("chat", [])


# ---------- piezas de la pestaña principal ----------

def header(event: str, summary: dict | None) -> None:
    st.title("Hawk & Dove")
    st.caption(f"ECB press conference · {event}")
    if not summary:
        st.info("summary.json not available yet (block 2).")
        return
    stance, voice, face = summary.get("stance", {}), summary.get("voice", {}), summary.get("face", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stance", stance.get("label", "–"), f"{stance.get('score', 0):+.2f}", delta_color="off")
    c2.metric("Percentile vs. history", f"{stance.get('percentile_vs_history', '–')}")
    c3.metric("Voice arousal (z vs. history)", f"{voice.get('arousal_z_vs_history', 0):+.1f}")
    c4.metric("Facial expression", face.get("label", "–"), f"valence {face.get('valence_mean', 0):+.2f}",
              delta_color="off")


def video_panel(event: str) -> None:
    path = data.video_path(event)
    if path:
        st.video(str(path), start_time=st.session_state["t"])
    else:
        st.info(f"Video not downloaded: expected at data/raw/{event}.mp4 (scripts/).")
    st.caption(f"Position: {data.mmss(st.session_state['t'])}")


def timeline(event: str, sentences: pd.DataFrame | None) -> None:
    """Línea temporal: postura (B2), tono de voz (B1) y expresión facial (B3)."""
    charts = []
    x = alt.X("start:Q", title="seconds", scale=alt.Scale(zero=False))
    cursor = alt.Chart(pd.DataFrame({"t": [st.session_state["t"]]})).mark_rule(color="black").encode(x="t:Q")

    if sentences is not None and "score" in sentences:
        stance = alt.Chart(sentences).mark_bar().encode(
            x=x, x2="end:Q", y=alt.Y("score:Q", title="stance", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("label:N", scale=alt.Scale(domain=list(STANCE_COLORS), range=list(STANCE_COLORS.values())),
                            legend=None),
            tooltip=["text", "label", "score"],
        )
        charts.append(stance + cursor)

    voice = data.load_csv(event, "voice.csv")
    if voice is not None and "arousal" in voice:
        charts.append(alt.Chart(voice).mark_line(color="#8e44ad").encode(
            x=x, y=alt.Y("arousal:Q", title="voice arousal")) + cursor)

    face = data.load_csv(event, "face.csv")
    if face is not None:
        face = face[face["face_detected"].astype(bool) & (face["person"] == "presidenta")].copy()
        face["valence_smooth"] = face["valence"].rolling(5, min_periods=1, center=True).mean()
        charts.append(alt.Chart(face).mark_line(color="#d35400").encode(
            x=x, y=alt.Y("valence_smooth:Q", title="face valence", scale=alt.Scale(domain=[-1, 1]))) + cursor)

    if charts:
        st.altair_chart(alt.vconcat(*[c.properties(height=90, width="container") for c in charts]),
                        width="stretch")
    else:
        st.info("No signals available yet.")


def transcript(sentences: pd.DataFrame | None) -> None:
    """Transcripción con postura por frase; al seleccionar una fila, el vídeo salta a ese minuto."""
    if sentences is None:
        st.info("No transcript yet (blocks 1 and 2).")
        return
    table = sentences.assign(time=sentences["start"].map(data.mmss))
    cols = [c for c in ["time", "speaker", "label", "score", "text", "face_valence", "voice_arousal"] if c in table]
    event = st.dataframe(table[cols], hide_index=True, width="stretch",
                         on_select="rerun", selection_mode="single-row", key="transcript")
    rows = event.selection.rows if event else []
    if rows and int(sentences.iloc[rows[0]]["start"]) != st.session_state.get("_last_row_jump"):
        st.session_state["_last_row_jump"] = int(sentences.iloc[rows[0]]["start"])
        jump_to(sentences.iloc[rows[0]]["start"])
        st.rerun()


def briefing(event: str, summary: dict | None) -> None:
    st.subheader("Briefing")
    if not summary:
        st.info("Briefing not available yet (block 2).")
        return
    st.write(summary.get("briefing", ""))
    citations = summary.get("citations", [])
    if citations:
        cols = st.columns(len(citations))
        for col, c in zip(cols, citations):
            col.button(f"▶ {c['label']}", key=f"cite_{c['start']}", on_click=jump_to, args=(c["start"],))
    audio = data.briefing_audio(event)
    if audio:
        st.audio(str(audio))
    else:
        st.caption("Audio briefing not available yet (block 1).")


def projections(event: str) -> None:
    st.subheader("Staff projections")
    df = data.load_csv(event, "projections.csv")
    if df is None:
        st.info("projections.csv not available yet.")
        return
    st.dataframe(df.pivot(index="variable", columns="year", values="value"), width="stretch")
    st.altair_chart(alt.Chart(df).mark_line(point=True).encode(
        x="year:O", y=alt.Y("value:Q", title="%"), color="variable:N"), width="stretch")


def chat(event: str) -> None:
    st.subheader("Ask the history")
    if live.responder is None:
        st.info("Chat not available yet (block 2: responder).")
        return
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask about past press conferences")
    audio_input = getattr(st, "audio_input", None)
    if live.voz_a_texto and audio_input:
        recording = audio_input("Or ask by voice")
        if recording is not None:
            question = live.voz_a_texto(recording.getvalue())
    if not question:
        return

    st.session_state["chat"].append({"role": "user", "content": question})
    answer = live.responder(question)
    st.session_state["chat"].append({"role": "assistant", "content": answer["answer"]})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer["answer"])
        for c in answer.get("citations", []):
            label = f"{c['date']} {data.mmss(c['start'])}" if c.get("start") is not None else c["date"]
            if c["date"] == event and c.get("start") is not None:
                st.button(f"▶ {label}", key=f"chat_{c['date']}_{c['start']}", on_click=jump_to, args=(c["start"],))
            else:
                st.caption(f"{label} · {c.get('snippet', '')}")
        if live.texto_a_voz:
            st.audio(live.texto_a_voz(answer["answer"]))


# ---------- pestañas ----------

def tab_event(event: str) -> None:
    summary = data.load_summary(event)
    sentences = data.load_csv(event, "signals.csv")
    if sentences is None:
        sentences = data.load_csv(event, "stance.csv")

    header(event, summary)
    left, right = st.columns([3, 2])
    with left:
        video_panel(event)
        timeline(event, sentences)
        transcript(sentences)
    with right:
        briefing(event, summary)
        projections(event)
        chat(event)


def tab_history() -> None:
    st.subheader("Stance across press conferences")
    hist = data.load_history()
    if hist is None:
        st.info("History not available yet (block 2).")
        return
    long = hist.melt(id_vars="date", value_vars=["score", "score_statement", "score_qa"], var_name="series")
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x="date:T", y=alt.Y("value:Q", title="stance score"), color="series:N"), width="stretch")
    st.dataframe(hist, hide_index=True, width="stretch")


def tab_benchmarks() -> None:
    st.subheader("Model benchmarks")
    tables = data.load_benchmarks()
    if not tables:
        st.info("No benchmark results yet (benchmarks/results/).")
    for name, df in tables.items():
        st.markdown(f"**{name}**")
        st.dataframe(df, hide_index=True, width="stretch")


def main() -> None:
    events = data.list_events()
    with st.sidebar:
        st.header("Press conference")
        if not events:
            st.warning("No processed press conferences in data/events/.")
            return
        event = st.selectbox("Date", events, on_change=lambda: st.session_state.update(t=0))
        missing = data.missing_files(event)
        if missing:
            st.caption("Pending files: " + ", ".join(f"{f} ({b})" for f, b in missing.items()))

    t1, t2, t3 = st.tabs(["Press conference", "History", "Benchmarks"])
    with t1:
        tab_event(event)
    with t2:
        tab_history()
    with t3:
        tab_benchmarks()


main()
