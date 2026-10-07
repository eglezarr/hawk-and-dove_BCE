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
DECISIONS = {"subida": "Rate hike", "bajada": "Rate cut", "mantenimiento": "Rates on hold"}
DISCLAIMER = "For information purposes only. This is not investment advice."


def fmt(value, spec: str = "", default: str = "–") -> str:
    """Formatea un valor del resumen; '–' si aún no existe (p. ej., voz y cara antes de la fase 5)."""
    return default if value is None else format(value, spec)


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
    # voice y face valen null hasta la fase 5: «or {}» evita fallar con None
    stance = summary.get("stance") or {}
    voice = summary.get("voice") or {}
    face = summary.get("face") or {}
    pct, prev = stance.get("percentile_vs_history"), stance.get("previous_percentile")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rate decision", DECISIONS.get(stance.get("decision"), "–"))
    c2.metric("Stance", stance.get("label", "–"), fmt(stance.get("score"), "+.2f", None), delta_color="off")
    c3.metric("Percentile vs. history", fmt(pct),
              f"{pct - prev:+d} vs. {stance.get('previous_date')}" if None not in (pct, prev) else None,
              delta_color="off")
    c4.metric("Voice arousal (z vs. history)", fmt(voice.get("arousal_z_vs_history"), "+.1f"))
    c5.metric("Facial expression", face.get("label") or "–",
              f"valence {face['valence_mean']:+.2f}" if face.get("valence_mean") is not None else None,
              delta_color="off")
    st.caption(f"Percentile of the statement: {fmt(stance.get('percentile_statement'))} · "
               f"Q&A: {fmt(stance.get('percentile_qa'))} · "
               f"compared with {fmt(stance.get('history_size'))} press conferences up to this date")


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

    if data.has_times(sentences) and "score" in sentences:
        stance = alt.Chart(sentences.dropna(subset=["start", "end"])).mark_bar().encode(
            x=x, x2="end:Q", y=alt.Y("score:Q", title="stance", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("label:N", scale=alt.Scale(domain=list(STANCE_COLORS), range=list(STANCE_COLORS.values())),
                            legend=None),
            tooltip=["text", "label", "score"],
        )
        charts.append(stance + cursor)
    elif sentences is not None:
        st.caption("Stance is not on the timeline yet: sentences are not aligned with the video (block 2, phase 5).")

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
    cols = [c for c in ["time", "speaker", "section", "label", "score", "text", "face_valence", "voice_arousal"]
            if c in table]
    event = st.dataframe(table[cols], hide_index=True, width="stretch",
                         on_select="rerun", selection_mode="single-row", key="transcript")
    if not data.has_times(sentences):
        st.caption("Times will appear once the transcript is aligned with the video (block 2, phase 5).")
    rows = event.selection.rows if event else []
    start = sentences.iloc[rows[0]]["start"] if rows else None
    if start is not None and pd.notna(start) and int(start) != st.session_state.get("_last_row_jump"):
        st.session_state["_last_row_jump"] = int(start)
        jump_to(start)
        st.rerun()


def citation_list(citations: list[dict], event: str, key: str) -> None:
    """Citas [n] del informe o del chat: el marcador [n] del texto remite a citations[n-1].

    Cada cita muestra su fuente y su fragmento. Si es de la rueda abierta y ya tiene
    segundos, un botón salta a ese punto del vídeo; si no, enlaza a la transcripción oficial.
    """
    for n, c in enumerate(citations, start=1):
        st.markdown(f"**[{n}]** {c.get('source') or c.get('date', '')}")
        st.caption(c.get("snippet", ""))
        if c.get("date") == event and c.get("start") is not None:
            st.button(f"▶ {data.mmss(c['start'])}", key=f"{key}_{n}", on_click=jump_to, args=(c["start"],))
        elif c.get("url"):
            st.markdown(f"[Official transcript]({c['url']})")


def key_moments(moments: list[dict]) -> None:
    """Momentos clave del informe: frases más hawkish/dovish y, con voz o cara, picos de esas señales."""
    if not moments:
        return
    st.markdown("**Key moments**")
    for i, m in enumerate(moments):
        score = m.get("score")
        color = STANCE_COLORS["hawkish"] if (score or 0) > 0 else STANCE_COLORS["dovish"] if (score or 0) < 0 \
            else STANCE_COLORS["neutral"]
        when = f"{data.mmss(m.get('start'))} · " if m.get("start") is not None else ""
        st.markdown(f"<span style='color:{color}'>●</span> {when}*{m.get('reason', '')}*"
                    f"{f' ({score:+.2f})' if score is not None else ''} — {m.get('text', '')}",
                    unsafe_allow_html=True)
        if m.get("start") is not None:
            st.button("▶ Play", key=f"moment_{i}", on_click=jump_to, args=(m["start"],))


def briefing(event: str, summary: dict | None) -> None:
    st.subheader("Briefing")
    if not summary:
        st.info("Briefing not available yet (block 2).")
        return
    st.write(summary.get("briefing", ""))
    audio = data.briefing_audio(event)
    if audio:
        st.audio(str(audio))
    else:
        st.caption("Audio briefing not available yet (block 1).")
    citations = summary.get("citations") or []
    if citations:
        with st.expander(f"Sources ({len(citations)})"):
            citation_list(citations, event, key="cite")
    key_moments(summary.get("key_moments") or [])
    st.caption(f"⚠️ {summary.get('disclaimer') or DISCLAIMER}")


def projections(event: str) -> None:
    st.subheader("Staff projections")
    df = data.load_csv(event, "projections.csv")
    if df is None:
        st.info("projections.csv not available yet.")
        return
    st.dataframe(df.pivot(index="variable", columns="year", values="value"), width="stretch")
    st.altair_chart(alt.Chart(df).mark_line(point=True).encode(
        x="year:O", y=alt.Y("value:Q", title="%"), color="variable:N"), width="stretch")


@st.cache_resource(show_spinner="Loading the chat index…")
def prepare_chat() -> None:
    """Carga una vez por servidor el índice y el LLM del chat, para que la primera pregunta no espere."""
    if live.preparar is not None:
        live.preparar()


def chat(event: str) -> None:
    st.subheader("Ask the history")
    if live.responder is None:
        st.info("Chat not available yet (block 2: responder).")
        return
    try:
        prepare_chat()
    except Exception as exc:  # p. ej., índice sin generar: la app sigue sin el chat
        st.warning(f"Chat not available: {exc}")
        return
    for i, msg in enumerate(st.session_state["chat"]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("citations"):
                with st.expander(f"Sources ({len(msg['citations'])})"):
                    citation_list(msg["citations"], event, key=f"chat{i}")

    question = st.chat_input("Ask about past press conferences")
    audio_input = getattr(st, "audio_input", None)
    if live.voz_a_texto and audio_input:
        recording = audio_input("Or ask by voice")
        if recording is not None:
            try:
                question = live.voz_a_texto(recording.getvalue())
            except Exception as exc:  # p. ej., modelo de ASR sin instalar
                st.warning(f"Voice input not available: {exc}")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Searching past press conferences…"):
        answer = live.responder(question)
    citations = answer.get("citations") or []
    n = len(st.session_state["chat"]) + 1
    st.session_state["chat"] += [{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer["answer"], "citations": citations}]
    with st.chat_message("assistant"):
        st.write(answer["answer"])
        if citations:
            with st.expander(f"Sources ({len(citations)})", expanded=True):
                citation_list(citations, event, key=f"chat{n}")
        if live.texto_a_voz:
            try:
                st.audio(live.texto_a_voz(answer["answer"]))
            except Exception as exc:  # p. ej., Kokoro sin instalar: la respuesta escrita sigue valiendo
                st.caption(f"Voice answer not available: {exc}")
    st.caption(f"⚠️ {DISCLAIMER}")


# ---------- pestañas ----------

def tab_event(event: str) -> None:
    summary = data.load_summary(event)
    sentences = data.load_sentences(event)

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
    groups = data.load_benchmarks()
    if not groups:
        st.info("No benchmark results yet (benchmarks/results/).")
    for group, files in groups.items():
        title = "results" if group == "." else group
        with st.expander(f"{title} · {len(files['tables'])} tables, {len(files['figures'])} figures"):
            for png in files["figures"]:
                st.image(str(png), caption=png.stem)
            for csv in files["tables"]:
                st.markdown(f"**{csv.stem}**")
                st.dataframe(pd.read_csv(csv), hide_index=True, width="stretch")


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
