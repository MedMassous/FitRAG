"""
app.py — FitRAG Streamlit Application
A clean, production-grade fitness assistant powered by RAG + Groq.
"""

import os
import sys
import uuid         
import json
import tempfile
import logging
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Streamlit page config (MUST be first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FitRAG — AI Fitness Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0d0f14; color: #e8eaf0; }

    [data-testid="stSidebar"] {
        background: #13161e !important;
        border-right: 1px solid #1e2333;
    }

    .fit-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1219 100%);
        border: 1px solid #252b3d;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .fit-header h1 {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6ee7f7, #7c8dfa, #c77dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .fit-header p { color: #7a849c; margin: 4px 0 0 0; font-size: 0.95rem; }

    .user-bubble {
        background: #1a2035; border: 1px solid #2a3050;
        border-radius: 16px 16px 4px 16px; padding: 14px 18px;
        margin: 12px 0; max-width: 80%; margin-left: auto;
        color: #c5cde8; font-size: 0.95rem;
    }
    .assistant-bubble {
        background: #141820; border: 1px solid #252d42;
        border-radius: 4px 16px 16px 16px; padding: 16px 20px;
        margin: 12px 0; max-width: 90%; color: #dde1ef; line-height: 1.7;
    }

    .source-badge {
        display: inline-block; background: #1c2538; border: 1px solid #2e3f60;
        color: #6ee7f7; font-family: 'DM Mono', monospace; font-size: 0.72rem;
        padding: 3px 10px; border-radius: 6px; margin: 2px 3px;
    }

    .intent-pill {
        display: inline-block; background: #1a1f30; border: 1px solid #2e3750;
        color: #7c8dfa; font-size: 0.72rem; font-weight: 600;
        padding: 2px 10px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }

    .metric-card {
        background: #141820; border: 1px solid #1e2535;
        border-radius: 12px; padding: 16px; text-align: center;
    }
    .metric-card .val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #6ee7f7; }
    .metric-card .lbl { font-size: 0.78rem; color: #5c6680; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

    .stButton > button {
        background: linear-gradient(135deg, #3a4fff, #7c3aed);
        color: white; border: none; border-radius: 10px; font-weight: 600; transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(124,61,237,0.4); }

    hr { border-color: #1e2535; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d0f14; }
    ::-webkit-scrollbar-thumb { background: #252d42; border-radius: 3px; }
    .stAlert { border-radius: 10px; }

    .chunk-text {
        font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #8896b0;
        background: #0d0f14; border-radius: 8px; padding: 10px;
        border: 1px solid #1a2030; white-space: pre-wrap;
        max-height: 180px; overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Imports (after env load)
# ─────────────────────────────────────────────
try:
    from ingestion import IngestionPipeline
    from rag_pipeline import FitnessRAGPipeline, UserProfile
except ImportError as e:
    st.error(f"Import error: {e}. Make sure all dependencies are installed.")
    st.stop()

# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "ingester" not in st.session_state:
        st.session_state.ingester = None
    if "profile" not in st.session_state:
        st.session_state.profile = UserProfile.load()
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()


def get_pipeline() -> Optional[FitnessRAGPipeline]:
    if st.session_state.pipeline is None:
        if not os.getenv("GROQ_API_KEY"):
            return None
        try:
            st.session_state.pipeline = FitnessRAGPipeline()
        except Exception as e:
            st.error(f"Pipeline init failed: {e}")
            return None
    return st.session_state.pipeline


def get_ingester() -> IngestionPipeline:
    if st.session_state.ingester is None:
        st.session_state.ingester = IngestionPipeline()
    return st.session_state.ingester


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏋️ **FitRAG**")
        st.markdown("*Your AI-powered fitness coach*")
        st.divider()

        # ── API Key ─────────────────────────
        st.markdown("### 🔑 API Key")
        api_key = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            placeholder="gsk_…",
            help="Get your free key at console.groq.com",
        )
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            # Invalidate cached pipeline so it's re-created with new key
            if st.session_state.get("pipeline") is not None:
                st.session_state.pipeline = None
            st.success("✅ API key set", icon="🔑")

        st.divider()

        # ── PDF Upload ──────────────────────
        st.markdown("### 📄 Upload Training PDFs")
        uploaded_files = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Workout plans, training programs, fitness guides…",
        )

        col1, col2 = st.columns(2)
        with col1:
            ingest_btn  = st.button("⚡ Index PDFs",  use_container_width=True)
        with col2:
            reindex_btn = st.button("🔄 Re-index", use_container_width=True)

        if (ingest_btn or reindex_btn) and uploaded_files:
            ingester = get_ingester()
            tmp_paths = []
            Path("data/uploads").mkdir(parents=True, exist_ok=True)

            for uf in uploaded_files:
                dest = Path("data/uploads") / uf.name
                dest.write_bytes(uf.read())
                tmp_paths.append(str(dest))

            with st.spinner("Indexing PDFs…"):
                if reindex_btn:
                    result = ingester.reindex_all(tmp_paths)
                else:
                    result = ingester.ingest(tmp_paths)

            if st.session_state.pipeline:
                st.session_state.pipeline.refresh_retriever()

            st.success(
                f"✅ Indexed {len(result['indexed'])} file(s) | "
                f"{result['total_chunks']} total chunks"
            )
            if result["skipped"]:
                st.info(f"ℹ️ Skipped (already indexed): {', '.join(result['skipped'])}")

        # ── Indexed Files ───────────────────
        ingester = get_ingester()
        indexed  = ingester.get_indexed_files()

        if indexed:
            st.divider()
            st.markdown("### 📚 Knowledge Base")
            for f in indexed:
                with st.expander(f"📄 {f['filename']}", expanded=False):
                    st.caption(
                        f"Pages: {f['pages']} | "
                        f"Chunks: {f['chunks']} | "
                        f"Indexed: {f['ingested_at'][:10]}"
                    )

        st.divider()

        if indexed:
            total_chunks = sum(f["chunks"] for f in indexed)
            c1, c2 = st.columns(2)
            with c1: st.metric("PDFs", len(indexed))
            with c2: st.metric("Chunks", total_chunks)

        # ── Quick Actions ───────────────────
        st.divider()
        st.markdown("### ⚡ Quick Prompts")
        quick_prompts = [
            "What should I train today?",
            "Build me a full-week workout split",
            "What mistakes might I be making?",
            "Suggest a progressive overload plan",
            "When should I take a rest day?",
            "Analyse my training program",
        ]
        for qp in quick_prompts:
            if st.button(qp, key=f"qp_{qp}", use_container_width=True):
                st.session_state["pending_query"] = qp
                st.rerun()


# ─────────────────────────────────────────────
# User Profile Panel
# ─────────────────────────────────────────────

def render_profile_tab():
    st.markdown("### 👤 Athlete Profile")
    st.caption("This information is injected into every query for personalised advice.")

    profile  = st.session_state.profile
    pipeline = get_pipeline()

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name   = st.text_input("Name",          value=profile.name)
            age    = st.number_input("Age",          value=profile.age, min_value=0, max_value=120,
                                     help="Enter 0 if not set")
            weight = st.number_input("Weight (kg)",  value=profile.weight_kg, min_value=0.0,
                                     max_value=500.0, step=0.5)
            height = st.number_input("Height (cm)",  value=profile.height_cm, min_value=0.0,
                                     max_value=300.0, step=0.5)
        with col2:
            level   = st.selectbox("Fitness Level", ["beginner","intermediate","advanced"],
                                   index=["beginner","intermediate","advanced"].index(profile.fitness_level))
            goal_options = ["general fitness","fat loss","muscle gain","endurance","strength","athletic performance"]
            goal    = st.selectbox("Primary Goal", goal_options,
                                   index=goal_options.index(profile.goal) if profile.goal in goal_options else 0)
            days    = st.slider("Days/week available", 1, 7, profile.days_per_week)
            injuries = st.text_area("Injuries / Limitations", value=profile.injuries, height=70)

        if st.form_submit_button("💾 Save Profile", use_container_width=True):
            if pipeline:
                updated = pipeline.update_profile(
                    name=name, age=age, weight_kg=weight, height_cm=height,
                    fitness_level=level, goal=goal, days_per_week=days, injuries=injuries,
                )
                st.session_state.profile = updated
            else:
                profile_new = UserProfile(
                    name=name, age=age, weight_kg=weight, height_cm=height,
                    fitness_level=level, goal=goal, days_per_week=days, injuries=injuries,
                )
                profile_new.save()
                st.session_state.profile = profile_new
            st.success("Profile saved! ✅")


# ─────────────────────────────────────────────
# History Panel
# ─────────────────────────────────────────────

def render_history_tab():
    st.markdown("### 📜 Conversation History")

    pipeline = get_pipeline()
    if not pipeline:
        st.warning("No pipeline — set your API key first.")
        return

    history = pipeline.memory.all()
    if not history:
        st.info("No conversations yet. Ask a question to get started!")
        return

    for i, entry in enumerate(history[:20]):
        with st.expander(
            f"**{entry.timestamp[:16]}** — {entry.query[:60]}…",
            expanded=False,
        ):
            st.markdown(f"**Q:** {entry.query}")
            st.markdown(f"**A:** {entry.answer[:500]}{'…' if len(entry.answer) > 500 else ''}")
            st.caption(f"Sources: {', '.join(entry.sources)}")
            if entry.feedback == 1:
                st.caption("👍 Helpful")
            elif entry.feedback == -1:
                st.caption("👎 Not helpful")


# ─────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────

def render_message(msg: dict):
    role    = msg["role"]
    content = msg["content"]
    meta    = msg.get("meta", {})

    if role == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {content}</div>',
            unsafe_allow_html=True,
        )
    else:
        intent_html = ""
        if meta.get("intent"):
            intent_html = f'<span class="intent-pill">{meta["intent"]}</span>&nbsp;&nbsp;'

        source_html = ""
        if meta.get("sources"):
            badges = "".join(
                f'<span class="source-badge">📄 {s}</span>'
                for s in meta["sources"]
            )
            source_html = f"<div style='margin-top:12px'><strong style='font-size:0.8rem;color:#5c6680'>SOURCES</strong><br>{badges}</div>"

        usage_html = ""
        if meta.get("usage"):
            u = meta["usage"]
            usage_html = (
                f"<div style='margin-top:8px;font-size:0.72rem;color:#3d4560'>"
                f"Tokens: {u.get('total_tokens','?')} | "
                f"Model: {meta.get('model','')}"
                f"</div>"
            )

        st.markdown(
            f'<div class="assistant-bubble">'
            f'{intent_html}'
            f'<div style="margin-top:6px">{content}</div>'
            f'{source_html}'
            f'{usage_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        msg_id = msg.get("id", "")
        if msg_id and msg_id not in st.session_state.feedback_given:
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
            with fb_col1:
                if st.button("👍", key=f"up_{msg_id}"):
                    _give_feedback(msg_id, content, meta.get("query", ""), 1)
            with fb_col2:
                if st.button("👎", key=f"dn_{msg_id}"):
                    _give_feedback(msg_id, content, meta.get("query", ""), -1)

        if meta.get("chunks") and st.checkbox(
            f"🔍 Show {len(meta['chunks'])} retrieved chunks",
            key=f"chunks_{msg_id}",
        ):
            for j, chunk in enumerate(meta["chunks"]):
                with st.expander(
                    f"[{j+1}] {chunk['source']} — Page {chunk.get('page','?')} "
                    f"| Score: {chunk.get('rerank_score',0):.3f}",
                    expanded=False,
                ):
                    st.markdown(
                        f'<div class="chunk-text">{chunk["text"]}</div>',
                        unsafe_allow_html=True,
                    )


def _give_feedback(msg_id: str, answer: str, query: str, feedback: int):
    pipeline = get_pipeline()
    if pipeline and query:
        pipeline.submit_feedback(query, feedback)
    st.session_state.feedback_given.add(msg_id)
    icon = "👍" if feedback == 1 else "👎"
    st.toast(f"{icon} Feedback recorded!", icon="✅")


def render_chat_tab():
    st.markdown("""
    <div class="fit-header">
        <span style="font-size:2.5rem">🏋️</span>
        <div>
            <h1>FitRAG — AI Fitness Coach</h1>
            <p>Upload your training PDFs and ask anything about your fitness journey</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pipeline = get_pipeline()
    ingester = get_ingester()
    indexed  = ingester.get_indexed_files()

    # ── Status bar ──────────────────────────
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ Set your **Groq API key** in the sidebar to start chatting.", icon="🔑")
    elif not indexed:
        st.info("📄 Upload and index your training PDFs using the sidebar to get personalised advice.", icon="💡")
    else:
        total_chunks = sum(f["chunks"] for f in indexed)
        model_name   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        st.success(
            f"✅ Ready! {len(indexed)} PDF(s) indexed | {total_chunks} knowledge chunks | "
            f"Model: **{model_name}**",
            icon="🚀",
        )

    st.divider()

    # ── Chat history ────────────────────────
    with st.container():
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#3d4560">
                <div style="font-size:3rem">💬</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;margin-top:12px">
                    Start a conversation
                </div>
                <div style="font-size:0.85rem;margin-top:8px">
                    Try: "What should I train today?" or "Build me a weekly split"
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                render_message(msg)

    st.divider()

    # ── Source / depth controls ─────────────
    # FIX #9: `ingester` is an IngestionPipeline and has no `.retriever`
    # attribute — that attribute lives on FitnessRAGPipeline.  The original
    # code's hasattr() guard silently fell through to the else-branch every
    # time.  Now we use pipeline.retriever when available for an accurate,
    # deduplicated source list; otherwise we fall back to the metadata index.
    if pipeline and pipeline.retriever.chunks:
        available_sources = pipeline.retriever.available_sources()
    else:
        available_sources = [f["filename"] for f in indexed]

    sources      = ["All sources"] + available_sources
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        source_filter = st.selectbox(
            "Filter source",
            sources,
            label_visibility="collapsed",
            help="Restrict answers to a specific PDF",
        )
    with ctrl_col2:
        top_k = st.slider("Context depth", 3, 10, 6, label_visibility="collapsed")

    # ── Chat input ───────────────────────────
    # st.chat_input must NOT be nested inside st.columns — it must be at the
    # tab/page root level.  Controls above are in their own columns row; the
    # chat input is a separate element below them.
    default_val = st.session_state.pop("pending_query", "")
    query = st.chat_input("Ask your fitness coach anything…") or default_val

    if query:
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please set your Groq API key in the sidebar first.")
            return

        user_id = str(uuid.uuid4())[:8]
        st.session_state.messages.append({
            "id":      user_id,
            "role":    "user",
            "content": query,
            "meta":    {},
        })

        sf = None if source_filter == "All sources" else source_filter

        if not pipeline:
            st.error("Pipeline failed to initialise. Check your API key.")
            return

        with st.spinner("🧠 Thinking…"):
            try:
                result = pipeline.ask(query, top_k=top_k, source_filter=sf)
            except RuntimeError as e:
                # Catch rate-limit / auth errors surfaced from rag_pipeline
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                return

        assist_id = str(uuid.uuid4())[:8]
        st.session_state.messages.append({
            "id":      assist_id,
            "role":    "assistant",
            "content": result["answer"],
            "meta": {
                "sources": result["sources"],
                "chunks":  result["chunks"],
                "intent":  result["intent"],
                "model":   result["model"],
                "usage":   result["usage"],
                "query":   query,
            },
        })

        st.rerun()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    for d in ["data", "data/uploads", "data/faiss_index"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    init_state()
    render_sidebar()

    tab_chat, tab_profile, tab_history = st.tabs([
        "💬 Chat",
        "👤 Profile",
        "📜 History",
    ])

    with tab_chat:
        render_chat_tab()

    with tab_profile:
        render_profile_tab()

    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    main()
