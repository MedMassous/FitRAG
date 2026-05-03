"""
rag_pipeline.py — RAG Pipeline with Groq LLM
Combines retrieval context with a structured fitness-focused prompt.
Includes: memory, user profile, feedback tracking, anti-hallucination guards.
"""

import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict, field

from groq import Groq
from groq import RateLimitError, AuthenticationError, APIError  

from retriever import FitnessRetriever

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────

MEMORY_PATH  = Path("data/memory.json")
PROFILE_PATH = Path("data/user_profile.json")
GROQ_MODEL  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = 0.4


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class UserProfile:
    name:          str   = "Athlete"
    age:           int   = 0
    weight_kg:     float = 0.0
    height_cm:     float = 0.0
    fitness_level: str   = "intermediate"
    goal:          str   = "general fitness"
    days_per_week: int   = 4
    injuries:      str   = "none"

    @classmethod
    def load(cls) -> "UserProfile":
        if PROFILE_PATH.exists():
            with open(PROFILE_PATH) as f:
                return cls(**json.load(f))
        return cls()

    def save(self):
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILE_PATH, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class MemoryEntry:
    query:     str
    answer:    str
    sources:   List[str]
    feedback:  Optional[int]  = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MemoryStore:
    """Rolling memory of past Q&A with feedback."""

    def __init__(self, max_entries: int = 50):
        self.path = MEMORY_PATH
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = self._load()

    def _load(self) -> List[MemoryEntry]:
        if self.path.exists():
            with open(self.path) as f:
                raw = json.load(f)
            return [MemoryEntry(**r) for r in raw]
        return []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def add(self, entry: MemoryEntry):
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        self._save()

    def set_feedback(self, query: str, feedback: int):
        for e in reversed(self.entries):
            if e.query == query:
                e.feedback = feedback
                self._save()
                return

    def recent_context(self, n: int = 3) -> str:
        if not self.entries:
            return ""
        recent = self.entries[-n:]
        lines  = []
        for e in recent:
            lines.append(f"Q: {e.query}\nA (summary): {e.answer[:200]}…")
        return "\n\n".join(lines)

    def all(self) -> List[MemoryEntry]:
        return list(reversed(self.entries))


# ─────────────────────────────────────────────
# Prompt Engineering
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert fitness coach and sports scientist.
You have access to the athlete's training documents and past workout history.
Your role is to provide SPECIFIC, EVIDENCE-BASED fitness advice drawn from those documents.

RULES:
1. Base your advice ONLY on the provided context chunks. Do NOT hallucinate programs.
2. If the context doesn't contain relevant information, say so clearly.
3. Always include progressive overload principles when building workouts.
4. Cite which document your advice comes from (e.g., "[from: program_A.pdf]").
5. Structure workout plans with: Exercise | Sets | Reps | Rest | Notes.
6. Flag potential injury risks when relevant.
7. Be concise but complete. Use markdown formatting."""


def _build_user_message(
    query:      str,
    context:    str,
    profile:    UserProfile,
    memory_ctx: str,
) -> str:
    profile_str = (
        f"Athlete Profile:\n"
        f"  - Level: {profile.fitness_level}\n"
        f"  - Goal: {profile.goal}\n"
        f"  - Days/week: {profile.days_per_week}\n"
        f"  - Weight: {profile.weight_kg} kg\n"
        f"  - Injuries/Limitations: {profile.injuries}\n"
    )

    memory_section = (
        f"\n--- Recent Conversation History ---\n{memory_ctx}\n"
        if memory_ctx else ""
    )

    return (
        f"{profile_str}"
        f"{memory_section}"
        f"\n--- Relevant Training Documents (use these as your primary source) ---\n"
        f"{context}\n"
        f"\n--- Athlete's Question ---\n"
        f"{query}\n\n"
        f"Provide a structured, practical response based on the documents above."
    )


def _format_context(chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant documents found in the knowledge base."
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Source: {c['source']} | Page {c.get('page', '?')}\n"
            f"{c['text'].strip()}\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Intent Detection
# ─────────────────────────────────────────────

QueryIntent = Literal[
    "plan_workout", "analyse_mistakes", "rest_day", "nutrition",
    "progress_check", "general"
]

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "plan_workout":    ["workout", "train", "exercise", "session", "build", "plan", "today", "split"],
    "analyse_mistakes":["mistake", "wrong", "improve", "bad", "error", "fix", "problem", "issue"],
    "rest_day":        ["rest", "recovery", "sleep", "fatigue", "tired", "off day"],
    "nutrition":       ["eat", "diet", "protein", "calories", "food", "nutrition", "meal", "macro"],
    "progress_check":  ["progress", "improve", "stronger", "better", "gains", "results"],
}

def detect_intent(query: str) -> QueryIntent:
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent  # type: ignore
    return "general"


# ─────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────

class FitnessRAGPipeline:
    """
    Full RAG pipeline:
      query → retrieve → prompt → Groq LLM → structured response
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable not set.")

        self.client    = Groq(api_key=api_key)
        self.retriever = FitnessRetriever()
        self.memory    = MemoryStore()
        self.profile   = UserProfile.load()

    # ── Public API ────────────────────────────

    def ask(
        self,
        query:         str,
        top_k:         int            = 6,
        source_filter: Optional[str]  = None,
        use_memory:    bool           = True,
    ) -> Dict:
        """
        Process a user query end-to-end.

        Returns:
            {
                "answer":   str,
                "sources":  List[str],
                "chunks":   List[Dict],
                "intent":   str,
                "model":    str,
                "usage":    Dict,
            }
        """
        intent         = detect_intent(query)
        log.info(f"Intent: {intent} | Query: '{query[:80]}'")
        expanded_query = self._expand_query(query, intent)

        chunks  = self.retriever.retrieve(expanded_query, top_k=top_k, source_filter=source_filter)
        context = _format_context(chunks)

        memory_ctx = self.memory.recent_context(n=3) if use_memory else ""
        user_msg   = _build_user_message(
            query=query, context=context, profile=self.profile, memory_ctx=memory_ctx
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=False,
            )
        except AuthenticationError:
            raise RuntimeError(
                "Groq authentication failed — please check your API key in the sidebar."
            )
        except RateLimitError:
            raise RuntimeError(
                "Groq rate limit reached. Wait a moment and try again, "
                "or switch to a smaller model (e.g. llama3-8b-8192)."
            )
        except APIError as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc

        answer = response.choices[0].message.content.strip()
        usage  = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }
        sources = list({c["source"] for c in chunks})

        self.memory.add(MemoryEntry(query=query, answer=answer, sources=sources))

        return {
            "answer":  answer,
            "sources": sources,
            "chunks":  chunks,
            "intent":  intent,
            "model":   GROQ_MODEL,
            "usage":   usage,
        }

    def submit_feedback(self, query: str, feedback: int):
        self.memory.set_feedback(query, feedback)
        log.info(f"Feedback {feedback} recorded for: '{query[:60]}'")

    def update_profile(self, **kwargs) -> UserProfile:
        for k, v in kwargs.items():
            if hasattr(self.profile, k):
                setattr(self.profile, k, v)
        self.profile.save()
        return self.profile

    def refresh_retriever(self):
        """Call after new PDFs are ingested."""
        self.retriever.refresh()

    # ── Fitness Logic Layer ───────────────────

    def _expand_query(self, query: str, intent: QueryIntent) -> str:
        expansions: Dict[str, str] = {
            "plan_workout":     "training program exercise sets reps workout split",
            "analyse_mistakes": "common mistakes errors programming faults technique",
            "rest_day":         "recovery rest active recovery deload fatigue management",
            "nutrition":        "nutrition diet macros protein calories meal timing",
            "progress_check":   "progression overload performance improvement tracking",
            "general":          "",
        }
        suffix = expansions.get(intent, "")
        return f"{query} {suffix}".strip() if suffix else query

    # ── Streaming version ─────────────────────

    def ask_stream(
        self,
        query:      str,
        top_k:      int  = 6,
        use_memory: bool = True,
    ):
        """
        Generator-based streaming response.
        Yields text tokens as they arrive from Groq.

        After iteration completes, call .send(None) is not needed — the
        generator simply exhausts.  Metadata is stored to memory automatically.
        """
        intent    = detect_intent(query)
        expanded  = self._expand_query(query, intent)
        chunks    = self.retriever.retrieve(expanded, top_k=top_k)
        context   = _format_context(chunks)
        memory_ctx = self.memory.recent_context(n=3) if use_memory else ""

        user_msg  = _build_user_message(query, context, self.profile, memory_ctx)
        messages  = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        try:
            stream = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )
        except AuthenticationError:
            raise RuntimeError("Groq authentication failed — check your API key.")
        except RateLimitError:
            raise RuntimeError("Groq rate limit reached. Please wait and try again.")
        except APIError as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc

        full_answer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_answer += delta
            yield delta

        sources = list({c["source"] for c in chunks})
        self.memory.add(MemoryEntry(query=query, answer=full_answer, sources=sources))
