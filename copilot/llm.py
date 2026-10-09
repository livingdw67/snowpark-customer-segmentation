"""Pick the chat model: Snowflake Cortex when live, a local Ollama model otherwise.

COPILOT_LLM   auto (default) | cortex | ollama   auto = cortex with a live session, else ollama
CORTEX_MODEL  Cortex model name (default llama3.1-70b, available in most regions)
OLLAMA_MODEL  Ollama model tag (default qwen2.5:7b: reliable text-to-SQL, fits in 8 GB of VRAM)
OLLAMA_HOST   Ollama server URL (default http://localhost:11434)
"""
from __future__ import annotations

import os


def make_llm(session=None):
    backend = os.getenv("COPILOT_LLM", "auto").lower()
    if backend == "auto":
        backend = "cortex" if session is not None else "ollama"

    if backend == "cortex":
        if session is None:
            raise RuntimeError("COPILOT_LLM=cortex needs a live Snowflake session (COPILOT_MODE=live).")
        from langchain_snowflake import ChatSnowflake

        return ChatSnowflake(session=session, model=os.getenv("CORTEX_MODEL", "llama3.1-70b"),
                             temperature=0)
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), temperature=0,
                          base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    raise ValueError(f"COPILOT_LLM must be auto, cortex or ollama, not {backend!r}")


def describe(llm) -> str:
    name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or "custom"
    return f"{type(llm).__name__} ({name})"
