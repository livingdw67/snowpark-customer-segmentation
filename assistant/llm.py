"""Pick the chat model: a platform-hosted model when live, a local Ollama model otherwise.

ASSISTANT_LLM   auto (default) | cortex | databricks | ollama   auto = the live platform's model, else ollama
CORTEX_MODEL  Cortex model name (default llama3.1-70b, available in most regions)
DATABRICKS_CHAT_ENDPOINT  serving endpoint name (default databricks-llama-3-3-70b)
OLLAMA_MODEL  Ollama model tag (default qwen2.5:7b: reliable text-to-SQL, fits in 8 GB of VRAM)
OLLAMA_HOST   Ollama server URL (default http://localhost:11434)
LLM_TIMEOUT   seconds to wait for one model reply before giving up (default 120)
"""
from __future__ import annotations

import os


def make_llm(session=None):
    backend = os.getenv("ASSISTANT_LLM", "auto").lower()
    if backend == "auto":
        databricks = os.getenv("ASSISTANT_MODE", "").lower() == "databricks"
        backend = "cortex" if session is not None else "databricks" if databricks else "ollama"

    if backend == "cortex":
        if session is None:
            raise RuntimeError("ASSISTANT_LLM=cortex needs a live Snowflake session (ASSISTANT_MODE=live).")
        from langchain_snowflake import ChatSnowflake

        return ChatSnowflake(session=session, model=os.getenv("CORTEX_MODEL", "llama3.1-70b"),
                             temperature=0)
    if backend == "databricks":
        from databricks_langchain import ChatDatabricks

        return ChatDatabricks(endpoint=os.getenv("DATABRICKS_CHAT_ENDPOINT", "databricks-llama-3-3-70b"), temperature=0)
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        # keep_alive holds the model in GPU memory between questions (Ollama's default is 5 min,
        # after which the next question waits for a reload); the timeout stops a stuck request.
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), temperature=0,
                          base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                          keep_alive="30m", client_kwargs={"timeout": float(os.getenv("LLM_TIMEOUT", 120))})
    raise ValueError(f"ASSISTANT_LLM must be auto, cortex, databricks or ollama, not {backend!r}")


def warm_up(llm) -> None:
    """Load a local model before the first question, so the first answer isn't the slow one."""
    if type(llm).__name__ == "ChatOllama":
        llm.invoke("Reply with one word: ready")


def describe(llm) -> str:
    name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or getattr(llm, "endpoint", None) or "custom"
    return f"{type(llm).__name__} ({name})"
