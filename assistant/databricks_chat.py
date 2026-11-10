"""A LangChain chat model over a Databricks model-serving endpoint, through databricks-sdk.

The assistant only sends a list of messages and reads back text, so this is all it needs. It uses the SDK the
Databricks data mode already depends on; databricks-langchain would add databricks-connect, which pins numpy
below 2 and conflicts with the pipeline's requirements.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def _content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):  # content blocks
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content


class ChatDatabricksEndpoint(BaseChatModel):
    endpoint: str
    temperature: float = 0.0
    profile: str | None = None
    _client: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "databricks-serving-endpoint"

    def _workspace(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient(profile=self.profile)
        return self._client

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None,
                  **kwargs) -> ChatResult:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        reply = self._workspace().serving_endpoints.query(
            name=self.endpoint, temperature=self.temperature, stop=stop,
            messages=[ChatMessage(role=ChatMessageRole(ROLES[m.type]), content=_content(m)) for m in messages])
        usage = reply.usage
        metadata = None
        if usage is not None:
            metadata = {"input_tokens": usage.prompt_tokens or 0, "output_tokens": usage.completion_tokens or 0,
                        "total_tokens": usage.total_tokens or 0}
        message = AIMessage(content=reply.choices[0].message.content or "", usage_metadata=metadata)
        return ChatResult(generations=[ChatGeneration(message=message)])
