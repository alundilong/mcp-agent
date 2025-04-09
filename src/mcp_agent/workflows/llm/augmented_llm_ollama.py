import json
import re
from typing import List, Type

import aiohttp  # Required for async HTTP requests

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
)

from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
    CallToolResult,
    ModelPreferences,
    RequestParams,
)

from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    ModelT,
)

from mcp_agent.logging.logger import get_logger
from mcp_agent.workflows.llm.augmented_llm_openai import MCPOpenAITypeConverter


class OllamaAugmentedLLM(
    AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]
):
    """
    An LLM implementation using a locally deployed Ollama model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPOpenAITypeConverter, **kwargs)

        self.provider = "Cerebras"
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        self.model_preferences = self.model_preferences or ModelPreferences(
            costPriority=0.5,
            speedPriority=0.3,
            intelligencePriority=0.2,
        )

        print("O" * 100)

        self.ollama_base_url = self.context.config.ollama.base_url
        chosen_model = "mistral"  # Adjust to your preferred local model

        # Get configuration from context if available
        if self.context and self.context.config and self.context.config.ollama:
            if hasattr(self.context.config.ollama, "default_model"):
                chosen_model = self.context.config.ollama.default_model

        self.default_request_params = self.default_request_params or RequestParams(
            model=chosen_model,
            modelPreferences=self.model_preferences,
            maxTokens=4096,
            systemPrompt=self.instruction,
            parallel_tool_calls=False,
            max_iterations=10,
            use_history=True,
        )

    @classmethod
    def convert_message_to_message_param(
        cls, message: ChatCompletionMessage, **kwargs
    ) -> ChatCompletionMessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=message.content,
            tool_calls=message.tool_calls,
            **kwargs,
        )

    async def generate(self, message, request_params: RequestParams | None = None):
        messages: List[ChatCompletionMessageParam] = []
        params = self.get_request_params(request_params)

        if params.use_history:
            messages.extend(self.history.get())

        if self.instruction and not messages:
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system", content=self.instruction
                )
            )

        if isinstance(message, str):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=message)
            )
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        prompt = self._build_prompt(messages)

        responses = []

        model = await self.select_model(params)
        async with aiohttp.ClientSession() as session:
            for i in range(params.max_iterations):
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                }

                self.logger.debug(f"Ollama API payload: {payload}")

                async with session.post(
                    f"{self.ollama_base_url}/", json=payload
                ) as resp:
                    result = await resp.text()
                    content = result

                    if not content:
                        break

                    message = ChatCompletionMessage(role="assistant", content=content)
                    responses.append(message)

                    messages.append(
                        ChatCompletionAssistantMessageParam(
                            role="assistant", content=content
                        )
                    )

                    break  # Stop after first generation for now

        if params.use_history:
            self.history.set(messages)

        return responses

    async def generate_str(
        self, message, request_params: RequestParams | None = None
    ) -> str:
        """
        Generate plain string output using the Ollama model.
        """
        responses = await self.generate(
            message=message,
            request_params=request_params,
        )

        final_text: List[str] = []

        for response in responses:
            content = response.content
            if not content:
                continue

            if isinstance(content, str):
                final_text.append(content)
            else:
                # In case Ollama returns content as structured parts (unlikely)
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        final_text.append(part["text"])

        return "\n".join(final_text)

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """
        Generate a structured response using Ollama by parsing the plain text output.
        This assumes the LLM returns JSON that can be parsed into a Pydantic model.
        """
        response = await self.generate_str(
            message=message,
            request_params=request_params,
        )

        self.logger.debug(f"Ollama raw structured response:\n{response}")

        try:
            # Try parsing as JSON
            parsed = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}\nRaw response: {response}"
            )

        try:
            # Convert JSON to the expected Pydantic model
            result = response_model.parse_obj(parsed)
        except Exception as e:
            raise ValueError(
                f"Failed to parse JSON into {response_model}: {e}\nParsed JSON: {parsed}"
            )

        return result

    def _build_prompt(self, messages: List[ChatCompletionMessageParam]) -> str:
        """Convert chat messages into a flat prompt string for Ollama."""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"[SYSTEM]: {content}\n"
            elif role == "user":
                prompt += f"[USER]: {content}\n"
            elif role == "assistant":
                prompt += f"[ASSISTANT]: {content}\n"
        return prompt.strip()
