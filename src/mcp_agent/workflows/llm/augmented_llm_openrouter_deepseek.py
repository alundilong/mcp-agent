import json
import re
from typing import Iterable, List, Type

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ModelPreferences,
    TextContent,
    TextResourceContents,
)

from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    ModelT,
    MCPMessageParam,
    MCPMessageResult,
    ProviderToMCPConverter,
    RequestParams,
)
from mcp_agent.logging.logger import get_logger

from mcp_agent.workflows.llm.augmented_llm_openai import MCPOpenAITypeConverter


class DeepSeekAugmentedLLM(
    AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]
):
    """
    An LLM implementation using DeepSeek v3 model via OpenRouter API.
    Maintains the same interface as OpenAIAugmentedLLM but routes requests to OpenRouter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPOpenAITypeConverter, **kwargs)

        self.provider = "DeepSeek (via OpenRouter)"
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        # DeepSeek-specific configuration
        self.model_preferences = self.model_preferences or ModelPreferences(
            costPriority=0.7,  # Higher weight for cost since DeepSeek is more cost-effective
            speedPriority=0.2,
            intelligencePriority=0.1,
        )

        # Default to DeepSeek v3 free model
        chosen_model = "deepseek/deepseek-chat-v3-0324:free"
        
        # Get configuration from context if available
        if self.context and self.context.config and self.context.config.openrouter:
            if hasattr(self.context.config.openrouter, "default_model"):
                chosen_model = self.context.config.openrouter.default_model
                
        self.default_request_params = self.default_request_params or RequestParams(
            model=chosen_model,
            modelPreferences=self.model_preferences,
            maxTokens=4096,  # DeepSeek supports up to 128k but keeping same as OpenAI for consistency
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
        """
        Process a query using DeepSeek model via OpenRouter.
        Maintains the same interface as OpenAIAugmentedLLM.generate().
        """
        config = self.context.config
        openrouter_client = OpenAI(
            api_key=config.openrouter.api_key,
            base_url=config.openrouter.base_url
        )

        messages: List[ChatCompletionMessageParam] = []
        params = self.get_request_params(request_params)

        if params.use_history:
            messages.extend(self.history.get())

        system_prompt = self.instruction or params.systemPrompt
        if system_prompt and len(messages) == 0:
            messages.append(
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            )

        if isinstance(message, str):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=message)
            )
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ChatCompletionToolParam] = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            )
            for tool in response.tools
        ]
        if not available_tools:
            available_tools = None

        responses: List[ChatCompletionMessage] = []
        model = await self.select_model(params)

        # OpenRouter specific headers
        extra_headers = {
            "HTTP-Referer": config.openrouter.site_url if hasattr(config.openrouter, "site_url") else "https://example.com",
            "X-Title": config.openrouter.site_name if hasattr(config.openrouter, "site_name") else "MCP Agent"
        }

        for i in range(params.max_iterations):
            arguments = {
                "extra_body": {},
                "model": model,
                "messages": messages,
                "extra_headers": extra_headers
            }

            self.logger.debug(f"OpenRouter API arguments: {arguments}")
            self._log_chat_progress(chat_turn=len(messages) // 2, model=model)
            print(f"{i} Sending Request")
            executor_result = await self.executor.execute(
                openrouter_client.chat.completions.create, **arguments
            )
            
            response = executor_result[0]
            print(f"Received response:{response}")

            self.logger.debug(
                "DeepSeek ChatCompletion response:",
                data=response,
            )

            if isinstance(response, BaseException):
                self.logger.error(f"Error: {response}")
                break

            if not response.choices or len(response.choices) == 0:
                break

            choice = response.choices[0]
            message = choice.message
            responses.append(message)

            sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", self.name) if isinstance(self.name, str) else None

            converted_message = self.convert_message_to_message_param(
                message, name=sanitized_name
            )
            messages.append(converted_message)

            if (
                choice.finish_reason in ["tool_calls", "function_call"]
                and message.tool_calls
            ):
                tool_tasks = [
                    self.execute_tool_call(tool_call)
                    for tool_call in message.tool_calls
                ]
                tool_results = await self.executor.execute(*tool_tasks)
                self.logger.debug(
                    f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                )
                for result in tool_results:
                    if isinstance(result, BaseException):
                        self.logger.error(
                            f"Warning: Unexpected error during tool execution: {result}. Continuing..."
                        )
                        continue
                    if result is not None:
                        messages.append(result)
            elif choice.finish_reason == "length":
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'length'"
                )
                break
            elif choice.finish_reason == "content_filter":
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'content_filter'"
                )
                break
            elif choice.finish_reason == "stop":
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'stop'"
                )
                break

        if params.use_history:
            self.history.set(messages)

        self._log_chat_finished(model=model)

        return responses

    # The following methods can remain identical to OpenAIAugmentedLLM as they don't need modification
    async def generate_str(self, message, request_params: RequestParams | None = None):
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
                continue

        return "\n".join(final_text)

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        import instructor

        response = await self.generate_str(
            message=message,
            request_params=request_params,
        )

        # For structured extraction, we'll still use OpenAI as instructor is optimized for it
        # Alternatively, you could implement DeepSeek-specific structured extraction
        client = instructor.from_openai(
            OpenAI(
                api_key=self.context.config.openai.api_key,
                base_url=self.context.config.openai.base_url,
            ),
            mode=instructor.Mode.TOOLS_STRICT,
        )

        params = self.get_request_params(request_params)
        model = await self.select_model(params)

        structured_response = client.chat.completions.create(
            model=model or "gpt-4",
            response_model=response_model,
            messages=[
                {"role": "user", "content": response},
            ],
        )

        return structured_response

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        return result

    async def execute_tool_call(
        self,
        tool_call: ChatCompletionToolParam,
    ) -> ChatCompletionToolMessageParam | None:
        tool_name = tool_call.function.name
        tool_args_str = tool_call.function.arguments
        tool_call_id = tool_call.id
        tool_args = {}

        try:
            if tool_args_str:
                tool_args = json.loads(tool_args_str)
        except json.JSONDecodeError as e:
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call_id,
                content=f"Invalid JSON provided in tool call arguments for '{tool_name}'. Failed to load JSON: {str(e)}",
            )

        tool_call_request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=tool_args),
        )

        result = await self.call_tool(
            request=tool_call_request, tool_call_id=tool_call_id
        )

        if result.content:
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call_id,
                content=[mcp_content_to_openai_content(c) for c in result.content],
            )

        return None

    def message_param_str(self, message: ChatCompletionMessageParam) -> str:
        if message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for part in content:
                    text_part = part.get("text")
                    if text_part:
                        final_text.append(str(text_part))
                    else:
                        final_text.append(str(part))

                return "\n".join(final_text)

        return str(message)

    def message_str(self, message: ChatCompletionMessage) -> str:
        content = message.content
        if content:
            return content

        return str(message)