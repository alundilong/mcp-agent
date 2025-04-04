from mcp_agent.workflows.swarm.swarm import Swarm
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openrouter_deepseek import DeepSeekAugmentedLLM
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class OpenRouterSwarm(Swarm, DeepSeekAugmentedLLM):
    """
    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.),
    using Anthropic's API as the LLM.
    """

    async def generate(self, message, request_params: RequestParams | None = None):
        # Default to DeepSeek v3 free model
        chosen_model = "deepseek/deepseek-chat-v3-0324:free"
        # Get configuration from context if available
        if self.context and self.context.config and self.context.config.openrouter:
            if hasattr(self.context.config.openrouter, "default_model"):
                chosen_model = self.context.config.openrouter.default_model
        params = self.get_request_params(
            request_params,
            default=RequestParams(
                model=chosen_model,
                maxTokens=8192,
                parallel_tool_calls=False,
            ),
        )
        iterations = 0
        response = None
        agent_name = str(self.aggregator.name) if self.aggregator else None

        while iterations < params.max_iterations and self.should_continue():
            response = await super().generate(
                message=message
                if iterations == 0
                else "Please resolve my original request. If it has already been resolved then end turn",
                request_params=params.model_copy(
                    update={"max_iterations": 1}
                ),  # TODO: saqadri - validate
            )
            logger.debug(f"Agent: {agent_name}, response:", data=response)
            agent_name = self.aggregator.name if self.aggregator else None
            iterations += 1

        # Return final response back
        return response
