import os
import asyncio
import streamlit as st

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_openrouter_deepseek import (
    DeepSeekAugmentedLLM,
)

# Set Streamlit page config
st.set_page_config(page_title="MCP Chatbot", layout="wide")
st.title("🤖 MCP Agent Chatbot")


# MCP app initialization (do this once)
@st.cache_resource
def initialize_mcp_app():
    app = MCPApp(name="mcp_basic_agent_openrouter")
    asyncio.run(app.initialize())
    return app


app = initialize_mcp_app()


# Function to handle a single user prompt
async def handle_prompt(prompt):
    async with app.run() as agent_app:
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem and can fetch URLs.
            Answer user questions using your tools and return content and file info as needed.""",
            server_names=["fetch", "filesystem"],
        )

        async with agent:
            llm = await agent.attach_llm(DeepSeekAugmentedLLM)
            response = await llm.generate_str(
                message=prompt,
                request_params=RequestParams(
                    modelPreferences=ModelPreferences(
                        costPriority=0.1,
                        speedPriority=0.2,
                        intelligencePriority=0.7,
                    )
                ),
            )
            return response


# Initialize session state for message history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# Chat input field (always shows up)
user_input = st.chat_input("Type your message...")

# Process input and update chat
if user_input:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(handle_prompt(user_input))
            st.markdown(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
