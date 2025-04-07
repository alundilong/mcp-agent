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

# --- Streamlit Page Setup ---
st.set_page_config(page_title="MCP Chatbot", layout="wide")
st.title("🤖 MCP Agent Chatbot")


# --- MCP App Initialization ---
@st.cache_resource
def initialize_mcp_app():
    app = MCPApp(name="mcp_basic_agent_openrouter")
    asyncio.run(app.initialize())
    return app


app = initialize_mcp_app()

# --- Session State Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False
if "latest_input" not in st.session_state:
    st.session_state.latest_input = ""


# --- Async LLM Call ---
async def generate_response(prompt):
    try:
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
    except asyncio.CancelledError:
        return "*[Response cancelled by user]*"


# --- User Input Trigger ---
user_input = st.chat_input("Type your message...")

if user_input and not st.session_state.is_generating:
    st.session_state.latest_input = user_input
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.is_generating = True
    st.session_state.cancel_requested = False
    st.rerun()


# --- Display Chat History ---
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])


# --- Assistant Response Generation / Cancel Block ---
if st.session_state.is_generating:
    with st.chat_message("assistant"):
        spinner = st.empty()
        spinner.markdown("⏳ Generating response... (press Cancel below)")

    col1, _ = st.columns([1, 5])
    with col1:
        if st.button("❌ Cancel"):
            st.session_state.cancel_requested = True
            st.session_state.is_generating = False
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "*[Cancelled by user]*"}
            )
            st.rerun()

    if not st.session_state.cancel_requested:
        response = asyncio.run(generate_response(st.session_state.latest_input))
        if not st.session_state.cancel_requested:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
        st.session_state.is_generating = False
        st.rerun()
