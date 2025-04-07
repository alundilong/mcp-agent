import asyncio
import streamlit as st
from typing import Dict
from mcp_agent.workflows.swarm.swarm import SwarmAgent, DoneAgent
from mcp_agent.workflows.swarm.swarm_openrouter import OpenRouterSwarm
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Set up page config
st.set_page_config(page_title="Flight Support Chat", layout="wide")


# === Tool Functions ===
def escalate_to_agent(reason: str = None) -> str:
    return f"📞 Escalating to human agent. Reason: {reason or 'N/A'}"


def valid_to_change_flight() -> str:
    return "✅ Customer eligible for flight change."


def change_flight() -> str:
    return "✈️ Flight changed successfully."


def initiate_refund() -> str:
    return "💸 Refund initiated."


def initiate_flight_credits() -> str:
    return "🎫 Flight credits applied."


def initiate_baggage_search() -> str:
    return "🧳 Baggage located."


def case_resolved() -> DoneAgent:
    return DoneAgent()


def transfer_to_triage() -> SwarmAgent:
    return triage_agent


def transfer_to_flight_modification() -> SwarmAgent:
    return flight_modification


def transfer_to_flight_cancel() -> SwarmAgent:
    return flight_cancel


def transfer_to_flight_change() -> SwarmAgent:
    return flight_change


def transfer_to_lost_baggage() -> SwarmAgent:
    return lost_baggage


# === Agent Instructions ===
FLY_AIR_AGENT_PROMPT = """You are a helpful airline support agent. Use tools to resolve issues. Never skip steps."""


def triage_instructions(ctx: Dict) -> str:
    return f"""Triage the user's issue to: flight modification or lost baggage.\nAsk clarifying questions if needed.\nCustomer: {ctx.get("customer_context", "")} | Flight: {ctx.get("flight_context", "")}"""


# === Agent Definitions ===
triage_agent = SwarmAgent(
    name="Triage Agent",
    instruction=lambda ctx: triage_instructions(ctx),
    functions=[transfer_to_flight_modification, transfer_to_lost_baggage],
    human_input_callback=None,
)

flight_modification = SwarmAgent(
    name="Flight Modification Agent",
    instruction=lambda ctx: f"Ask user if they want to cancel or change.\n{ctx.get('customer_context', '')} | {ctx.get('flight_context', '')}",
    functions=[transfer_to_flight_cancel, transfer_to_flight_change],
    human_input_callback=None,
)

flight_cancel = SwarmAgent(
    name="Flight Cancel Agent",
    instruction=lambda ctx: f"{FLY_AIR_AGENT_PROMPT}\nPolicy: flight_cancellation_policy.md",
    functions=[
        escalate_to_agent,
        initiate_refund,
        initiate_flight_credits,
        transfer_to_triage,
        case_resolved,
    ],
    human_input_callback=None,
    server_names=["filesystem"],
)

flight_change = SwarmAgent(
    name="Flight Change Agent",
    instruction=lambda ctx: f"{FLY_AIR_AGENT_PROMPT}\nPolicy: flight_change_policy.md",
    functions=[
        escalate_to_agent,
        change_flight,
        valid_to_change_flight,
        transfer_to_triage,
        case_resolved,
    ],
    human_input_callback=None,
    server_names=["filesystem"],
)

lost_baggage = SwarmAgent(
    name="Lost Baggage Agent",
    instruction=lambda ctx: f"{FLY_AIR_AGENT_PROMPT}\nPolicy: lost_baggage_policy.md",
    functions=[
        escalate_to_agent,
        initiate_baggage_search,
        transfer_to_triage,
        case_resolved,
    ],
    human_input_callback=None,
    server_names=["filesystem"],
)


# === Session State Management ===
def initialize_session_state():
    """Initialize all required session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.messages = []
        st.session_state.processing = False
        st.session_state.context_collected = False
        st.session_state.context_variables = {
            "config": {
                "openrouter": {"default_model": "deepseek/deepseek-chat-v3-0324:free"}
            }
        }
        st.session_state.initialized = True


def initialize_swarm():
    """Initialize the swarm agent with collected context"""
    if not st.session_state.get("swarm") and st.session_state.context_collected:
        st.session_state.swarm = OpenRouterSwarm(
            agent=triage_agent, context_variables=st.session_state.context_variables
        )


# === Context Collection ===
def collect_initial_context():
    """Handle initial context collection conversation"""
    if not st.session_state.context_collected:
        if not st.session_state.messages:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Welcome to Airline Support! Let's get started.\n\n"
                    "Please provide:\n1. Your name and customer ID (if available)\n"
                    "2. Your flight details (route, flight number, date)\n\n"
                    "Example:\n1. John Doe, ID: CUST123\n2. AA123 from JFK to LAX on 2024-06-15",
                }
            ]

        # Display existing messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle user input
        if user_input := st.chat_input("Enter your information..."):
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Simple parsing - in production you'd use more robust parsing
            try:
                parts = [p.strip() for p in user_input.split("\n") if p.strip()]
                if len(parts) >= 2:
                    st.session_state.context_variables.update(
                        {"customer_context": parts[0], "flight_context": parts[1]}
                    )
                    st.session_state.context_collected = True
                    initialize_swarm()
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Thank you! How can I assist you with your flight today?",
                        }
                    )
                    st.rerun()
                else:
                    raise ValueError("Incomplete information")
            except Exception:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I didn't get that. Please provide:\n1. Your name and ID\n2. Flight details\n\nExample:\n1. John Doe\n2. AA123 JFK to LAX",
                    }
                )
                st.rerun()
        return True
    return False


# In your handle_user_message function, modify the response handling:
async def handle_user_message(user_input: str):
    try:
        st.session_state.processing = True
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = await st.session_state.swarm.generate(
            message=user_input,
            request_params=RequestParams(
                model="deepseek/deepseek-chat-v3-0324:free",
                maxTokens=8192,
                parallel_tool_calls=False,
            ),
        )

        # Clean up the response before displaying
        if hasattr(response, "content"):
            # Extract and format the content properly
            clean_content = response.content.split("Final Response:")[-1].strip()
            clean_content = clean_content.replace("\\n", "\n")  # Fix newlines

            # Remove any remaining technical artifacts
            if "**," in clean_content:
                clean_content = clean_content.split("**,")[0]

            # Proper markdown formatting
            clean_content = clean_content.replace("##", "###")  # Make headers smaller
        else:
            clean_content = str(response)

        agent_name = getattr(st.session_state.swarm.aggregator, "name", "System")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"**{agent_name}**: {clean_content}"}
        )

    except Exception as e:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"❌ Error: {str(e)}"}
        )
    finally:
        st.session_state.processing = False
        st.rerun()


# === App Layout ===
def main():
    initialize_session_state()

    if collect_initial_context():
        st.stop()  # Stop here while collecting context

    st.title("🛫 Airline Assistant")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle normal chat input
    if user_input := st.chat_input(
        "How can I help with your flight today?", disabled=st.session_state.processing
    ):
        asyncio.run(handle_user_message(user_input))

    if st.session_state.processing:
        st.spinner("Processing your request...")


if __name__ == "__main__":
    main()
