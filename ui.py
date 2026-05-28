"""Streamlit UI for the multi-agent answer aggregator."""

import streamlit as st
from agent_pipeline import answer_question, ask_for_clarification


st.set_page_config(
    page_title="Multi-Agent Answer Aggregator",
    page_icon="A",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp { background-color: #F8F8F4; }
        h1, h2, h3 { color: #222222; font-weight: 500; }
        section[data-testid="stSidebar"] {
            background-color: #F1F1EB;
            border-right: 1px solid #E0E0E0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_chat() -> None:
    """Clear the current chat and return the app to the first step."""

    st.session_state.messages = []
    st.session_state.current_step = "input"
    st.session_state.pending_question = ""
    st.session_state.clarification_question = ""


def initialize_session_state() -> None:
    """Create Streamlit session fields used by the simple chat flow."""

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = "input"
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""
    if "clarification_question" not in st.session_state:
        st.session_state.clarification_question = ""


def show_messages() -> None:
    """Render the chat history using Streamlit chat bubbles."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


initialize_session_state()


with st.sidebar:
    st.header("Project Scope")
    st.write("A beginner PoC for a multi-agent answer aggregator.")
    st.markdown(
        """
        Steps:
        1. Clarify the question if needed.
        2. Rewrite it into a stronger prompt.
        3. Ask two worker agents.
        4. Aggregate the best answer.
        """
    )
    st.divider()
    if st.button("New chat", use_container_width=True):
        reset_chat()
        st.rerun()


st.title("Multi-Agent Answer Aggregator")
st.caption(
    "Ask one question. Two worker agents answer independently, then an aggregator merges the result."
)

show_messages()


if st.session_state.current_step == "input":
    question = st.chat_input("Ask a question for the agent team")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.pending_question = question

        with st.status("Checking question clarity...", expanded=True) as status:

            def show_trace(message: str) -> None:
                """Show backend trace messages in the Streamlit status panel."""

                st.write(message)

            clarification = ask_for_clarification(
                question,
                trace_callback=show_trace,
            )
            status.update(label="Question clarity check complete.", state="complete")

        if clarification:
            st.session_state.clarification_question = clarification
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Clarification needed: {clarification}"}
            )
            st.session_state.current_step = "clarification"
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": "✅ Question is clear. Running agents."}
            )
            st.session_state.current_step = "processing"

        st.rerun()


elif st.session_state.current_step == "clarification":
    with st.form("clarification_form"):
        st.info(st.session_state.clarification_question)
        extra_context = st.text_input("Add context, or leave empty and skip.")
        submitted = st.form_submit_button("Continue")

    if submitted:
        if extra_context:
            st.session_state.pending_question = (
                f"{st.session_state.pending_question}\n\n"
                f"Extra context from user: {extra_context}"
            )
            st.session_state.messages.append(
                {"role": "user", "content": f"Extra context: {extra_context}"}
            )
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": "Continuing with the original question."}
            )

        st.session_state.current_step = "processing"
        st.rerun()


elif st.session_state.current_step == "processing":
    with st.status("Running the agent graph...", expanded=True) as status:

        def show_trace(message: str) -> None:
            """Show backend trace messages in the Streamlit status panel."""

            st.write(message)

        result = answer_question(
            st.session_state.pending_question,
            trace_callback=show_trace,
        )
        status.update(label="Done", state="complete", expanded=False)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["final_answer"]}
    )
    st.session_state.current_step = "input"
    st.rerun()
