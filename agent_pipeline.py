"""Chat interface backend for a research-assistance multi-agent aggregator.

On receival of user's query, this file runs the core agentic AI pipeline for
the project. It is intentionally kept separate from the Streamlit UI so
beginners can read the agent workflow without also reading layout code.

Pipeline scope:
1. Check whether the user's question needs clarification.
2. Rewrite the question into a clearer prompt for worker agents.
3. Send the prompt to two different LLM agents (can be increased)
4. Ask an aggregator agent to merge the strongest parts from each agent into one cohesive answer.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph


load_dotenv()


class AgentState(TypedDict, total=False):
    """Standard Shared memory in LangGraph that moves between graph nodes.

    Each node reads one or more fields and returns only the fields it updates.
    LangGraph merges those updates into the state.
    """

    question: str
    refined_question: str
    llama_answer: str
    qwen_answer: str
    final_answer: str


TraceCallback = Callable[[str], None]
"""Optional function used only to show progress messages to users.

The agent pipeline still works without tracing. Traces are added for better
user experience in the UI and terminal demo, so users can see which step is
currently running instead of waiting on a silent app.
"""


LLAMA_MODEL_NAME = "llama-3.3-70b-versatile"
QWEN_MODEL_NAME = "qwen/qwen3-32b"
AGGREGATOR_MODEL_NAME = "openai/gpt-oss-120b"


def remove_thinking_blocks(text: str) -> str:
    """Remove hidden reasoning blocks that some models may include."""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def emit_trace(trace_callback: TraceCallback | None, message: str) -> None:
    """Send an optional status message to the UI or terminal.

    If no trace callback is provided, this function does nothing. The trace
    messages are not required for the agents to reason or produce an answer.
    """

    if trace_callback:
        trace_callback(message)


@lru_cache(maxsize=1)  # Create model clients once, then reuse them across calls.
def get_models():
    """Create the LLM clients used by the project.
    The project uses two open-source models through Groq for the worker agents.
    The aggregator uses GPT-OSS through Groq so the final judge is different
    from both worker agents while still using one provider and one API key.
    
    In production settings and to get better results, you'll need a paid version
    of APIs as OpenAI, Anthropic, or any provider.
    """

    llama = init_chat_model(
        LLAMA_MODEL_NAME,
        model_provider="groq",
        temperature=0,
    )
    qwen = init_chat_model(
        QWEN_MODEL_NAME,
        model_provider="groq",
        temperature=0,
    )
    aggregator = init_chat_model(
        AGGREGATOR_MODEL_NAME,
        model_provider="groq",
        temperature=0,
    )
    return llama, qwen, aggregator


def ask_for_clarification(
    question: str,
    trace_callback: TraceCallback | None = None,
) -> str:
    """Return one short clarification question as follow-up for the user's query
    or an empty string if the query is clear enough for the worker agents. 

    This function is kept outside the graph because a UI needs to pause and let
    the user answer before the rest of the agent pipeline runs.

    The main aggregator model makes this decision because it later judges the
    final answer. That keeps "is this query answerable?" and "is the final
    answer good enough?" under the same quality standard.

    For this PoC, clarification is intentionally prioritized to be made so the feature 
    is easy to see during a demo. In production, make this less strict to avoid 
    annoying users with unnecessary follow-up questions.
    """

    system_prompt = """
    You are the intake assistant for a research-assistance chat interface.
    Decide whether the user's question is ready for a multi-agent answer.

    This is a PoC demo, so prioritize showing the clarification feature.
    Ask one clarification question when extra context would noticeably improve
    the answer, even if you could still give a generic first answer.

    If the question is already specific and actionable, answer exactly:
    CLEAR

    If a follow-up would improve the answer, answer exactly:
    CLARIFY: <one short question that asks for the most important missing detail>

    Good PoC reasons to ask:
    - The topic is missing.
    - The user's goal is missing.
    - The expected output format is missing.
    - The domain, audience, constraints, or success criteria are unclear.

    Production note:
    In a real product, this policy should be less strict. Too many
    clarification questions can annoy users and slow down the experience.
    """

    emit_trace(trace_callback, "🔎 Checking if the question needs clarification (GPT-OSS).")

    _, _, aggregator_model = get_models()
    response = aggregator_model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
    )
    content = response.content.strip()

    if content.upper().startswith("CLARIFY:"):
        clarification = content.split(":", 1)[1].strip()
    else:
        clarification = ""

    if clarification:
        emit_trace(trace_callback, f"❓ Clarification needed: {clarification}")
    else:
        emit_trace(trace_callback, "✅ Question is clear. No clarification needed.")

    return clarification


def refine_question(
    state: AgentState,
    trace_callback: TraceCallback | None = None,
) -> AgentState:
    """Use GPT-OSS to rewrite the user question for the worker agents.

    The same main model that checks clarification and aggregates the final
    answer also prepares the task. This keeps the pipeline's quality standard
    consistent before and after the worker agents respond.
    """

    system_prompt = """
    You are a prompt editor for a beginner-friendly multi-agent assistant.
    Rewrite the user's question into a clear task.

    Keep the original intent. Add only structure:
    - role
    - context
    - exact output expected

    Do not add facts that the user did not provide. Output only the rewritten
    prompt, not an explanation of your rewrite.
    """

    emit_trace(trace_callback, "🧭 Refining the question into a stronger task (GPT-OSS).")

    _, _, aggregator_model = get_models()
    response = aggregator_model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["question"]),
        ]
    )
    emit_trace(trace_callback, "✅ Refined question is ready for worker agents.")
    return {"refined_question": response.content.strip()}


def ask_llama_agent(
    state: AgentState,
    trace_callback: TraceCallback | None = None,
) -> AgentState:
    """Ask the Llama worker agent (first worker) for an independent answer."""

    system_prompt = """
    You are Worker Agent 1 in a research-assistance chat pipeline.
    Give a direct, useful answer to the refined task.

    Rules:
    - State assumptions briefly when needed.
    - Avoid unsupported claims.
    - Prefer clear structure over long prose.
    """

    try:
        emit_trace(trace_callback, "🤖 Asking LLM 1: Llama.")
        llama_model, _, _ = get_models()
        response = llama_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["refined_question"]),
            ]
        )
        emit_trace(trace_callback, "✅ LLM 1: Llama answered.")
        return {"llama_answer": response.content.strip()}
    except Exception as error:
        emit_trace(trace_callback, f"⚠️ LLM 1: Llama failed: {error}")
        return {"llama_answer": f"Llama agent error: {error}"}


def ask_qwen_agent(
    state: AgentState,
    trace_callback: TraceCallback | None = None,
) -> AgentState:
    """Ask the Qwen worker agent (second agent) for an independent answer."""

    system_prompt = """
    You are Worker Agent 2 in a research-assistance chat pipeline.
    Give an independent second answer to the refined task.

    Rules:
    - Look for missing assumptions or weak reasoning.
    - Avoid copying the style of another agent.
    - Keep the answer concise and evidence-aware.
    """

    try:
        emit_trace(trace_callback, "🤖 Asking LLM 2: Qwen.")
        _, qwen_model, _ = get_models()
        response = qwen_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["refined_question"]),
            ]
        )
        emit_trace(trace_callback, "✅ LLM 2: Qwen answered.")
        return {"qwen_answer": remove_thinking_blocks(response.content)}
    except Exception as error:
        emit_trace(trace_callback, f"⚠️ LLM 2: Qwen failed: {error}")
        return {"qwen_answer": f"Qwen agent error: {error}"}


# To add more worker LLMs:
# 1. Add a new answer field to AgentState, such as "mistral_answer".
# 2. Add the model client in get_models().
# 3. Create a function like ask_mistral_agent().
# 4. Add the new node and edges in build_graph().
# 5. Include the new answer in aggregate_answers().


def aggregate_answers(
    state: AgentState,
    trace_callback: TraceCallback | None = None,
) -> AgentState:
    """Merge worker agents' answers into one final response.

    GPT-OSS is used as a separate aggregator agent. It should not reveal model
    names or private critique. It should use disagreement as a signal to be
    careful, not as content to show the user.
    """

    prompt = f"""
    Original question:
    {state["question"]}

    Refined task:
    {state["refined_question"]}

    Worker answer 1:
    {state["llama_answer"]}

    Worker answer 2:
    {state["qwen_answer"]}

    Write one final answer for the user.

    Rules:
    - Keep the answer direct and useful.
    - Merge the strongest points from both workers.
    - Remove claims that are unsupported, contradictory, or too speculative.
    - Do not mention worker agents, model names, or internal critique.
    - If the workers disagree, use the more cautious answer.
    - If both workers are uncertain, say what information is missing.
    - Output only the final answer.
    """

    emit_trace(trace_callback, "🧩 Aggregating worker answers (GPT-OSS).")

    _, _, aggregator_model = get_models()
    response = aggregator_model.invoke([HumanMessage(content=prompt)])
    emit_trace(trace_callback, "✅ Final aggregated answer is ready.")
    return {"final_answer": response.content.strip()}


def build_graph():
    """Build the LangGraph workflow that fans out to workers and fans back in."""

    workflow = StateGraph(AgentState)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("llama_agent", ask_llama_agent)
    workflow.add_node("qwen_agent", ask_qwen_agent)
    workflow.add_node("aggregate_answers", aggregate_answers)

    workflow.set_entry_point("refine_question")
    workflow.add_edge("refine_question", "llama_agent")
    workflow.add_edge("refine_question", "qwen_agent")
    workflow.add_edge("llama_agent", "aggregate_answers")
    workflow.add_edge("qwen_agent", "aggregate_answers")
    workflow.add_edge("aggregate_answers", END)

    return workflow.compile()


AGENT_GRAPH = build_graph()


def answer_question(
    question: str,
    trace_callback: TraceCallback | None = None,
) -> AgentState:
    """Run the full multi-agent pipeline for one user question.

    The graph object above shows the LangGraph structure. This function runs the
    same steps with optional trace messages so the UI and terminal can show
    progress. If tracing is disabled, the pipeline still runs normally.
    """

    state: AgentState = {"question": question}
    state.update(refine_question(state, trace_callback))

    emit_trace(trace_callback, "🚦 Dispatching worker agents.")
    if trace_callback:
        # Streamlit UI updates are safer when emitted from the main thread, so
        # traced runs call workers one after another to keep progress visible.
        state.update(ask_llama_agent(state, trace_callback))
        state.update(ask_qwen_agent(state, trace_callback))
    else:
        # ThreadPoolExecutor runs both independent worker-agent calls at the
        # same time. This is useful here because Llama and Qwen do not depend on
        # each other; both only need the refined question.
        with ThreadPoolExecutor(max_workers=2) as executor:
            llama_future = executor.submit(ask_llama_agent, state)
            qwen_future = executor.submit(ask_qwen_agent, state)
            state.update(llama_future.result())
            state.update(qwen_future.result())

    state.update(aggregate_answers(state, trace_callback))
    return state


def run_cli() -> None:
    """Run the pipeline from the terminal instead of Streamlit.

    This is useful for testing the backend alone. It asks for one question,
    optionally asks for clarification, runs the LangGraph pipeline, and prints
    the final aggregated answer. The Streamlit app uses the same backend
    functions, so this CLI is only a small debugging entry point.
    """

    question = input("Ask a question: ").strip()
    if not question:
        print("No question provided.")
        return

    def print_trace(message: str) -> None:
        """Print trace messages for terminal users."""

        print(f"[trace] {message}")

    clarification = ask_for_clarification(question, trace_callback=print_trace)
    if clarification:
        print(f"\nClarification needed: {clarification}")
        answer = input("Your extra context, or press Enter to skip: ").strip()
        if answer:
            question = f"{question}\n\nExtra context from user: {answer}"

    result = answer_question(question, trace_callback=print_trace)
    print("\nFinal answer:\n")
    print(result["final_answer"])


if __name__ == "__main__":
    run_cli()
