# Multi-Agent Answer Aggregator
A helper blog post as your walkthrough for the code can be found [here](https://medium.com/@hosna_53144/your-first-agentic-ai-app-for-beginners-with-multi-agents-and-streamlit-ebca4213eb8b) </br>
## Intro
Beginner-friendly PoC for an agentic AI workflow:

1. The app checks whether the user's question needs clarification.
2. A prompt editor agent rewrites the question into a clearer task.
3. Two worker agents answer the same task independently.
4. An aggregator agent merges the strongest parts into one final answer.

## Files

- `agent_pipeline.py` contains the core LangGraph agent workflow.
- `ui.py` contains the Streamlit chat UI.
- `__init__.py` marks the folder as a Python package.

## Setup

Create a `.env` file with your Groq key:

```bash
GROQ_API_KEY=your_key_here
```

Install the needed packages:

```bash
pip install -r requirements.txt
```

## Run

Command-line demo:

```bash
python agent_pipeline.py
```

Streamlit UI:

```bash
streamlit run ui.py
```

if this failed, you can try: 
```bash
python -m streamlit run ui.py
```
