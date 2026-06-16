# 🦜🕸️ LangGraph — Agentic AI Workflow Laboratory

> A structured, hands-on exploration of **LangGraph** — from the simplest sequential chains to production-grade agents with persistent memory, human-in-the-loop control, and tool-calling capabilities.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green?logo=langchain)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange?logo=langchain)](https://github.com/langchain-ai/langchain)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-checkpointing-blue?logo=postgresql)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Concepts Covered](#-concepts-covered)
  - [Chatbot](#1-chatbot)
  - [Sequential Workflows](#2-sequential-workflows)
  - [Conditional Workflows](#3-conditional-workflows)
  - [Parallel Workflows](#4-parallel-workflows)
  - [Iterative Workflows](#5-iterative-workflows)
  - [Persistence](#6-persistence)
  - [Short-Term Memory](#7-short-term-memory)
  - [Long-Term Memory](#8-long-term-memory)
  - [Subgraphs](#9-subgraphs)
  - [Human-in-the-Loop (HITL)](#10-human-in-the-loop-hitl)
  - [Tools in LangGraph](#11-tools-in-langgraph)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [LLM Providers Used](#-llm-providers-used)
- [Tech Stack](#-tech-stack)
- [Learning Path](#-learning-path)
- [Contributing](#-contributing)

---

## 🌐 Overview

This repository is a **progressive learning lab** for LangGraph — the graph-based orchestration framework built on top of LangChain. Each folder isolates a specific concept or workflow pattern, implemented as Jupyter notebooks or standalone Python scripts, so you can study and run them independently.

The goal is to build intuition for:

- How **state graphs** encode agent logic as nodes and edges
- When to use **sequential vs. conditional vs. parallel** execution
- How to give agents **memory** — both within a session (short-term) and across sessions (long-term)
- How to keep humans in control with **interrupt-and-resume** (HITL)
- How to compose complex systems using **subgraphs**
- How to make agents take real-world actions with **tool calling**

---

## 📁 Repository Structure

```
LangGraph/
│
├── Chatbot/
│   └── v1.ipynb                        # Minimal LangGraph chatbot
│
├── Sequential Workflows/
│   ├── blog_prompt_chain.ipynb         # Multi-step blog generation chain
│   ├── bmi_workflow_agent.ipynb        # BMI calculator as a linear graph
│   └── uni_llm_agent.ipynb             # University info sequential agent
│
├── Conditional Workflows/
│   ├── customer_support.ipynb          # Intent-based routing agent
│   └── quadratic_workflow.ipynb        # Conditional branching on discriminant
│
├── Parallel Workflows/
│   ├── batsman.ipynb                   # Cricket batsman analysis in parallel
│   └── essay_workflow.ipynb            # Parallel essay section generation
│
├── Iterative Workflows/
│   └── Tweet_generator.ipynb           # Self-refining tweet writer with loop
│
├── Persistence/
│   ├── basic_application.ipynb         # MemorySaver-based state persistence
│   └── fault_tolerance.ipynb           # Recovering from failures with checkpointing
│
├── Short-Term-Memory/
│   ├── basic_implementation.ipynb      # In-thread conversation memory
│   ├── token-trimming.ipynb            # Managing context window with trimming
│   └── using_postgresql.ipynb          # PostgreSQL as the checkpointer backend
│
├── Long-Term-Memory/
│   ├── inmemorystore.ipynb             # Cross-session memory with InMemoryStore
│   ├── read-only-implementation.ipynb  # Reading stored memories in graph
│   ├── write-implementation.ipynb      # Writing memories during graph execution
│   ├── write-implement.py              # Script version of write implementation
│   ├── postgre-implement.py            # PostgreSQL-backed long-term memory
│   ├── merged.py                       # Combined read+write memory agent
│   └── prompt.py                       # Prompt template helpers
│
├── Subgraphs/
│   ├── isolated.ipynb                  # Subgraph with isolated state
│   └── shared.ipynb                    # Subgraph with shared parent state
│
├── hitl.ipynb                          # Human-in-the-Loop with interrupt/resume
├── terminal-chatbot-hitl.py            # Terminal HITL chatbot (runnable script)
├── tools_langgraph.ipynb               # Tool calling & ReAct-style agents
├── requirements.txt                    # All dependencies
└── .gitignore
```

---

## 🧠 Concepts Covered

### 1. Chatbot
**📂 `Chatbot/v1.ipynb`**

The entry point of the repo. Implements the most basic LangGraph chatbot — a single `chatbot` node connected to a `StateGraph` with `MessagesState`. Teaches:
- Defining a `StateGraph` and compiling it
- How `messages` are passed and accumulated through the graph state
- The `invoke()` / `stream()` interface

---

### 2. Sequential Workflows
**📂 `Sequential Workflows/`**

Linear chains where each node executes one after another — no branching.

| Notebook | Description |
|---|---|
| `blog_prompt_chain.ipynb` | A multi-stage blog pipeline: topic → outline → draft → polish, each handled by a separate LLM node |
| `bmi_workflow_agent.ipynb` | User inputs height/weight → BMI node calculates → category node classifies → advice node responds |
| `uni_llm_agent.ipynb` | University information agent that passes context through structured nodes |

**Key concepts:** `add_node`, `add_edge`, `START → node → END` patterns, typed `TypedDict` state.

---

### 3. Conditional Workflows
**📂 `Conditional Workflows/`**

Graphs with **dynamic routing** — the next node is chosen at runtime based on the current state.

| Notebook | Description |
|---|---|
| `customer_support.ipynb` | Classifies incoming queries (billing / technical / general) and routes to specialized handler nodes |
| `quadratic_workflow.ipynb` | Solves quadratic equations with branching: discriminant > 0 → two roots; = 0 → one root; < 0 → complex roots |

**Key concepts:** `add_conditional_edges`, router functions, `Literal` type hints for edge labels.

---

### 4. Parallel Workflows
**📂 `Parallel Workflows/`**

Graphs that fan out to multiple nodes simultaneously and merge results — the **map-reduce** pattern in LangGraph.

| Notebook | Description |
|---|---|
| `batsman.ipynb` | Analyses a cricket batsman's batting average, strike rate, and consistency in parallel, then synthesizes a report |
| `essay_workflow.ipynb` | Generates introduction, body paragraphs, and conclusion in parallel, then merges into a complete essay |

**Key concepts:** Fan-out edges, fan-in aggregation nodes, `Annotated` state fields with reducer functions (e.g. `operator.add`).

---

### 5. Iterative Workflows
**📂 `Iterative Workflows/`**

Agents that loop — repeatedly refining output until a condition is satisfied.

| Notebook | Description |
|---|---|
| `Tweet_generator.ipynb` | Generates a tweet, evaluates it against quality criteria (character limit, engagement), and rewrites in a loop until approved |

**Key concepts:** Cycle detection, conditional back-edges, iteration counters in state, loop termination via `END` routing.

---

### 6. Persistence
**📂 `Persistence/`**

How LangGraph preserves graph state across turns and recovers from failures.

| Notebook | Description |
|---|---|
| `basic_application.ipynb` | Full multi-turn chatbot using `MemorySaver` — state is snapshotted after every node via `thread_id` |
| `fault_tolerance.ipynb` | Demonstrates how checkpointing allows a graph to resume from the last successful node after a crash |

**Key concepts:** `MemorySaver`, `thread_id` in `config`, `get_state()`, `update_state()`, checkpoint inspection.

---

### 7. Short-Term Memory
**📂 `Short-Term-Memory/`**

Managing conversation history **within a single thread** — what the agent remembers during a session.

| Notebook | Description |
|---|---|
| `basic_implementation.ipynb` | Standard in-thread message accumulation with `MessagesState` |
| `token-trimming.ipynb` | Keeps context window bounded using `trim_messages` to drop oldest messages when approaching token limits |
| `using_postgresql.ipynb` | Swaps `MemorySaver` for a **PostgreSQL checkpointer** (`langgraph-checkpoint-postgres`) for durable, production-ready short-term memory |

**Key concepts:** `MessagesState`, `trim_messages`, `PostgresSaver`, connection pooling with `psycopg`.

---

### 8. Long-Term Memory
**📂 `Long-Term-Memory/`**

Memory that **persists across threads and sessions** — the agent remembers users between conversations.

| File | Description |
|---|---|
| `inmemorystore.ipynb` | Uses LangGraph's `InMemoryStore` to save and retrieve user facts across threads |
| `read-only-implementation.ipynb` | A graph that reads from the store at the start of each conversation to personalise responses |
| `write-implementation.ipynb` | A graph that extracts and writes new facts to the store during conversation |
| `merged.py` | Production-style agent combining both read and write memory in a single graph |
| `postgre-implement.py` | PostgreSQL-backed long-term memory store for durable cross-session recall |
| `prompt.py` | Reusable prompt templates incorporating retrieved memory |

**Key concepts:** `InMemoryStore`, namespaced memory (`("users", user_id)`), `store.put()` / `store.get()` / `store.search()`, memory extraction prompts.

---

### 9. Subgraphs
**📂 `Subgraphs/`**

Encapsulating complex logic into **reusable, composable sub-graphs** that plug into a parent graph.

| Notebook | Description |
|---|---|
| `isolated.ipynb` | Subgraph with its own private state — communicates with parent only through defined input/output keys |
| `shared.ipynb` | Subgraph that shares part of the parent state — reads and writes shared keys directly |

**Key concepts:** Compiling a subgraph separately, passing it as a node to a parent `StateGraph`, state schema inheritance and isolation boundaries.

---

### 10. Human-in-the-Loop (HITL)
**📂 `hitl.ipynb` · `terminal-chatbot-hitl.py`**

Agents that **pause mid-execution** and wait for a human to review, approve, or modify state before continuing.

| File | Description |
|---|---|
| `hitl.ipynb` | Notebook walkthrough: setting `interrupt_before=["node"]`, using `stream()` to pause, inspecting state, and calling `Command(resume=...)` to continue |
| `terminal-chatbot-hitl.py` | Fully runnable terminal chatbot with HITL — type your message, review the agent's planned action, approve or reject, and watch it proceed |

**Key concepts:** `interrupt_before`, `interrupt_after`, `graph.get_state()`, `graph.update_state()`, `Command(resume=value)`, human approval gates.

---

### 11. Tools in LangGraph
**📂 `tools_langgraph.ipynb`**

Equipping LangGraph agents with **real-world tools** — web search, calculators, APIs — and building ReAct-style tool-calling loops.

- Integrates **DuckDuckGo search** (`duckduckgo-search` / `ddgs`) for live web retrieval
- Demonstrates the `ToolNode` abstraction for automatic tool dispatch
- Shows the full **ReAct loop**: `agent → tools → agent → ...→ END`
- Tool binding with `llm.bind_tools([...])` and conditional routing on `tool_calls`

**Key concepts:** `@tool` decorator, `ToolNode`, `tools_condition`, binding tools to LLMs, handling `AIMessage` with `tool_calls`.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/HarshRaj4343/LangGraph.git
cd LangGraph

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

For PostgreSQL-based notebooks, ensure you have a running PostgreSQL instance and set the `DB_URI` environment variable (see below).

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
# LLM API Keys (use whichever providers you need)
GOOGLE_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key        # if using OpenAI models

# PostgreSQL (for persistence / short-term memory / long-term memory notebooks)
DB_URI=postgresql://user:password@localhost:5432/langgraph_db
```

All notebooks use `python-dotenv` to load these automatically.

---

## 🤖 LLM Providers Used

The notebooks are intentionally **provider-agnostic** and demonstrate multiple backends:

| Provider | Package | Used In |
|---|---|---|
| Google Gemini | `langchain-google-genai` | Most notebooks (primary) |
| Groq (Llama / Mixtral) | `langchain-groq` | Fast inference notebooks |
| Ollama (local models) | `langchain-ollama` | Offline / privacy-focused demos |
| HuggingFace | `langchain-huggingface` | Embedding & smaller LLM demos |

You can swap providers by changing a single import line — the graph logic stays identical.

---

## 🛠 Tech Stack

| Technology | Role |
|---|---|
| **LangGraph** | Core graph orchestration framework |
| **LangChain** | LLM abstractions, message types, tool utilities |
| **PostgreSQL** | Production checkpointer & long-term memory store |
| **psycopg3** | Async PostgreSQL driver (`psycopg[binary,pool]`) |
| **DuckDuckGo Search** | Live web search tool for agents |
| **Streamlit** | (Available) UI layer for agent demos |
| **Jupyter** | Interactive notebook environment for all examples |

---

## 🗺 Learning Path

If you're new to LangGraph, follow this order:

```
1. Chatbot/v1.ipynb                         ← Hello World
2. Sequential Workflows/bmi_workflow_agent  ← Linear graphs
3. Conditional Workflows/quadratic_workflow ← Routing logic
4. Parallel Workflows/essay_workflow        ← Fan-out / fan-in
5. Iterative Workflows/Tweet_generator      ← Loops & self-refinement
6. Persistence/basic_application            ← Multi-turn memory
7. Short-Term-Memory/token-trimming         ← Context management
8. Short-Term-Memory/using_postgresql       ← Production persistence
9. Long-Term-Memory/write-implementation    ← Cross-session memory
10. Subgraphs/isolated                      ← Composition patterns
11. hitl.ipynb                              ← Human oversight
12. tools_langgraph.ipynb                   ← Tool-calling agents
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add a new workflow pattern, fix a bug, or improve documentation:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push and open a Pull Request

Please keep each notebook focused on a single concept and include markdown cells explaining the key ideas.

---

<div align="center">
  <sub>Built with ❤️ while learning agentic AI at IIT Mandi</sub>
</div>
