# 🥦 GroceryOnTheGo — Enterprise AI Support Swarm (Phase 3)

**Domain Selected:** E-Commerce Quick-Commerce Order & Refund Support (GroceryOnTheGo)  
**Curriculum Scope:** Built incrementally through **Session 7 (Multi-Agent Subgraphs & Master Topologies)**.

A production-grade, stateful AI support platform built with **LangGraph**, **Google Gemini 2.5 Flash**, and **FastAPI**. The platform safely handles quick-commerce grocery delivery inquiries, spoiled perishable claims, order tracking, and security filtering.

---

## Architectural Progression Log (Sessions 1–7)

| Milestone | Architecture Added | Status | Description |
| :--- | :--- | :---: | :--- |
| **Session 1** | **The Blueprint** | ✅ **Live** | Typed `GroceryAgentState` schema, Gemini triage classifier (`triage_classifier_node`), and deterministic routing. |
| **Session 2** | **Tool Binding** | ✅ **Live** | `get_order_details` (CRM/courier lookup) and `search_grocery_policy` bound to Gemini via LangGraph `ToolNode`. |
| **Session 3** | **ReAct Loop & Circuit Breakers** | ✅ **Live** | Full ReAct loop bounded by `MAX_ITERATIONS = 5` and SHA-256 tool-call argument fingerprinting to stop loops. |
| **Session 4** | **Persistence & Thread Isolation** | ✅ **Live** | Checkpointing with SQLite (`SqliteSaver`), thread isolation via `thread_id`, and multi-turn state preservation. |
| **Session 5** | **Context Management** | ✅ **Live** | Rolling summarizer (`summarization_node`) with `RemoveMessage` pruning when message count exceeds 8. |
| **Session 6** | **Guardrails Sandwich** | ✅ **Live** | Microsoft Presidio PII anonymizer + 14 regex prompt-injection heuristics (0-token rejection) at ingress, plus egress safety verification. |
| **Session 7** | **Multi-Agent Topologies** | ✅ **Live** | Modular, independently compiled `delivery_subgraph` and `quality_subgraph` wired inside a Master Graph with `operator.add` reducers. |
| *Session 8–12* | *Supervisor Swarm, HITL & Time Travel* | 🔄 *Future Roadmap* | Hub-and-Spoke supervisor, parallel Send API, human interrupt breakpoints, and time-travel forensics. |

---

## Tech Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Agent Framework** | LangGraph (`>=0.2.70`) | State machine, compiled subgraphs, and conditional routing |
| **LLM Engine** | Google Gemini 2.5 Flash | Structured triage classification and specialist reasoning |
| **Persistence** | `langgraph-checkpoint-sqlite` | Thread-isolated checkpoint ledger in SQLite |
| **Security Layer** | Presidio Analyzer / Anonymizer | Entity masking (Phone, Email, Credit Cards) + 14 regex injection patterns |
| **Backend & API** | FastAPI + Uvicorn | REST endpoints (`/api/run`, `/health`) and web UI serving |
| **Frontend** | Vanilla HTML5 / CSS3 / JS | Dark glassmorphism dashboard with execution inspector |

---

## 15-Minute Local Setup & Reproduction Guide

### 1. Clone & Enter Repository
```bash
git clone <YOUR-GITHUB-REPO-URL>
cd grocery-on-the-go
2. Create and Activate Virtual Environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
3. Install Dependencies & Download spaCy Model
pip install -r requirements.txt
python -m spacy download en_core_web_sm
4. Configure Environment Variables
Copy .env.example to .env:
cp .env.example .env
Inside .env, insert your Google Gemini API key:
GEMINI_API_KEY="your_actual_gemini_api_key_here"
5. Launch the Application
python -m uvicorn api:app --reload --port 8000
Open http://localhost:8000 in your browser.
Automated Verification Suite
Run the full Session 7 multi-agent test suite directly from the CLI:
python -m app.session7_subgraphs
Verification Checks:
PII Masking: Customer phone and email are masked before reaching the model.
Injection Defense: Adversarial prompts are caught and blocked at ingress with zero LLM tokens spent.
Subgraph Routing: Perishable quality complaints route into the Quality Subgraph and look up return policies.
State Reducer Durability: internal_notes preserves audit traces across both subgraphs via operator.add.
