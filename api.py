import os
import sqlite3
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.session7_subgraphs import build_master_graph

load_dotenv()

app = FastAPI(title="GroceryOnTheGo Support Agent - Session 7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "data/grocery_support.db"
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = build_master_graph(checkpointer)


class TicketRequest(BaseModel):
    ticket: str
    thread_id: Optional[str] = "SESSION-7-DEFAULT"
    customer_id: Optional[str] = "CUST-101"


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FileResponse("index.html")


@app.post("/api/run")
async def run_ticket(req: TicketRequest):
    """Executes a customer inquiry through the Session 7 Multi-Agent Hierarchical Graph."""
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # Check if this thread already has existing history in SQLite
    existing_history = list(graph.get_state_history(config))
    
    if len(existing_history) == 0:
        initial_input = {
            "messages": [HumanMessage(content=req.ticket)],
            "customer_id": req.customer_id,
            "thread_id": req.thread_id,
            "iteration_count": 0,
            "tool_call_fingerprints": [],
            "internal_notes": []
        }
    else:
        # Follow-up turn: SQLite checkpointer reloads previous context
        initial_input = {
            "messages": [HumanMessage(content=req.ticket)]
        }

    result = graph.invoke(initial_input, config=config)
    
    return {
        "thread_id": req.thread_id,
        "category": result.get("ticket_category", "general"),
        "pii_detected": result.get("pii_detected", False),
        "injection_blocked": result.get("injection_attempt_detected", False),
        "final_response": result.get("final_response", ""),
        "internal_notes": result.get("internal_notes", [])
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "completed_milestones": "Sessions 1 through 7", "domain": "GroceryOnTheGo"}
