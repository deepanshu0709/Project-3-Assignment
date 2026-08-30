import os
import sqlite3
from typing import Dict, Any, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.state import GroceryAgentState
from app.security import scan_for_prompt_injection, anonymize_pii
from app.subgraphs import build_delivery_subgraph, build_quality_subgraph

load_dotenv()

DB_PATH = "data/grocery_support.db"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)

# Compile subgraphs
delivery_subgraph = build_delivery_subgraph()
quality_subgraph = build_quality_subgraph()


# --- MASTER NODE 1: Ingress Guardrail ---
def master_ingress_node(state: GroceryAgentState) -> Dict[str, Any]:
    last_msg = state["messages"][-1].content
    
    if scan_for_prompt_injection(last_msg):
        return {
            "injection_attempt_detected": True,
            "final_response": "Security Alert: Invalid request pattern.",
            "internal_notes": ["Master Ingress: Blocked injection attempt."]
        }
        
    sanitized_text, pii_found = anonymize_pii(last_msg)
    return {
        "messages": [HumanMessage(content=sanitized_text)],
        "pii_detected": pii_found,
        "injection_attempt_detected": False,
        "internal_notes": [f"Master Ingress: Sanitized ticket (PII: {pii_found})"]
    }


# --- MASTER NODE 2: Triage Classification ---
class TriageDecision(BaseModel):
    category: Literal["delivery", "quality_refund"] = Field(
        description="Route to delivery for rider/delay issues, or quality_refund for spoiled/missing items."
    )

def master_triage_node(state: GroceryAgentState) -> Dict[str, Any]:
    print("🧭 [Master Graph]: Triaging ticket to appropriate subgraph...")
    last_user_msg = state["messages"][-1].content
    
    structured_llm = llm.with_structured_output(TriageDecision)
    decision: TriageDecision = structured_llm.invoke([
        SystemMessage(content="You are Master Triage. Route the customer's grocery issue to either 'delivery' or 'quality_refund'."),
        HumanMessage(content=last_user_msg)
    ])
    
    return {
        "ticket_category": decision.category,
        "internal_notes": [f"Master Triage routed to {decision.category} subgraph."]
    }


# --- MASTER NODE 3: Synthesizer Node ---
def master_synthesizer_node(state: GroceryAgentState) -> Dict[str, Any]:
    """Unifies notes and messages from the subgraphs into a polished response."""
    print("✨ [Master Synthesizer]: Formulating customer response from subgraph findings...")
    notes_context = "\n".join(state.get("internal_notes", []))
    
    prompt = [
        SystemMessage(content=(
            "You are the senior support lead for GroceryOnTheGo.\n"
            f"Internal specialist findings:\n{notes_context}\n\n"
            "Formulate a direct, helpful, and empathetic answer for the customer based on the conversation history."
        )),
        HumanMessage(content=state["messages"][-1].content)
    ]
    
    response = llm.invoke(prompt)
    return {
        "messages": [response],
        "final_response": response.content,
        "internal_notes": ["Master Synthesizer: Customer response generated."]
    }


# --- ROUTING LOGIC ---
def route_to_subgraph(state: GroceryAgentState) -> str:
    if state.get("ticket_category") == "delivery":
        return "delivery_subgraph"
    return "quality_subgraph"


# --- COMPOSE MASTER GRAPH ---
def build_master_graph(checkpointer: SqliteSaver):
    master = StateGraph(GroceryAgentState)
    
    # Register master nodes and subgraphs
    master.add_node("ingress", master_ingress_node)
    master.add_node("triage", master_triage_node)
    master.add_node("delivery_subgraph", delivery_subgraph)
    master.add_node("quality_subgraph", quality_subgraph)
    master.add_node("synthesizer", master_synthesizer_node)
    
    # Define execution flow
    master.add_edge(START, "ingress")
    master.add_edge("ingress", "triage")
    
    # Conditional branching into subgraphs
    master.add_conditional_edges(
        "triage",
        route_to_subgraph,
        ["delivery_subgraph", "quality_subgraph"]
    )
    
    # Subgraphs converge back into the master synthesizer
    master.add_edge("delivery_subgraph", "synthesizer")
    master.add_edge("quality_subgraph", "synthesizer")
    master.add_edge("synthesizer", END)
    
    return master.compile(checkpointer=checkpointer)


#verification

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_master_graph(checkpointer)
    
    print("\n========================================================")
    print("Test Scenario: Spoiled Strawberries & Warm Milk (ORD-8821)")
    print("========================================================")
    config = {"configurable": {"thread_id": "SUBGRAPH-TEST-101"}}
    
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Hello, my order ORD-8821 arrived 20 minutes ago. "
                            "The milk carton is warm and leaking, and the strawberries are completely mouldy. What can you do?"
                )
            ],
            "customer_id": "CUST-101",
            "thread_id": "SUBGRAPH-TEST-101",
            "iteration_count": 0,
            "tool_call_fingerprints": [],
            "internal_notes": []
        },
        config=config
    )
    
    print("\n--- Final Customer Response ---")
    print(result["final_response"])
    
    print("\n--- Internal Scratchpad (Aggregated Across Subgraphs) ---")
    for note in result.get("internal_notes", []):
        print(f"• {note}")