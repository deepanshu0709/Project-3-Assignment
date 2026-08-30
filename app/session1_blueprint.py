import os
from typing import Literal, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from app.state import GroceryAgentState

# Load GEMINI_API_KEY from .env
load_dotenv()

# Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)

# Define structured output schema for deterministic routing
class TicketClassification(BaseModel):
    category: Literal["delivery", "quality_refund", "fraud", "general"] = Field(
        description="Classify the grocery support request into exactly one category."
    )
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Urgency level based on perishable goods or courier status."
    )
    reasoning: str = Field(description="One-sentence explanation for the classification.")


# --- NODE 1: Triage Classifier ---
def triage_classifier_node(state: GroceryAgentState) -> Dict[str, Any]:
    """Reads the user message and classifies category and urgency."""
    last_msg = state["messages"][-1].content
    
    prompt = [
        SystemMessage(content=(
            "You are the senior triage dispatcher for 'GroceryOnTheGo', a 20-minute quick-commerce grocery delivery service.\n"
            "Categories:\n"
            "- 'delivery': courier delays, wrong address, rider tracking issues\n"
            "- 'quality_refund': spoiled produce, leaking milk, damaged or missing grocery items\n"
            "- 'fraud': abusive refund threats, suspicious demands for unverified credits\n"
            "- 'general': general questions about store hours, membership, app usage"
        )),
        HumanMessage(content=last_msg)
    ]
    
    structured_classifier = llm.with_structured_output(TicketClassification)
    decision: TicketClassification = structured_classifier.invoke(prompt)
    
    note = f"Triage: {decision.category.upper()} (Urgency: {decision.urgency}) - {decision.reasoning}"
    print(f"\n🔍 [Triage Classifier Node Executed]: {note}")
    
    return {
        "ticket_category": decision.category,
        "urgency_level": decision.urgency,
        "internal_notes": [note]
    }


# --- HANDLER STUB NODES ---
def delivery_handler(state: GroceryAgentState) -> Dict[str, Any]:
    print("📍 [Delivery Handler Stub Executed]")
    return {
        "final_response": "Delivery Team: We have checked on your courier. Your order is being prioritized.",
        "internal_notes": ["Handled by Delivery Team Stub"]
    }

def quality_refund_handler(state: GroceryAgentState) -> Dict[str, Any]:
    print("🥑 [Quality & Refund Handler Stub Executed]")
    return {
        "final_response": "Quality Team: We apologize for the damaged grocery items. Reviewing refund eligibility.",
        "internal_notes": ["Handled by Quality & Refund Team Stub"]
    }

def fraud_handler(state: GroceryAgentState) -> Dict[str, Any]:
    print("🛡️ [Fraud & Risk Handler Stub Executed]")
    return {
        "final_response": "Support Supervisor: Your inquiry has been routed to our verification unit.",
        "internal_notes": ["Handled by Fraud Handler Stub"]
    }

def general_handler(state: GroceryAgentState) -> Dict[str, Any]:
    print("ℹ️ [General Handler Stub Executed]")
    return {
        "final_response": "Customer Support: Thank you for contacting GroceryOnTheGo. How else can we help?",
        "internal_notes": ["Handled by General Handler Stub"]
    }


# --- ROUTING LOGIC ---
def route_ticket(state: GroceryAgentState) -> str:
    category = state.get("ticket_category")
    if category == "delivery":
        return "delivery_handler"
    elif category == "quality_refund":
        return "quality_refund_handler"
    elif category == "fraud":
        return "fraud_handler"
    return "general_handler"


# --- COMPILE SESSION 1 GRAPH ---
def build_session1_graph():
    builder = StateGraph(GroceryAgentState)
    
    # Add Nodes
    builder.add_node("triage", triage_classifier_node)
    builder.add_node("delivery_handler", delivery_handler)
    builder.add_node("quality_refund_handler", quality_refund_handler)
    builder.add_node("fraud_handler", fraud_handler)
    builder.add_node("general_handler", general_handler)
    
    # Add Edges
    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        route_ticket,
        ["delivery_handler", "quality_refund_handler", "fraud_handler", "general_handler"]
    )
    builder.add_edge("delivery_handler", END)
    builder.add_edge("quality_refund_handler", END)
    builder.add_edge("fraud_handler", END)
    builder.add_edge("general_handler", END)
    
    return builder.compile()


if __name__ == "__main__":
    app_graph = build_session1_graph()
    
    # Test ticket 1: Rotten produce
    test_state = {
        "messages": [HumanMessage(content="The organic milk was warm and leaking, and the strawberries are mouldy!")],
        "customer_id": "CUST-101",
        "thread_id": "SESSION-1-DEMO",
        "internal_notes": []
    }
    
    result = app_graph.invoke(test_state)
    print("\n--- Final Graph Output ---")
    print(f"Assigned Category: {result['ticket_category']}")
    print(f"Urgency Level:     {result['urgency_level']}")
    print(f"Response:          {result['final_response']}")