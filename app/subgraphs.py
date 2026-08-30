import os
from typing import Dict, Any, Literal
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from app.state import GroceryAgentState
from app.tools import get_order_details, search_grocery_policy

load_dotenv()

# Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)

# ---------------------------------------------------------------------------
# SUBGRAPH 1: DELIVERY & COURIER SPECIALIST
# ---------------------------------------------------------------------------

delivery_llm = llm.bind_tools([get_order_details])

def delivery_analyst_node(state: GroceryAgentState) -> Dict[str, Any]:
    """Inspects courier transit, delays, and order timestamps."""
    print("  🚴 [Delivery Subgraph]: Analyzing delivery status...")
    system_prompt = SystemMessage(content=(
        "You are the Delivery & Courier Specialist for GroceryOnTheGo.\n"
        "Your sole focus is verifying order status, rider location, and transit timestamps.\n"
        "Use the `get_order_details` tool if you need order or courier info."
    ))
    
    response = delivery_llm.invoke([system_prompt] + state["messages"])
    return {
        "messages": [response],
        "internal_notes": ["Delivery Subgraph: Evaluated courier telemetry."]
    }

def delivery_tool_node(state: GroceryAgentState) -> Dict[str, Any]:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    tool_messages = []
    
    for call in tool_calls:
        if call["name"] == "get_order_details":
            output = get_order_details.invoke(call["args"])
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
            
    return {"messages": tool_messages}

def delivery_should_continue(state: GroceryAgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return "__end__"

def build_delivery_subgraph():
    sub = StateGraph(GroceryAgentState)
    sub.add_node("analyst", delivery_analyst_node)
    sub.add_node("tools", delivery_tool_node)
    
    sub.add_edge(START, "analyst")
    sub.add_conditional_edges("analyst", delivery_should_continue, {"tools": "tools", "__end__": END})
    sub.add_edge("tools", "analyst")
    return sub.compile()


# ---------------------------------------------------------------------------
# SUBGRAPH 2: QUALITY & REFUND POLICY SPECIALIST
# ---------------------------------------------------------------------------

quality_llm = llm.bind_tools([search_grocery_policy])

def quality_analyst_node(state: GroceryAgentState) -> Dict[str, Any]:
    """Inspects perishable damage, spoiled food rules, and refund windows."""
    print("  🥑 [Quality Subgraph]: Evaluating perishable damage & policy rules...")
    system_prompt = SystemMessage(content=(
        "You are the Quality & Perishables Specialist for GroceryOnTheGo.\n"
        "Your focus is reviewing customer claims about damaged produce, spoiled dairy, and missing groceries.\n"
        "Use `search_grocery_policy` to verify return windows and store credit rules."
    ))
    
    response = quality_llm.invoke([system_prompt] + state["messages"])
    return {
        "messages": [response],
        "internal_notes": ["Quality Subgraph: Evaluated perishable damage policy."]
    }

def quality_tool_node(state: GroceryAgentState) -> Dict[str, Any]:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    tool_messages = []
    
    for call in tool_calls:
        if call["name"] == "search_grocery_policy":
            output = search_grocery_policy.invoke(call["args"])
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
            
    return {"messages": tool_messages}

def quality_should_continue(state: GroceryAgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return "__end__"

def build_quality_subgraph():
    sub = StateGraph(GroceryAgentState)
    sub.add_node("analyst", quality_analyst_node)
    sub.add_node("tools", quality_tool_node)
    
    sub.add_edge(START, "analyst")
    sub.add_conditional_edges("analyst", quality_should_continue, {"tools": "tools", "__end__": END})
    sub.add_edge("tools", "analyst")
    return sub.compile()