import os
import hashlib
from typing import Dict, Any, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from app.state import GroceryAgentState
from app.tools import SESSION_2_TOOLS, get_order_details, search_grocery_policy

load_dotenv()

# Safety thresholds
MAX_ITERATIONS = 5

# Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)
llm_with_tools = llm.bind_tools(SESSION_2_TOOLS)

# Tool lookup dictionary for manual guarded execution
TOOL_MAP = {
    "get_order_details": get_order_details,
    "search_grocery_policy": search_grocery_policy
}


def make_call_fingerprint(tool_name: str, tool_args: dict) -> str:
    """Generates a deterministic SHA-256 fingerprint of a tool call."""
    raw = f"{tool_name}:{sorted(tool_args.items())}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# --- NODE 1: Guarded ReAct Agent ---
def react_agent_node(state: GroceryAgentState) -> Dict[str, Any]:
    current_iter = state.get("iteration_count", 0) + 1
    print(f"\n🤖 [ReAct Agent Step] - Iteration {current_iter}/{MAX_ITERATIONS}")
    
    system_prompt = SystemMessage(content=(
        "You are the senior resolution specialist for GroceryOnTheGo.\n"
        "You have tools to check order details and store policies.\n"
        "Investigate the customer's issue step by step. When you have sufficient information, provide a final helpful answer."
    ))
    
    # Keep the system prompt + last 6 messages to prevent runaway context
    recent_messages = state["messages"][-6:] if len(state["messages"]) > 6 else state["messages"]
    full_messages = [system_prompt] + recent_messages
    
    response = llm_with_tools.invoke(full_messages)
    
    return {
        "messages": [response],
        "iteration_count": current_iter,
        "internal_notes": [f"ReAct iteration {current_iter} executed."]
    }


# --- NODE 2: Guarded Tool Execution Node ---
def guarded_tool_node(state: GroceryAgentState) -> Dict[str, Any]:
    """
    Executes tool calls with duplicate prevention.
    If an identical tool call was already run, it skips execution and returns cached/warning feedback.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    
    existing_fingerprints = set(state.get("tool_call_fingerprints", []))
    new_fingerprints = []
    tool_messages = []
    notes = []
    
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        call_id = call["id"]
        
        fingerprint = make_call_fingerprint(tool_name, tool_args)
        
        # Check for duplicate execution
        if fingerprint in existing_fingerprints:
            print(f"⚠️ [Duplicate Guard Triggered]: Skipping identical call to '{tool_name}' with args {tool_args}")
            tool_messages.append(
                ToolMessage(
                    content=f"DUPLICATE_CALL_DETECTED: You already queried '{tool_name}' with these exact parameters. Do not repeat this query.",
                    tool_call_id=call_id
                )
            )
            notes.append(f"Blocked duplicate call: {tool_name}")
        else:
            print(f"🛠️ [Executing Tool]: {tool_name}({tool_args})")
            tool_fn = TOOL_MAP.get(tool_name)
            if tool_fn:
                tool_output = tool_fn.invoke(tool_args)
            else:
                tool_output = f"Tool '{tool_name}' is not registered."
            
            tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=call_id))
            new_fingerprints.append(fingerprint)
            notes.append(f"Executed {tool_name} (FP: {fingerprint})")
            
    return {
        "messages": tool_messages,
        "tool_call_fingerprints": new_fingerprints,
        "internal_notes": notes
    }


# --- NODE 3: Fallback Circuit Breaker Node ---
def circuit_breaker_node(state: GroceryAgentState) -> Dict[str, Any]:
    """Invoked when maximum iterations are exceeded without resolution."""
    print("🚨 [Circuit Breaker Triggered]: Max iterations exceeded. Halting loop safely.")
    fallback_message = AIMessage(
        content=(
            "I apologize, but I am taking longer than expected to resolve your grocery inquiry. "
            "I have escalated your ticket to our senior human dispatch team with all collected diagnostic notes."
        )
    )
    return {
        "messages": [fallback_message],
        "final_response": fallback_message.content,
        "internal_notes": ["Circuit breaker tripped: MAX_ITERATIONS reached."]
    }


# --- CONDITIONAL ROUTING WITH SAFETY CHECK ---
def should_continue(state: GroceryAgentState) -> Literal["guarded_tools", "circuit_breaker", "__end__"]:
    # 1. Check Circuit Breaker threshold
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "circuit_breaker"
        
    last_msg = state["messages"][-1]
    
    # 2. Check if agent wants to execute more tools
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "guarded_tools"
        
    # 3. Otherwise, agent has produced final text
    return "__end__"


# --- BUILD SESSION 3 GRAPH ---
def build_session3_graph():
    builder = StateGraph(GroceryAgentState)
    
    # Add Nodes
    builder.add_node("react_agent", react_agent_node)
    builder.add_node("guarded_tools", guarded_tool_node)
    builder.add_node("circuit_breaker", circuit_breaker_node)
    
    # Define Flow
    builder.add_edge(START, "react_agent")
    
    builder.add_conditional_edges(
        "react_agent",
        should_continue,
        {
            "guarded_tools": "guarded_tools",
            "circuit_breaker": "circuit_breaker",
            "__end__": END
        }
    )
    
    # After executing tools, return to agent to reason about the results
    builder.add_edge("guarded_tools", "react_agent")
    builder.add_edge("circuit_breaker", END)
    
    return builder.compile()


if __name__ == "__main__":
    app_graph = build_session3_graph()
    
    # Test Ticket: Multiple steps (Look up order + Check policy on spoiled perishables)
    test_state = {
        "messages": [
            HumanMessage(
                content="I ordered ORD-8821 25 minutes ago. The milk is spoiled and warm. "
                        "What is your policy for damaged milk and can I get a refund?"
            )
        ],
        "customer_id": "CUST-101",
        "thread_id": "SESSION-3-DEMO",
        "iteration_count": 0,
        "tool_call_fingerprints": [],
        "internal_notes": []
    }
    
    print("\n==========================================")
    print("Running Session 3 ReAct Loop with Guards")
    print("==========================================")
    
    result = app_graph.invoke(test_state)
    
    print("\n--- Final Conversation Transcript ---")
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"\n👤 Customer: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"\n🤖 Agent Tool Call: {[tc['name'] for tc in msg.tool_calls]}")
        elif isinstance(msg, ToolMessage):
            print(f"🛠️ Tool Result: {msg.content[:90]}...")
        elif isinstance(msg, AIMessage):
            print(f"\n🤖 Agent Response:\n{msg.content}")
            
    print("\n--- Diagnostic Trace Notes ---")
    for note in result.get("internal_notes", []):
        print(f"• {note}")