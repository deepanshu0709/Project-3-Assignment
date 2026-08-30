import os
import sqlite3
from typing import Dict, Any, Literal
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    RemoveMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.state import GroceryAgentState
from app.tools import SESSION_2_TOOLS, get_order_details, search_grocery_policy

load_dotenv()

SUMMARY_THRESHOLD = 8  # Trigger compression when messages exceed this count
DB_PATH = "data/grocery_support.db"

# Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)
llm_with_tools = llm.bind_tools(SESSION_2_TOOLS)

TOOL_MAP = {
    "get_order_details": get_order_details,
    "search_grocery_policy": search_grocery_policy
}


# --- NODE 1: Rolling Summarizer Node ---
def summarization_node(state: GroceryAgentState) -> Dict[str, Any]:
    """
    Condenses the oldest messages into state['system_summary']
    and emits RemoveMessage commands to prune them from SQLite.
    """
    messages = state["messages"]
    existing_summary = state.get("system_summary") or "No prior summary."
    
    # Take the oldest slice of messages to summarize (keep the latest 4 untouched)
    messages_to_summarize = messages[:-4]
    
    summary_prompt = [
        SystemMessage(content=(
            "You are a conversation summarization engine for GroceryOnTheGo support.\n"
            f"Current summary: {existing_summary}\n"
            "Compress the older customer support interactions above into a concise, factual summary. "
            "Preserve critical facts: Order IDs, specific damaged items, courier notes, and policy quotes. "
            "Do not exceed 3-4 sentences."
        )),
        HumanMessage(content=f"Older messages to incorporate:\n{[m.content for m in messages_to_summarize if m.content]}")
    ]
    
    summary_response = llm.invoke(summary_prompt)
    new_summary = summary_response.content
    print(f"\n📦 [Summarization Triggered]: Condensed {len(messages_to_summarize)} messages into state summary.")
    
    # Emit RemoveMessage directives for every message being pruned
    prune_actions = [RemoveMessage(id=msg.id) for msg in messages_to_summarize if getattr(msg, "id", None)]
    
    return {
        "system_summary": new_summary,
        "messages": prune_actions,
        "internal_notes": [f"Summarizer pruned {len(prune_actions)} messages into rolling summary."]
    }


# --- NODE 2: Persistent Agent Node with Injected Summary ---
def support_agent_node(state: GroceryAgentState) -> Dict[str, Any]:
    current_iter = state.get("iteration_count", 0) + 1
    existing_summary = state.get("system_summary")
    
    system_text = (
        "You are the senior AI support specialist for GroceryOnTheGo.\n"
        "Resolve customer inquiries efficiently using available tools when necessary.\n"
    )
    if existing_summary:
        system_text += f"\n--- RELEVANT CONVERSATION BACKGROUND ---\n{existing_summary}\n----------------------------------------\n"
        
    system_prompt = SystemMessage(content=system_text)
    full_messages = [system_prompt] + state["messages"]
    
    response = llm_with_tools.invoke(full_messages)
    
    return {
        "messages": [response],
        "iteration_count": current_iter,
        "internal_notes": [f"Agent response generated (Iteration {current_iter})."]
    }


# --- NODE 3: Tool Execution Node ---
def tool_execution_node(state: GroceryAgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    
    tool_messages = []
    notes = []
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        call_id = call["id"]
        
        tool_fn = TOOL_MAP.get(tool_name)
        output = tool_fn.invoke(tool_args) if tool_fn else f"Tool '{tool_name}' not recognized."
        tool_messages.append(ToolMessage(content=str(output), tool_call_id=call_id))
        notes.append(f"Executed {tool_name}")
        
    return {
        "messages": tool_messages,
        "internal_notes": notes
    }


# --- ROUTING LOGIC ---
def check_context_length(state: GroceryAgentState) -> Literal["summarizer", "agent"]:
    """Routes to summarizer if message count exceeds threshold."""
    if len(state["messages"]) > SUMMARY_THRESHOLD:
        return "summarizer"
    return "agent"


def should_continue(state: GroceryAgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return "__end__"


# --- COMPOSE GRAPH ---
def build_session5_graph(checkpointer: SqliteSaver):
    builder = StateGraph(GroceryAgentState)
    
    # Add Nodes
    builder.add_node("summarizer", summarization_node)
    builder.add_node("agent", support_agent_node)
    builder.add_node("tools", tool_execution_node)
    
    # Starting edge: check message volume first
    builder.add_conditional_edges(
        START,
        check_context_length,
        {
            "summarizer": "summarizer",
            "agent": "agent"
        }
    )
    
    # From summarizer, transition smoothly into agent
    builder.add_edge("summarizer", "agent")
    
    # Agent tool calling cycle
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    builder.add_edge("tools", "agent")
    
    return builder.compile(checkpointer=checkpointer)
Step 3: Write the Multi-Turn Verification Test
Append this test runner to the bottom of app/session5_summarization.py:
if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_session5_graph(checkpointer)
    
    config = {"configurable": {"thread_id": "THREAD-SUMMARY-TEST-101"}}
    
    print("\n========================================================")
    print("Testing Session 5: Rolling Summarization & Message Pruning")
    print("========================================================")
    
    # We simulate a multi-turn conversation that crosses the threshold of 8 messages
    simulated_conversation = [
        "Hi, I had an issue with my order ORD-8821.",
        "The delivery was late by 40 minutes.",
        "Also, my avocados arrived smashed and completely unusable.",
        "What is your compensation policy for the late delivery and damaged items?",
        "Can you confirm what payment method I originally used for this order?"
    ]
    
    for i, user_text in enumerate(simulated_conversation, 1):
        print(f"\n--- Turn {i}: Customer sends message ---")
        print(f"Customer: {user_text}")
        
        result = app.invoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "customer_id": "CUST-101",
                "thread_id": "THREAD-SUMMARY-TEST-101",
                "iteration_count": 0,
                "tool_call_fingerprints": [],
                "internal_notes": []
            },
            config=config
        )
        
        # Display the agent's latest answer
        latest_ai_msg = [m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls][-1]
        print(f"Agent: {latest_ai_msg.content[:140]}...")
        print(f"Current Message Count in State: {len(result['messages'])}")
        if result.get("system_summary"):
            print(f"Active System Summary: {result['system_summary']}")