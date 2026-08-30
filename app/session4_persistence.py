import os
import sqlite3
from typing import Dict, Any, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.state import GroceryAgentState
from app.tools import SESSION_2_TOOLS, get_order_details, search_grocery_policy

load_dotenv()

MAX_ITERATIONS = 5
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


# --- AGENT NODE ---
def persistent_agent_node(state: GroceryAgentState) -> Dict[str, Any]:
    current_iter = state.get("iteration_count", 0) + 1
    
    system_prompt = SystemMessage(content=(
        "You are the senior resolution specialist for GroceryOnTheGo.\n"
        "You have memory of previous turns in this conversation.\n"
        "Use tools only when new information is needed. If the answer is already in the conversation history, answer directly without re-querying tools.\n"
        "Be concise, polite, and direct."
    ))
    
    # We pass the full history retrieved by the checkpointer
    full_messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(full_messages)
    
    return {
        "messages": [response],
        "iteration_count": current_iter,
        "internal_notes": [f"Persistent turn executed. Iteration: {current_iter}"]
    }


# --- GUARDED TOOL NODE ---
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
        if tool_fn:
            output = tool_fn.invoke(tool_args)
        else:
            output = f"Tool '{tool_name}' not recognized."
            
        tool_messages.append(ToolMessage(content=str(output), tool_call_id=call_id))
        notes.append(f"Executed tool: {tool_name}")
        
    return {
        "messages": tool_messages,
        "internal_notes": notes
    }


def should_continue(state: GroceryAgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return "__end__"


# --- BUILD GRAPH WITH CHECKPOINTER ---
def build_session4_graph(checkpointer: SqliteSaver):
    builder = StateGraph(GroceryAgentState)
    
    builder.add_node("agent", persistent_agent_node)
    builder.add_node("tools", tool_execution_node)
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "agent")
    
    # Compile with persistence
    return builder.compile(checkpointer=checkpointer)

##Verification

if __name__ == "__main__":
    # Open persistent SQLite connection
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_session4_graph(checkpointer)
    
    # Thread A: Sarah Jenkins inquiring about her order
    config_thread_a = {"configurable": {"thread_id": "THREAD-SARAH-101"}}
    
    print("\n--- TURN 1 (Thread A: Sarah asks about order ORD-8821) ---")
    turn1 = app.invoke(
        {
            "messages": [HumanMessage(content="Hi, what items are in my order ORD-8821?")],
            "customer_id": "CUST-101",
            "thread_id": "THREAD-SARAH-101",
            "iteration_count": 0,
            "tool_call_fingerprints": [],
            "internal_notes": []
        },
        config=config_thread_a
    )
    print(f"Agent:\n{turn1['messages'][-1].content}")
    
    print("\n--- TURN 2 (Thread A: Follow-up question relying on memory) ---")
    # Notice we ONLY send the new message; LangGraph loads previous messages from SQLite automatically!
    turn2 = app.invoke(
        {
            "messages": [HumanMessage(content="Which of those items are perishable?")]
        },
        config=config_thread_a
    )
    print(f"Agent:\n{turn2['messages'][-1].content}")
    
    print("\n--- TURN 3 (Thread B: New user, isolated conversation) ---")
    config_thread_b = {"configurable": {"thread_id": "THREAD-ALEX-909"}}
    turn3 = app.invoke(
        {
            "messages": [HumanMessage(content="What did I just ask you?")],
            "customer_id": "CUST-909",
            "thread_id": "THREAD-ALEX-909",
            "iteration_count": 0,
            "tool_call_fingerprints": [],
            "internal_notes": []
        },
        config=config_thread_b
    )
    print(f"Agent:\n{turn3['messages'][-1].content}")