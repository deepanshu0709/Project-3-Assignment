import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.state import GroceryAgentState
from app.tools import SESSION_2_TOOLS

load_dotenv()

# 1. Initialize Gemini and bind our tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)
llm_with_tools = llm.bind_tools(SESSION_2_TOOLS)


# 2. Agent Node (Calls Gemini with conversation history)
def support_agent_node(state: GroceryAgentState) -> Dict[str, Any]:
    system_prompt = SystemMessage(content=(
        "You are the senior AI support specialist for GroceryOnTheGo.\n"
        "You have tools to check order details (get_order_details) and store policies (search_grocery_policy).\n"
        "When a customer mentions an issue with an order, use your tools to look up the facts before answering.\n"
        "Be empathetic, concise, and helpful."
    ))
    
    # Prepend system prompt to conversation messages
    full_messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(full_messages)
    
    # Append the AI message (which may contain tool_calls or final text)
    return {
        "messages": [response],
        "internal_notes": [f"Agent called. Has tool calls: {bool(response.tool_calls)}"]
    }


# 3. Assemble the Graph
def build_session2_graph():
    builder = StateGraph(GroceryAgentState)
    
    # Define Nodes
    builder.add_node("support_agent", support_agent_node)
    builder.add_node("tools", ToolNode(SESSION_2_TOOLS))
    
    # Define Flow
    builder.add_edge(START, "support_agent")
    
    # Conditional routing:
    # If the agent called a tool, route to 'tools', otherwise route to END
    builder.add_conditional_edges(
        "support_agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    
    # Once tool finishes executing, return back to agent so it can read tool results and answer
    builder.add_edge("tools", "support_agent")
    
    return builder.compile()


if __name__ == "__main__":
    app_graph = build_session2_graph()
    
    # Test ticket: customer asks about their order items and delivery status
    test_state = {
        "messages": [HumanMessage(content="Hi, I am Sarah Jenkins. Can you check the status of my order ORD-8821? My avocados arrived smashed.")],
        "customer_id": "CUST-101",
        "thread_id": "SESSION-2-DEMO",
        "internal_notes": []
    }
    
    print("\n--- Running Session 2 Agent Execution ---")
    result = app_graph.invoke(test_state)
    
    print("\n--- Execution Conversation Log ---")
    for msg in result["messages"]:
        print(f"\n[{msg.__class__.__name__}]: {msg.content or msg.tool_calls}")