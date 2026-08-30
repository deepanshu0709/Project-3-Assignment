import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GroceryAgentState(TypedDict):
    """
    Session 6 State Schema:
    Adds security guardrail flags to the state ledger.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: str
    thread_id: str
    ticket_category: Optional[str]
    urgency_level: Optional[str]
    internal_notes: Annotated[List[str], operator.add]
    iteration_count: int
    tool_call_fingerprints: Annotated[List[str], operator.add]
    system_summary: Optional[str]
    
    # Session 6: Security Guardrail Flags
    pii_detected: bool
    injection_attempt_detected: bool
    
    final_response: Optional[str]