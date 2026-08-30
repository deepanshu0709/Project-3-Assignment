import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class GroceryAgentState(TypedDict):
    """
    Session 1 State Blueprint:
    Defines the blackboard memory shared across all nodes in the graph.
    """
    # Messages list uses operator.add so new messages append rather than overwrite
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Customer and session identifiers
    customer_id: str
    thread_id: str
    
    # Triage classification output
    ticket_category: Optional[str]  # "delivery", "quality_refund", "fraud", "general"
    urgency_level: Optional[str]    # "low", "medium", "high", "critical"
    
    # Trace notes for debugging
    internal_notes: Annotated[List[str], operator.add]
    
    # Final response delivered to the customer
    final_response: Optional[str]