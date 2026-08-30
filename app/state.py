import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class GroceryAgentState(TypedDict):
    """
    Session 3 State Schema:
    Adds iteration bounding and execution fingerprints to prevent runaway loops.
    """
    # Messages list with append reducer
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Customer and session identifiers
    customer_id: str
    thread_id: str
    
    # Triage classification
    ticket_category: Optional[str]
    urgency_level: Optional[str]
    
    # Trace notes
    internal_notes: Annotated[List[str], operator.add]
    
    # Session 3: Circuit breaker & Loop detection guards
    iteration_count: int
    tool_call_fingerprints: Annotated[List[str], operator.add]
    
    # Final response delivered to the customer
    final_response: Optional[str]