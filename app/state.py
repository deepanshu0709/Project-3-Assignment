import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class GroceryAgentState(TypedDict):
    """
    Session 5 State Schema:
    Adds system_summary for long-context compression and message pruning.
    """
    # Messages list with append reducer (also processes RemoveMessage)
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Customer and session identifiers
    customer_id: str
    thread_id: str
    
    # Triage classification
    ticket_category: Optional[str]
    urgency_level: Optional[str]
    
    # Trace notes
    internal_notes: Annotated[List[str], operator.add]
    
    # Circuit breaker & loop detection guards
    iteration_count: int
    tool_call_fingerprints: Annotated[List[str], operator.add]
    
    # Session 5: Rolling context summarization
    system_summary: Optional[str]
    
    # Final response delivered to the customer
    final_response: Optional[str]