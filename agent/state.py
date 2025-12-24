from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State definition for the shipment delay prediction agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Workflow phase
    phase: str

    # Email data
    customer_name: Optional[str]
    customer_email: Optional[str]

    # Form data for prediction request
    form_payload: Optional[Dict[str, Any]]

    # Tool results
    last_prediction: Optional[Dict[str, Any]]
    last_email_result: Optional[Dict[str, Any]]