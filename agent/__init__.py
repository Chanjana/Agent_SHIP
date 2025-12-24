"""
Shipment Delay Prediction Agent

A LangGraph-based AI agent that predicts shipment delays using
an Explainable Boosting Machine (EBM) model and provides
conversational explanations.
"""

from .graph import app
from .state import AgentState
from .ebm_tool import predict_with_ebm
from .email_tool import draft_email, send_email

__all__ = [
    'app',
    'AgentState',
    'predict_with_ebm',
    'draft_email',
    'send_email',
]

__version__ = '2.0.0'