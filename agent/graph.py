"""
LangGraph Workflow - Agent_SHIP
Professional supply chain delay prediction with explainable AI
Simplified version without chat interface
"""
import os
from typing import Literal, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from .state import AgentState
from .ebm_tool import predict_with_ebm
from .email_tool import draft_email, send_email
from .knowledge_base import (
    get_knowledge,
    get_recommendations,
    format_knowledge_for_prompt
)

load_dotenv()

# Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initializing Ollama
try:
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.3,
        base_url=OLLAMA_BASE_URL,
    )
    LLM_AVAILABLE = True
    print(f"[GRAPH] LLM initialized: {LLM_MODEL}")
except Exception as e:
    print(f"[GRAPH] LLM unavailable: {e}")
    LLM_AVAILABLE = False

# Agent Prompt with strict fact based analysis
SYSTEM_PROMPT = """You are a professional Supply Chain Analysis Assistant specializing in logistics optimization and delay prediction.

CRITICAL RULES:
1. Use ONLY the data provided - never invent, assume, or speculate
2. Maintain a professional, authoritative tone without casual language
3. Be direct and concise - avoid filler phrases
4. Reference specific numerical data and percentages from the analysis
5. Base all explanations on the knowledge base information provided
6. Do not use phrases like "I think", "maybe", "probably", "I'm no expert"
7. Avoid emotional language, analogies, or hypothetical scenarios
8. Provide actionable, evidence-based recommendations

COMMUNICATION STYLE:
- Professional business terminology
- Clear, direct statements
- Evidence-based conclusions
- Concise delivery (200-300 words maximum)
- Structured analysis format

REQUIRED STRUCTURE:
1. State the prediction with exact probability
2. Identify top 3-4 contributing factors with exact percentage contributions
3. Explain each factor's business impact using knowledge base data
4. Provide 2-3 specific, actionable recommendations
5. End with clear next step

PROHIBITED:
- Casual conversational phrases
- Speculation beyond provided data
- Invented explanations
- Personal opinions
- Analogies or stories
- Hedging language"""


def predict_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute EBM prediction and generate professional analysis.

    Args:
        state: Current agent state with form payload

    Returns:
        Updated state with prediction results and analysis
    """
    print("[PREDICT_NODE] Initiating shipment delay prediction")

    payload = state.get("form_payload")

    if not payload:
        print("[PREDICT_NODE] ERROR: No payload provided")
        return {
            "messages": [AIMessage(content="Shipment data required to generate prediction.")],
            "phase": "idle",
            "last_prediction": None
        }

    if "shipping_date" not in payload:
        print("[PREDICT_NODE] ERROR: Missing required field 'shipping_date'")
        return {
            "messages": [AIMessage(content="Shipping date is required for prediction.")],
            "phase": "idle",
            "last_prediction": None
        }

    print(f"[PREDICT_NODE] Payload validated: {len(payload)} fields")
    print(f"[PREDICT_NODE] Shipping date: {payload['shipping_date']}")

    try:
        # EBM prediction
        print("\n[PREDICT_NODE] Executing EBM model prediction...")
        result = predict_with_ebm(payload)

        print(f"[PREDICT_NODE] Prediction complete:")
        print(f"  Status: {result['prediction_label']}")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Drivers extracted: {len(result.get('drivers', []))}")

        # Retrieving recommendations from knowledge base
        print("\n[PREDICT_NODE] Retrieving knowledge base recommendations...")
        recommendations = get_recommendations(
            result['prediction_label'],
            result.get('drivers', [])
        )
        print(f"[PREDICT_NODE] Recommendations generated: {len(recommendations)}")

        # Generating professional analysis
        print("\n[PREDICT_NODE] Generating professional analysis...")

        if LLM_AVAILABLE:
            try:
                analysis_text = generate_llm_analysis(result, payload, recommendations)
                print("[PREDICT_NODE] LLM analysis generated successfully")
            except Exception as llm_error:
                print(f"[PREDICT_NODE] LLM generation failed: {llm_error}")
                analysis_text = generate_template_analysis(result, recommendations, payload)
                print("[PREDICT_NODE] Using template-based analysis")
        else:
            analysis_text = generate_template_analysis(result, recommendations, payload)
            print("[PREDICT_NODE] Using template-based analysis (LLM unavailable)")

        print(f"[PREDICT_NODE] Analysis length: {len(analysis_text)} characters")
        print("\n")
        print("[PREDICT_NODE] Prediction workflow completed successfully")

        return {
            "last_prediction": result,
            "form_payload": None,
            "phase": "analysis",
            "messages": [AIMessage(content=analysis_text)]
        }

    except Exception as e:
        print(f"\n[PREDICT_NODE] ERROR: Prediction failed")
        print(f"  Exception: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "messages": [AIMessage(content=f"Error during prediction: {str(e)}")],
            "phase": "idle",
            "last_prediction": None
        }


def generate_llm_analysis(result: dict, payload: dict, recommendations: list) -> str:
    """
    Generate professional analysis using LLM with strict constraints.

    Args:
        result: EBM prediction results
        payload: Original form data
        recommendations: Knowledge base recommendations

    Returns:
        Professional analysis text
    """
    status = result['prediction_label']
    prob = result['probability']
    drivers = result.get('drivers', [])[:10]

    # Building knowledge context for top factors
    knowledge_context = {}
    for driver in drivers[:5]:
        feature = driver['feature']
        knowledge = get_knowledge("delay_factors", feature)
        if knowledge:
            knowledge_context[feature] = {
                'description': knowledge.get('description', ''),
                'impact': knowledge.get('impact', ''),
                'mitigation': knowledge.get('mitigation', ''),
                'best_practices': knowledge.get('best_practices', [])
            }

    prompt = f"""TASK: Generate a professional shipment delay analysis based ONLY on the provided data.

PREDICTION DATA:
- Status: {status}
- Probability: {prob * 100:.1f}%

TOP CONTRIBUTING FACTORS (with exact percentages):
"""

    for i, d in enumerate(drivers[:5], 1):
        direction = "increases" if d['weight'] > 0 else "reduces"
        prompt += f"{i}. {d['feature']}: {direction} delay risk by {d['weight_percent']}%\n"

    prompt += "\nKNOWLEDGE BASE INFORMATION:\n"

    for feature, info in knowledge_context.items():
        prompt += f"""
{feature}:
- Description: {info['description']}
- Business Impact: {info['impact']}
- Mitigation Strategy: {info['mitigation']}
"""

    prompt += f"""
RECOMMENDED ACTIONS (from knowledge base):
{chr(10).join(f"- {rec}" for rec in recommendations[:5])}

SHIPMENT CONTEXT:
- Shipping Mode: {payload.get('shipping_mode', 'Unknown')}
- Scheduled Delivery: {payload.get('scheduled_shipping_days', 'N/A')} days
- Market Region: {payload.get('market', 'Unknown')}
- Payment Method: {payload.get('payment_type', 'Unknown')}
- Order Value: ${payload.get('sales', 0):.2f}

INSTRUCTIONS:
1. Write analysis in 250-300 words
2. State prediction probability as: "{prob*100:.1f}%"
3. Explain top 3-4 factors using their exact percentage contributions
4. Use ONLY the knowledge base information provided
5. List 2-3 specific recommended actions
6. Use professional, direct language
7. No casual phrases, analogies, or speculation
8. End with a clear, actionable next step

Generate the professional analysis now:"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)


def generate_template_analysis(result: dict, recommendations: list, payload: dict) -> str:
    """
    Generate professional template-based analysis (fallback when LLM unavailable).

    Args:
        result: EBM prediction results
        recommendations: Knowledge base recommendations
        payload: Original form data

    Returns:
        Professional analysis text
    """
    status = result['prediction_label']
    prob = result['probability']
    drivers = result.get('drivers', [])[:5]

    # Executive summary
    if status == "Delayed":
        risk_assessment = "HIGH RISK" if prob > 0.7 else "MODERATE RISK"
        summary = f"""SHIPMENT DELAY PREDICTION - {risk_assessment}

The analysis indicates a {prob * 100:.1f}% probability of delay for this shipment. Based on current shipping parameters, the shipment faces significant timing challenges requiring immediate attention.
"""
    else:
        confidence = "HIGH CONFIDENCE" if prob < 0.3 else "MODERATE CONFIDENCE"
        summary = f"""ON-TIME DELIVERY PREDICTION - {confidence}

The analysis indicates a {(1-prob) * 100:.1f}% probability of on-time delivery. Current shipping parameters support successful delivery within the scheduled timeframe.
"""

    # Contributing factors analysis
    factors_text = "\nPrimary Contributing Factors:\n\n"

    for i, d in enumerate(drivers, 1):
        feature = d['feature']
        weight = d['weight']
        pct = d['weight_percent']

        feature_display = feature.replace('_', ' ').title()
        direction = "increases" if weight > 0 else "reduces"

        # Get knowledge base explanation
        knowledge = get_knowledge("delay_factors", feature)
        explanation = knowledge.get('impact', '') if knowledge else f"This factor {direction} delay probability based on historical patterns."

        factors_text += f"{i}. {feature_display} ({direction} risk by {pct}%)\n"
        factors_text += f"   {explanation}\n\n"

    # Shipment context
    context_text = f"""Shipment Context:
- Mode: {payload.get('shipping_mode', 'Unknown')}
- Scheduled: {payload.get('scheduled_shipping_days', 'N/A')} days
- Market: {payload.get('market', 'Unknown')}
- Payment: {payload.get('payment_type', 'Unknown')}
- Value: ${payload.get('sales', 0):.2f}

"""

    # Recommendations
    rec_text = "Recommended Actions:\n"
    for i, rec in enumerate(recommendations[:3], 1):
        rec_text += f"{i}. {rec}\n"

    rec_text += f"\nNext Step: {'Implement risk mitigation strategies immediately.' if status == 'Delayed' else 'Continue standard monitoring procedures.'}"

    return summary + factors_text + context_text + rec_text


def email_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate and send customer notification email.

    Args:
        state: Current agent state

    Returns:
        Updated state with email results
    """
    print("\n[EMAIL_NODE] Initiating email generation")

    lp = state.get("last_prediction")
    if not lp:
        print("[EMAIL_NODE] ERROR: No prediction available")
        return {
            "messages": [AIMessage(content="Prediction required before generating email notification.")],
            "phase": "idle"
        }

    customer_name = state.get("customer_name") or "Valued Customer"
    customer_email = state.get("customer_email") or ""

    if not customer_email:
        print("[EMAIL_NODE] ERROR: No customer email provided")
        return {
            "messages": [AIMessage(content="Customer email address required.")],
            "phase": "idle"
        }

    print(f"[EMAIL_NODE] Generating email for: {customer_name} ({customer_email})")

    # Draft email
    draft = draft_email(
        customer_name=customer_name,
        customer_email=customer_email,
        prediction_label=lp["prediction_label"],
        probability=lp["probability"],
        explanations=lp.get("drivers", []),
    )

    print(f"[EMAIL_NODE] Email drafted: {draft['subject']}")

    # Preview
    result = send_email(
        to_email=draft["to"],
        subject=draft["subject"],
        text=draft["text"],
        html=draft.get("html")
    )

    if result.get("sent"):
        response = f"Email sent successfully to {draft['to']}"
        print(f"[EMAIL_NODE] Email delivered successfully")
    else:
        response = f"Email preview generated (SMTP not configured)\n\nRecipient: {draft['to']}\nSubject: {draft['subject']}"
        print(f"[EMAIL_NODE] Email preview generated")

    return {
        "last_email_result": result,
        "phase": "complete",
        "messages": [AIMessage(content=response)]
    }

# LangGraph Workflow

print("[GRAPH] Building Agent SHIP")

# Initialize state graph
graph = StateGraph(AgentState)

# Adding nodes
print("[GRAPH] Adding nodes predict, email")
graph.add_node("predict", predict_node)
graph.add_node("email", email_node)

# Setting entry point
print("[GRAPH] Setting entry point as predict")
graph.set_entry_point("predict")

# No conditional routing
print("[GRAPH] Adding edges")
graph.add_edge("predict", END)
graph.add_edge("email", END)

# Compile workflow
print("[GRAPH] Compiling workflow")
app = graph.compile()

print("[GRAPH] Agent SHIP workflow compiled successfully")