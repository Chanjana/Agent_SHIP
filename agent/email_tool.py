"""
Email Tool - Agent_SHIP
Professional email generation and delivery for customer notifications
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# SMTP Configuration
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("SMTP_FROM", "noreply@shipment.com")
FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Shipment Operations")


def format_feature_name(feature: str) -> str:
    """
    Convert technical feature names to human-readable format.

    Args:
        feature: Technical feature name from model

    Returns:
        Human-readable feature name
    """
    name_map = {
        "scheduled_shipping_days": "Scheduled Delivery Timeframe",
        "ship_hour": "Processing Hour",
        "ship_dayofweek": "Day of Week",
        "ship_month": "Month",
        "is_weekend": "Weekend Processing",
        "is_peak_hour": "Peak Hour Processing",
        "is_night": "Overnight Processing",
        "shipping_mode": "Shipping Service Level",
        "type": "Payment Method",
        "order_item_quantity": "Order Quantity",
        "order_item_total": "Order Total Value",
        "order_item_discount": "Discount Amount",
        "order_item_discount_rate": "Discount Percentage",
        "sales": "Net Order Value",
        "order_profit_per_order": "Order Profitability",
        "market": "Market Region",
        "latitude": "Delivery Latitude",
        "longitude": "Delivery Longitude",
        "product_price": "Product Price"
    }
    return name_map.get(feature, feature.replace("_", " ").title())


def explain_factor(feature: str, weight: float, payload: dict = None) -> str:
    """
    Generate business-focused explanation for each contributing factor.

    Args:
        feature: Feature name
        weight: Feature contribution weight
        payload: Original form data for context

    Returns:
        Professional explanation text
    """
    explanations = {
        "scheduled_shipping_days": {
            "positive": "Tight delivery schedule creates operational pressure and reduces buffer time for unexpected delays. Shorter timeframes (1-2 days) historically show 40-60% higher delay rates compared to standard schedules.",
            "negative": "Adequate delivery timeframe provides operational flexibility and buffer capacity. Standard schedules (3-5 days) support reliable on-time performance."
        },
        "shipping_mode": {
            "positive": "Selected shipping service level has specific operational requirements and carrier dependencies that can introduce timing variability, particularly during peak periods or adverse conditions.",
            "negative": "Chosen shipping mode offers reliable performance with established carrier networks and proven delivery consistency."
        },
        "type": {
            "positive": "Payment verification process for this payment method requires additional processing steps and fraud prevention checks, which can extend order fulfillment time.",
            "negative": "Payment method enables streamlined processing with minimal verification delays, supporting faster order fulfillment and dispatch."
        },
        "ship_hour": {
            "positive": "Processing during this hour coincides with peak volume periods when warehouse congestion and carrier capacity constraints typically slow operations.",
            "negative": "Processing during off-peak hours allows for dedicated resource allocation and faster order fulfillment without capacity bottlenecks."
        },
        "is_weekend": {
            "positive": "Weekend processing encounters reduced carrier availability, limited customer service support, and potential delays in customs clearance for international shipments.",
            "negative": "Weekday processing benefits from full carrier network capacity, complete operational staffing, and normal service levels across all logistics partners."
        },
        "is_peak_hour": {
            "positive": "Peak hour processing (7-9 AM, 5-7 PM) creates resource competition, with multiple shipments competing for limited warehouse labor and carrier pickup slots.",
            "negative": "Off-peak processing allows prioritized handling with dedicated resources and immediate carrier access without congestion delays."
        },
        "is_night": {
            "positive": "Overnight processing may require orders to wait until morning carrier pickup windows, adding 6-12 hours of idle time before shipments enter the delivery network.",
            "negative": "Daytime processing enables same-day carrier pickup and immediate network entry, accelerating delivery timelines significantly."
        },
        "order_item_quantity": {
            "positive": "Multi-item orders increase picking complexity, inventory coordination requirements, and quality verification time. Bulk orders (5+ items) show 15% higher delay risk.",
            "negative": "Single-item fulfillment enables streamlined picking, minimal quality checks, and rapid packing operations."
        },
        "order_item_discount_rate": {
            "positive": "High discount percentage indicates promotional period timing, which typically correlates with elevated order volumes that can overwhelm warehouse capacity and extend processing times.",
            "negative": "Standard pricing indicates normal operational flow without the volume surges and capacity constraints associated with promotional events."
        },
        "sales": {
            "positive": "High-value order receives enhanced security protocols, additional quality verification steps, and potentially signature-required delivery.",
            "negative": "Standard order value follows optimized standard fulfillment procedures without additional security or verification requirements."
        },
        "order_profit_per_order": {
            "positive": "Profit margin considerations may affect resource allocation and processing priority within operational workflows.",
            "negative": "Healthy profit margins support investment in premium fulfillment services and expedited processing capabilities."
        },
        "market": {
            "positive": "Destination market presents specific logistics challenges including longer transit distances, potential customs delays, and limited last-mile carrier options.",
            "negative": "Destination market benefits from mature logistics infrastructure, established carrier partnerships, and reliable delivery networks."
        },
        "product_price": {
            "positive": "High-value product requires enhanced package security, insurance verification, and potentially signature confirmation at delivery, adding processing time.",
            "negative": "Standard product pricing enables efficient processing through automated workflows without special handling requirements."
        },
        "latitude": {
            "positive": "Delivery location's geographic characteristics may affect last-mile logistics efficiency and carrier accessibility.",
            "negative": "Delivery coordinates indicate accessible location with good carrier coverage and established delivery routes."
        },
        "longitude": {
            "positive": "Geographic distance from primary distribution network increases transit time requirements and potential for weather-related delays.",
            "negative": "Location proximity to fulfillment centers and distribution hubs supports rapid delivery and reduced transit time."
        }
    }

    direction = "positive" if weight > 0 else "negative"
    return explanations.get(feature, {}).get(
        direction,
        f"This factor {'increases' if weight > 0 else 'decreases'} delay risk based on historical shipping patterns and operational data."
    )


def draft_email(
        customer_name: str,
        customer_email: str,
        prediction_label: str,
        probability: float,
        explanations: List[Dict] = None,
) -> Dict:
    """
    Draft professional customer notification email.

    Args:
        customer_name: Recipient name
        customer_email: Recipient email address
        prediction_label: Prediction result (Delayed/On-Time)
        probability: Delay probability (0-1)
        explanations: List of contributing factors with weights

    Returns:
        Dictionary with email components
    """
    pct = round(probability * 100, 1)

    if prediction_label == "Delayed":
        status_summary = f"our predictive analysis indicates your shipment may experience delays, with a {pct}% probability of arrival beyond the scheduled delivery timeframe."
        recommendation = "We recommend planning accordingly and have implemented enhanced monitoring for your shipment. Our team will provide proactive updates as your order progresses through our network."
        subject_prefix = "Shipment Status Alert"
    else:
        status_summary = f"we are pleased to inform you that your shipment is projected for on-time delivery, with a {100 - pct}% probability of meeting the scheduled arrival date."
        recommendation = "We are monitoring your shipment closely to ensure it arrives as scheduled. No action is required on your part."
        subject_prefix = "Shipment Status Confirmation"

    # Factor explanations
    factor_details = []
    if explanations:
        top_factors = explanations[:5]
        for i, d in enumerate(top_factors, 1):
            feature = d['feature']
            weight = d['weight']
            pct_contrib = d['weight_percent']

            readable_name = format_feature_name(feature)
            explanation = explain_factor(feature, weight)

            factor_details.append({
                'number': i,
                'name': readable_name,
                'contribution': pct_contrib,
                'direction': 'increases' if weight > 0 else 'reduces',
                'explanation': explanation,
            })

    text = f"""Dear {customer_name},

{status_summary}

Our machine learning analysis has identified the following key factors affecting your shipment:

"""

    for factor in factor_details:
        text += f"""{factor['number']}. {factor['name']} ({factor['direction']} risk by {factor['contribution']}%)
   {factor['explanation']}

"""

    text += f"""Recommendations

{recommendation}

This prediction is based on comprehensive analysis of shipping modes, order characteristics, payment processing timelines, and market-specific logistics factors. Our predictive model analyzes historical data to provide actionable insights for supply chain optimization.

If you have questions about your shipment or require additional information, please contact our customer service team.

Best regards,
{FROM_NAME}

---
Please note that this is an automated notification generated by our supply chain intelligence system.
For shipment tracking and support, please visit our customer portal or contact customer service.
"""

    return {
        "to": customer_email,
        "subject": f"{subject_prefix}: {prediction_label}",
        "text": text,
        "html": None
    }


def send_email(to_email: str, subject: str, text: str, html: str = None) -> Dict:
    """
    Send email via SMTP or generate preview if SMTP not configured.

    Args:
        to_email: Recipient email address
        subject: Email subject line
        text: Plain text email body
        html: HTML email body (optional)

    Returns:
        Dictionary with send results or preview
    """
    # To check SMTP configuration
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and FROM_EMAIL):
        preview = f"""{"=" * 70}
EMAIL PREVIEW - SMTP NOT CONFIGURED
{"=" * 70}
To: {to_email}
From: {FROM_NAME} <{FROM_EMAIL}>
Subject: {subject}
{"=" * 70}

{text}

{"-" * 70}
To send real emails, configure SMTP credentials in .env file:
  SMTP_HOST=your.smtp.server
  SMTP_PORT=587
  SMTP_USER=your_username
  SMTP_PASS=your_password
  SMTP_FROM=noreply@yourdomain.com
{"-" * 70}
"""
        return {
            "sent": False,
            "preview": preview,
            "to": to_email,
            "subject": subject
        }

    if html:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))
    else:
        msg = MIMEText(text, "plain")

    msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = subject

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()  # Enable TLS encryption
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_EMAIL, [to_email], msg.as_string())

        return {
            "sent": True,
            "to": to_email,
            "subject": subject,
            "message": "Email delivered successfully..."
        }

    except Exception as e:
        preview = f"""{"=" * 70}
EMAIL DELIVERY FAILED
{"-" * 70}
Error: {str(e)}

Email Preview:
{"-" * 70}
To: {to_email}
From: {FROM_NAME} <{FROM_EMAIL}>
Subject: {subject}
{"-" * 70}

{text}

{"-" * 70}
Please verify SMTP configuration and network connectivity.
{"-" * 70}
"""
        return {
            "sent": False,
            "error": str(e),
            "preview": preview,
            "to": to_email,
            "subject": subject
        }