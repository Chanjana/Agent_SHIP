import os
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agent.graph import app as agent_app
from agent.email_tool import draft_email, send_email

load_dotenv()

st.set_page_config(
    page_title="Agent SHIP",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# JavaScript to force sidebar visible
st.markdown("""
<script>
window.addEventListener('load', function() {
    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
        sidebar.style.display = 'block';
        sidebar.style.visibility = 'visible';
        sidebar.style.marginLeft = '0';
    }
    const collapseBtn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
    if (collapseBtn) collapseBtn.style.display = 'none';
});
</script>
""", unsafe_allow_html=True)


# Load CSS
def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# Session Management
STORAGE_PATH = "storage/sessions.json"
os.makedirs("storage", exist_ok=True)


def load_sessions():
    if os.path.exists(STORAGE_PATH):
        with open(STORAGE_PATH) as f:
            return json.load(f)
    return {}


def save_sessions(sessions):
    with open(STORAGE_PATH, "w") as f:
        json.dump(sessions, f, indent=2)


# Initialize session state
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "form"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "email_draft" not in st.session_state:
    st.session_state.email_draft = None

# Sidebar
with st.sidebar:
    st.title("Sessions")

    if st.button("New Session", use_container_width=True):
        sid = str(uuid.uuid4())[:8]
        st.session_state.sessions[sid] = {
            "name": f"Session {datetime.now().strftime('%m/%d %H:%M')}",
            "created_at": datetime.now().isoformat(),
            "prediction": None,
            "explanation": None
        }
        st.session_state.session_id = sid
        st.session_state.workflow_stage = "form"
        st.session_state.prediction_result = None
        st.session_state.email_draft = None
        save_sessions(st.session_state.sessions)
        st.rerun()

    st.divider()

    if st.session_state.sessions:
        for sid, data in list(st.session_state.sessions.items()):
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                        data["name"],
                        key=f"s_{sid}",
                        use_container_width=True,
                        type="primary" if st.session_state.session_id == sid else "secondary"
                ):
                    st.session_state.session_id = sid
                    st.session_state.prediction_result = data.get("prediction")
                    st.session_state.workflow_stage = "form" if not data.get("prediction") else "analysis"
                    st.rerun()
            with col2:
                if st.button("✖", key=f"d_{sid}"):
                    if st.session_state.session_id == sid:
                        st.session_state.session_id = None
                    st.session_state.sessions.pop(sid)
                    save_sessions(st.session_state.sessions)
                    st.rerun()

if not st.session_state.session_id:
    # Welcome Page
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
            <div style="padding: 3rem 2rem;">
                <h1 style="font-size: 3rem; font-weight: 700; 
                           background: linear-gradient(135deg, #7c4dff 0%, #536dfe 100%);
                           -webkit-background-clip: text;
                           -webkit-text-fill-color: transparent;
                           margin-bottom: 1rem;">
                    Agent SHIP
                </h1>
                <p style="font-size: 1.25rem; color: #5e6c84; margin-bottom: 2rem;">
                    AI-powered shipment delay prediction with explainable analysis.
                </p>
                <div style="background: linear-gradient(135deg, #f5f3ff 0%, #e8eaf6 100%);
                            padding: 1.5rem; border-radius: 12px; border-left: 4px solid #7c4dff;
                            margin-bottom: 1.5rem;">
                    <h3 style="color: #7c4dff; margin-top: 0;">Key Features</h3>
                    <ul style="color: #2d3748; line-height: 1.8;">
                        <li>EBM-powered delay predictions with 70%+ accuracy.</li>
                        <li>Interactive feature contribution visualizations.</li>
                        <li>Automated customer notifications via email.</li>
                        <li>Actionable insights with supply chain knowledge base.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="padding-top: 2rem;">', unsafe_allow_html=True)
        st.image("ui/assets/IMG_SHIP1.png", use_container_width=True, caption="Agent Architecture")
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

session = st.session_state.sessions[st.session_state.session_id]

# Header Card
st.markdown(f"""
    <div class="header-card">
        <h1>{session['name']}</h1>
        <p>Your Supply Chain AI Assistant</p>
    </div>
""", unsafe_allow_html=True)

# STAGE 1 -> Shipment Detail Form
if st.session_state.workflow_stage == "form":
    st.subheader("Enter Shipment Details")

    with st.form("shipment_form"):
        # Shipping & Logistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Shipping Schedule**")

            default_datetime = datetime.now()

            shipping_datetime_input = st.text_input(
                "Shipping Date & Time",
                value=default_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                help="Format - YYYY-MM-DD HH:MM:SS (e.g., 2025-12-21 23:30:00)"
            )

            scheduled_shipping_days = st.number_input(
                "Scheduled Delivery (days)",
                value=4,
                min_value=1,
                max_value=30,
                help="Planned delivery days"
            )

        with col2:
            st.markdown("**Logistics**")
            shipping_mode = st.selectbox(
                "Shipping Mode",
                ["Standard Class", "First Class", "Second Class", "Same Day"],
                help="Delivery service type"
            )

            payment_type = st.selectbox(
                "Payment Type",
                ["Transfer", "Debit", "Payment", "Cash"],
                help="Payment method used"
            )

        with col3:
            st.markdown("**Market & Location**")
            market = st.selectbox(
                "Market",
                ["USCA", "LATAM", "Europe", "Pacific Asia", "Africa"],
                help="Geographic market region"
            )

            latitude = st.number_input(
                "Latitude",
                value=37.7749,
                format="%.4f",
                help="Delivery location latitude"
            )

            longitude = st.number_input(
                "Longitude",
                value=-122.4194,
                format="%.4f",
                help="Delivery location longitude"
            )

        st.divider()

        # Order Details
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Order Information**")
            order_item_quantity = st.number_input(
                "Item Quantity",
                value=1,
                min_value=1,
                help="Number of items in an order"
            )

            order_item_total = st.number_input(
                "Order Total ($)",
                value=327.75,
                min_value=0.01,
                format="%.2f",
                help="Total order value"
            )

        with col2:
            st.markdown("**Pricing & Discounts**")
            order_item_discount = st.number_input(
                "Discount Amount ($)",
                value=13.11,
                min_value=0.0,
                format="%.2f",
                help="Discount applied"
            )

            order_item_discount_rate = st.number_input(
                "Discount Rate (%)",
                value=4.0,
                min_value=0.0,
                max_value=100.0,
                format="%.1f",
                help="Discount in Percentage"
            )

        with col3:
            st.markdown("**Economics**")
            sales = st.number_input(
                "Net Sales ($)",
                value=314.64,
                min_value=0.01,
                format="%.2f",
                help="Order value after discounts"
            )

            order_profit_per_order = st.number_input(
                "Profit per Order ($)",
                value=91.25,
                format="%.2f",
                help="Profit margin"
            )

            product_price = st.number_input(
                "Product Price ($)",
                value=327.75,
                min_value=0.01,
                format="%.2f"
            )

        st.divider()
        submitted = st.form_submit_button(
            "Analyze Shipment",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        print("[APP] --- Form Submitted ---")

        # Proper datetime parsing
        try:
            # Parse the datetime string
            parsed_datetime = datetime.strptime(shipping_datetime_input.strip(), "%Y-%m-%d %H:%M:%S")
            # Convert to string in the exact format expected
            shipping_date_str = parsed_datetime.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[APP] ✓ Parsed datetime: {parsed_datetime}")
            print(f"[APP] ✓ Shipping datetime string: {shipping_date_str}")
        except ValueError as e:
            st.error(f"Invalid datetime format. Please use: YYYY-MM-DD HH:MM:SS (e.g., 2025-12-21 23:30:00)")
            print(f"[APP] ✗ Datetime parsing error: {e}")
            st.stop()

        # Need to include the parsed shipping_date
        form_payload = {
            "shipping_date": shipping_date_str,  # This is the KEY fix
            "scheduled_shipping_days": scheduled_shipping_days,
            "shipping_mode": shipping_mode,
            "payment_type": payment_type,
            "market": market,
            "latitude": latitude,
            "longitude": longitude,
            "order_item_quantity": order_item_quantity,
            "order_item_total": order_item_total,
            "order_item_discount": order_item_discount,
            "order_item_discount_rate": order_item_discount_rate / 100.0,
            "sales": sales,
            "order_profit_per_order": order_profit_per_order,
            "product_price": product_price
        }

        print("\n[APP] ---- Final Payload ----")
        for key, val in form_payload.items():
            print(f"[APP]   {key}: {val}")
        print("[APP] ---------------------------\n")

        with st.spinner("Analyzing shipment via EBM model..."):
            inputs = {
                "messages": [HumanMessage(content="Analyze this shipment for delay probability.")],
                "form_payload": form_payload,
                "phase": "idle"
            }

            try:
                prediction_result = None
                ai_explanation = None

                for state in agent_app.stream(inputs, stream_mode="values"):
                    if state.get("last_prediction"):
                        prediction_result = state["last_prediction"]
                        print(f"[APP] Received prediction: {prediction_result['prediction_label']}")
                    if state.get("messages") and len(state["messages"]) > 0:
                        last_msg = state["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            ai_explanation = last_msg.content
                            print(f"[APP] Received AI explanation")

                if not prediction_result:
                    st.error("Prediction failed - no result returned from model")
                    st.stop()

                st.session_state.prediction_result = prediction_result
                session["prediction"] = prediction_result
                session["explanation"] = ai_explanation
                save_sessions(st.session_state.sessions)

                st.session_state.workflow_stage = "analysis"
                st.success("✓ Analysis complete!")
                st.rerun()

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                print(f"\n[ERROR] Analysis failed: {e}")
                import traceback
                traceback.print_exc()

# STAGE 2 -> Analysis

elif st.session_state.workflow_stage == "analysis":
    result = st.session_state.prediction_result
    explanation = session.get("explanation", "")

    if not result:
        st.error("No prediction result available.")
        if st.button("Start New Analysis"):
            st.session_state.workflow_stage = "form"
            st.rerun()
        st.stop()

    # Display prediction
    status = result['prediction_label']
    prob = result['probability']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", status, delta=None)
    with col2:
        st.metric("Delay Probability", f"{prob * 100:.1f}%")
    with col3:
        confidence = "High" if prob > 0.7 or prob < 0.3 else "Moderate"
        st.metric("Confidence", confidence)

    st.divider()

    # AI Explanation
    st.subheader("Analysis")
    if explanation:
        st.markdown(explanation)
    else:
        fallback = f"The analysis indicates a **{prob * 100:.1f}% probability** that this shipment will be **{status.lower()}**.\n\n"
        drivers = result.get('drivers', [])[:5]
        if drivers:
            fallback += "**Primary Contributing Factors:**\n\n"
            for i, d in enumerate(drivers, 1):
                feat = d['feature'].replace('_', ' ').title()
                direction = "increases" if d['weight'] > 0 else "reduces"
                fallback += f"{i}. **{feat}**: {direction} delay risk by {d['weight_percent']}%\n\n"
        st.markdown(fallback)

    st.divider()

    # EBM Chart
    if result.get('plot_path') and os.path.exists(result['plot_path']):
        st.subheader("Feature Contribution Analysis")

        import plotly.graph_objects as go

        try:
            with open(result['plot_path']) as f:
                fig_json = json.load(f)
                fig = go.Figure(json.loads(fig_json))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart unavailable: {e}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start New Analysis", use_container_width=True):
            st.session_state.workflow_stage = "form"
            st.session_state.prediction_result = None
            st.rerun()
    with col2:
        if st.button("Notify Customer", use_container_width=True, type="primary"):
            st.session_state.workflow_stage = "email_form"
            st.rerun()


# STAGE 3 -> Email
elif st.session_state.workflow_stage == "email_form":
    st.subheader("Customer Notification")

    with st.form("email_form"):
        col1, col2 = st.columns(2)
        with col1:
            customer_name = st.text_input("Customer Name", placeholder="John Smith")
        with col2:
            customer_email = st.text_input("Customer Email", placeholder="john@example.com")

        col_submit, col_back = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("Draft Email", use_container_width=True, type="primary")
        with col_back:
            back = st.form_submit_button("Back to Analysis", use_container_width=True)

    if back:
        st.session_state.workflow_stage = "analysis"
        st.rerun()

    if submitted:
        if not customer_name or not customer_email:
            st.error("Please provide both customer name and email.")
        else:
            result = st.session_state.prediction_result
            email_draft = draft_email(
                customer_name=customer_name,
                customer_email=customer_email,
                prediction_label=result["prediction_label"],
                probability=result["probability"],
                explanations=result.get("drivers", []),
            )
            st.session_state.email_draft = email_draft
            st.session_state.workflow_stage = "email_preview"
            st.rerun()

elif st.session_state.workflow_stage == "email_preview":
    draft = st.session_state.email_draft
    if not draft:
        st.error("No email draft available.")
        st.session_state.workflow_stage = "email_form"
        st.rerun()
        st.stop()

    st.subheader("Email Preview")
    st.markdown(f'**To:** {draft["to"]}')
    st.markdown(f'**Subject:** {draft["subject"]}')
    st.divider()

    edited_content = st.text_area("Email content:", value=draft['text'], height=400)

    if edited_content != draft['text']:
        draft['text'] = edited_content
        st.session_state.email_draft = draft

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back", use_container_width=True):
            st.session_state.workflow_stage = "email_form"
            st.rerun()
    with col2:
        if st.button("Send Email", use_container_width=True, type="primary"):
            with st.spinner("Sending..."):
                result = send_email(draft["to"], draft["subject"], draft["text"], draft.get("html"))
                st.session_state.email_result = result
                st.session_state.workflow_stage = "complete"
                st.rerun()
    with col3:
        if st.button("Cancel", use_container_width=True):
            st.session_state.workflow_stage = "analysis"
            st.rerun()

elif st.session_state.workflow_stage == "complete":
    email_result = st.session_state.get("email_result", {})
    st.subheader("Process Complete")

    if email_result.get("sent"):
        st.success(f"Email sent to {email_result.get('to')}")
    else:
        st.info("Email Preview (SMTP not configured)")
        if "preview" in email_result:
            st.code(email_result["preview"])

    st.divider()
    if st.button("Start New Analysis", use_container_width=True, type="primary"):
        st.session_state.workflow_stage = "form"
        st.session_state.prediction_result = None
        st.session_state.email_draft = None
        st.rerun()

# Footer
st.divider()
st.caption("Powered by Agent SHIP | LangGraph + EBM + Ollama | 2025")