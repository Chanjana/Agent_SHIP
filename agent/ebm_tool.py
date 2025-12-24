import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from interpret.glassbox import ExplainableBoostingClassifier

MODEL_PATH = os.path.abspath(os.path.join("models", "ebm_model.pkl"))
ARTIFACT_DIR = os.path.abspath("artifacts")


def to_model_features(payload: dict) -> pd.DataFrame:
    """
    Convert form payload to model features.

    CRITICAL: This function handles datetime parsing to extract temporal features.
    The shipping_date comes as a STRING in format "YYYY-MM-DD HH:MM:SS"
    """

    # To get shipping_date as string and validate it exists
    shipping_date_str = payload.get("shipping_date")

    if shipping_date_str is None:
        raise ValueError("shipping_date is missing from payload!")

    print(f"[DEBUG] shipping_date from payload: '{shipping_date_str}' (type: {type(shipping_date_str)})")

    # To convert string to pandas Timestamp
    try:
        shipping_date = pd.to_datetime(shipping_date_str)
        print(f"[DEBUG] Parsed to Timestamp: {shipping_date} (type: {type(shipping_date)})")
    except Exception as e:
        raise ValueError(f"Failed to parse shipping_date '{shipping_date_str}': {e}")

    # Extract new time features
    ship_hour = int(shipping_date.hour)
    ship_dayofweek = int(shipping_date.dayofweek)
    ship_month = int(shipping_date.month)
    is_weekend = 1 if ship_dayofweek in [5, 6] else 0
    is_peak_hour = 1 if ship_hour in [7, 8, 9, 17, 18, 19] else 0
    is_night = 1 if ship_hour in [0, 1, 2, 3, 4, 22, 23] else 0

    print(f"[DEBUG] Extracted temporal features:")
    print(f"  ship_hour: {ship_hour}")
    print(f"  ship_dayofweek: {ship_dayofweek}")
    print(f"  ship_month: {ship_month}")
    print(f"  is_weekend: {is_weekend}")
    print(f"  is_peak_hour: {is_peak_hour}")
    print(f"  is_night: {is_night}")

    # Extract other features
    scheduled_shipping_days = float(payload.get("scheduled_shipping_days", 4))
    shipping_mode = str(payload.get("shipping_mode", "Standard Class"))
    payment_type = str(payload.get("payment_type", "Transfer"))

    # Mapping payment type to match training data
    payment_map = {
        "Transfer": "TRANSFER",
        "Debit": "DEBIT",
        "Payment": "PAYMENT",
        "Cash": "CASH"
    }
    type_value = payment_map.get(payment_type, payment_type.upper())

    order_item_quantity = int(payload.get("order_item_quantity", 1))
    order_item_total = float(payload.get("order_item_total", 0))
    order_item_discount = float(payload.get("order_item_discount", 0))
    order_item_discount_rate = float(payload.get("order_item_discount_rate", 0))

    sales = float(payload.get("sales", 0))
    order_profit_per_order = float(payload.get("order_profit_per_order", 0))
    product_price = float(payload.get("product_price", 0))

    market = str(payload.get("market", "USCA"))
    latitude = float(payload.get("latitude", 0))
    longitude = float(payload.get("longitude", 0))

    # Building feature row
    row = {
        "scheduled_shipping_days": scheduled_shipping_days,
        "ship_hour": ship_hour,
        "ship_dayofweek": ship_dayofweek,
        "ship_month": ship_month,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "is_night": is_night,
        "shipping_mode": shipping_mode,
        "type": type_value,
        "order_item_quantity": order_item_quantity,
        "order_item_total": order_item_total,
        "order_item_discount": order_item_discount,
        "order_item_discount_rate": order_item_discount_rate,
        "sales": sales,
        "order_profit_per_order": order_profit_per_order,
        "market": market,
        "latitude": latitude,
        "longitude": longitude,
        "product_price": product_price
    }

    print(f"[SUCCESS] Created {len(row)} features")

    df = pd.DataFrame([row])
    print(f"[SUCCESS] DataFrame shape: {df.shape}")
    print(f"[SUCCESS] Features: {list(df.columns)}")

    return df


def create_local_explanation_chart(drivers, top_k=10) -> str:
    """Create interactive Plotly chart for feature contributions"""
    if not drivers:
        return ""

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    top_drivers = drivers[:top_k]
    features = [d["feature"] for d in top_drivers]
    contributions = [d["weight"] for d in top_drivers]

    colors = ['rgb(239, 85, 59)' if c > 0 else 'rgb(99, 110, 250)' for c in contributions]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features[::-1],
        x=contributions[::-1],
        orientation='h',
        marker=dict(color=colors[::-1]),
        text=[f"{c:.3f}" for c in contributions[::-1]],
        textposition='auto',
    ))

    fig.update_layout(
        title="Local Feature Contributions (EBM)",
        xaxis_title="Contribution to Delay Risk",
        yaxis_title="Feature",
        height=max(400, 50 * len(features)),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=80)
    )

    chart_path = os.path.join(ARTIFACT_DIR, "ebm_local_explanation.json")
    with open(chart_path, 'w') as f:
        json.dump(fig.to_json(), f)

    return os.path.abspath(chart_path)


def predict_with_ebm(form_payload: dict) -> dict:
    """
    Run EBM prediction on shipment data.

    Args:
        form_payload: Dictionary containing all shipment features

    Returns:
        Dictionary with prediction results and explanations
    """
    print("\n" + "=" * 70)
    print("[PREDICT_WITH_EBM] Starting prediction workflow")
    print("=" * 70)

    # Validate payload
    if not form_payload:
        raise ValueError("Empty payload provided")

    if "shipping_date" not in form_payload:
        raise ValueError("shipping_date is required in payload")

    # Load model
    print(f"\n[1/4] Loading EBM model...")
    print(f"[INFO] Model path: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print(f"[SUCCESS] ✓ Model loaded: {type(model).__name__}")

    # Build features
    print(f"\n[2/4] Extracting features from payload...")
    try:
        X = to_model_features(form_payload)
        print(f"[SUCCESS] ✓ Features extracted successfully")
    except Exception as e:
        print(f"[ERROR] ✗ Feature extraction failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Align features to model expectations
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        print(f"\n[3/4] Aligning features to model...")
        print(f"[INFO] Model expects {len(expected_features)} features")

        # Convert categorical columns to string
        cat_cols = X.select_dtypes(include='object').columns
        X[cat_cols] = X[cat_cols].astype(str)

        # Reindex to match model features
        X = X.reindex(columns=expected_features, fill_value=0)
        print(f"[SUCCESS] ✓ Features aligned")

    # Run prediction
    print(f"\n[4/4] Running EBM prediction...")
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    label = "Delayed" if pred == 1 else "On-Time"

    print(f"[SUCCESS] ✓ Prediction: {label}")
    print(f"[SUCCESS] ✓ Probability: {proba:.4f} ({proba * 100:.1f}%)")

    # Extract explanations
    print(f"\n[5/5] Extracting feature contributions...")
    drivers = []
    try:
        local_exp = model.explain_local(X)

        feature_names = None
        contributions = None

        # Try to extract data from explanation object
        if hasattr(local_exp, 'data'):
            try:
                data_dict = local_exp.data(0)
                if isinstance(data_dict, dict):
                    feature_names = data_dict.get('names')
                    for key in ['scores', 'values', 'importance']:
                        if key in data_dict:
                            contributions = data_dict[key]
                            break
            except:
                pass

        # Fallback to scores attribute
        if contributions is None and hasattr(local_exp, 'scores'):
            contributions = local_exp.scores
            if contributions is not None and len(contributions) > 0:
                contributions = contributions[0]

        # Use model features if names not found
        if feature_names is None and expected_features is not None:
            feature_names = list(expected_features)

        # Build drivers list
        if contributions is not None and len(contributions) > 0:
            contributions = np.array(contributions).flatten()

            if feature_names and len(feature_names) == len(contributions):
                pairs = list(zip(feature_names, contributions))
            else:
                pairs = [(f"feature_{i}", v) for i, v in enumerate(contributions)]

            # Sort by absolute contribution
            pairs = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)
            total = sum(abs(v) for _, v in pairs) or 1.0

            drivers = [
                {
                    "feature": n,
                    "weight": float(v),
                    "weight_percent": round(abs(v) * 100 / total, 2)
                }
                for n, v in pairs
            ]

            print(f"[SUCCESS] ✓ Extracted {len(drivers)} feature contributions")
    except Exception as e:
        print(f"[WARNING] Explanation extraction failed: {e}")
        drivers = []

    # Create visualization
    print(f"\n[6/6] Creating visualization...")
    plot_path = create_local_explanation_chart(drivers) if drivers else ""
    if plot_path:
        print(f"Chart saved to: {plot_path}")

    # Build result
    result = {
        "prediction_label": label,
        "probability": proba,
        "drivers": drivers,
        "plot_path": plot_path,
    }

    print("\n" + "=" * 70)
    print(f"[PREDICT_WITH_EBM] ✓ COMPLETE: {label} ({proba:.1%} probability)")
    print("=" * 70 + "\n")

    return result