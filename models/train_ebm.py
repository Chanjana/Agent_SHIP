"""
EBM Model Training Script - Agent_SHIP
Training EBM model on Full_SC_data.csv dataset
"""
import pandas as pd
import numpy as np
import time
import joblib
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

print("AGENT_SHIP - EBM Model Training")

# Load data
print("\n[1/6] Loading dataset...")
df = pd.read_csv('../data/Full_SC_data.csv', encoding='latin1')
print(f"Loaded {len(df)} rows")

# Cleaning column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df = df.rename(columns={
    'days_for_shipping_(real)': 'real_shipping_days',
    'days_for_shipment_(scheduled)': 'scheduled_shipping_days'
})

# Creating target variable
print("\n[2/6] Creating target variable...")
df = df.dropna(subset=['real_shipping_days', 'scheduled_shipping_days'])
df['shipment_delay'] = (df['real_shipping_days'] > df['scheduled_shipping_days']).astype(int)

print(f"After target construction: {len(df)} rows")
print(f"Target distribution:")
print(df['shipment_delay'].value_counts(normalize=True))

# Feature engineering
print("\n[3/6] Engineering features...")

# Shipping date features
df['shipping_date'] = pd.to_datetime(df['shipping_date_(dateorders)'], errors='coerce')
df['ship_hour'] = df['shipping_date'].dt.hour.fillna(-1)
df['ship_dayofweek'] = df['shipping_date'].dt.dayofweek.fillna(-1)
df['ship_month'] = df['shipping_date'].dt.month.fillna(-1)
df['is_weekend'] = df['ship_dayofweek'].isin([5, 6]).astype(int)
df['is_peak_hour'] = df['ship_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
df['is_night'] = df['ship_hour'].isin([0, 1, 2, 3, 4, 22, 23]).astype(int)

# Handling missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna("UNKNOWN")

print(f"Missing values handled")

# Capping outliers using IQR
def cap_outliers_iqr(df, cols):
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

cap_features = [
    'order_item_total',
    'sales',
    'product_price',
    'order_item_discount',
    'order_profit_per_order'
]
df = cap_outliers_iqr(df, cap_features)
print(f"Outliers capped for: {cap_features}")

# Selecting final features
include_cols = [
    'shipment_delay',
    'scheduled_shipping_days',
    'ship_hour',
    'ship_dayofweek',
    'ship_month',
    'is_weekend',
    'is_peak_hour',
    'is_night',
    'shipping_mode',
    'type',
    'order_item_quantity',
    'order_item_total',
    'order_item_discount',
    'order_item_discount_rate',
    'sales',
    'order_profit_per_order',
    'market',
    'latitude',
    'longitude',
    'product_price'
]

df = df[[c for c in include_cols if c in df.columns]]
print(f"Final dataset: {len(df)} rows × {len(df.columns)} columns")
print(f"Features: {[c for c in df.columns if c != 'shipment_delay']}")

# Train/Test split
print("\n[4/6] Splitting data...")
X = df.drop(columns=['shipment_delay'])
y = df['shipment_delay'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert categorical columns to string
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
X_train[cat_cols] = X_train[cat_cols].astype(str)
X_test[cat_cols] = X_test[cat_cols].astype(str)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training class distribution:")
print(y_train.value_counts(normalize=True))

# Train EBM
print("\n[5/6] Training EBM model...")
start_time = time.time()

ebm = ExplainableBoostingClassifier(
    interactions=4,
    learning_rate=0.02333040808923146,
    max_rounds=13602,
    max_bins=228,
    max_interaction_bins=80,
    outer_bags=1,
    validation_size=0.16218731234599282,
    early_stopping_rounds=93,
    n_jobs=-1,
    random_state=42
)

ebm.fit(X_train, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Model Evaluation
print("\n[6/6] Evaluating model...")
y_pred = ebm.predict(X_test)
y_proba = ebm.predict_proba(X_test)[:, 1]

print("\n--- Performance of the Test Set---")
print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              On-Time  Delayed")
print(f"Actual On-Time   {cm[0][0]:>5}   {cm[0][1]:>5}")
print(f"       Delayed   {cm[1][0]:>5}   {cm[1][1]:>5}")

# Feature importance
print("\nFeature Importance")


ebm_global = ebm.explain_global()
importance_df = pd.DataFrame({
    'feature': ebm_global.data()['names'],
    'importance': ebm_global.data()['scores']
})
importance_df = importance_df.sort_values('importance', ascending=False)

print("\nTop 15 Features:")
print(importance_df.head(15).to_string(index=False))

# Plot
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(top_features['feature'], top_features['importance'], color='#7c4dff')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Top 15 Feature Importance - Agent_SHIP')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150, bbox_inches='tight')
print("\n✓Plot saved: feature_importances.png")

# Saving trained model
print("\n" + "-" * 70)
print("Saving Model")
print("-" * 70)

joblib.dump(ebm, 'ebm_model.pkl')
print("Model saved: ebm_model.pkl")

with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(X.columns.tolist()))
print("Features saved: feature_names.txt")

print("\n" + "=" * 70)
print(" EBM Model Training Completed...")
print(f"\nModel trained on {len(X.columns)} features:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")