import os
import json
import streamlit as st
import pandas as pd
import requests

# -----------------------------
# Settings
# -----------------------------
DEFAULT_API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Toyota Corolla Price Predictor", layout="centered")

# Sidebar: API configuration
st.sidebar.header("API settings")
api_url = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_API_URL, help="Example: http://127.0.0.1:8000")
st.sidebar.caption("Change this if your FastAPI app runs on another port (e.g., 8001).")

# -----------------------------
# Utilities
# -----------------------------
REQUIRED_COLS = [
    "Model","Fuel_Type","Color","Age_08_04","KM","HP","Doors","Gears",
    "Quarterly_Tax","Weight","CC","Met_Color","Automatic","Mfr_Guarantee",
    "BOVAG_Guarantee","Guarantee_Period","ABS","Airbag_1","Airbag_2","Airco",
    "Automatic_airco","Boardcomputer","CD_Player","Central_Lock","Powered_Windows",
    "Power_Steering","Radio","Mistlamps","Sport_Model","Backseat_Divider",
    "Metallic_Rim","Radio_cassette","Parking_Assistant","Tow_Bar"
]

DEFAULTS = {
    "Met_Color": 1, "Automatic": 0, "Mfr_Guarantee": 0, "BOVAG_Guarantee": 0,
    "Guarantee_Period": 0, "ABS": 1, "Airbag_1": 1, "Airbag_2": 1, "Airco": 1,
    "Automatic_airco": 0, "Boardcomputer": 0, "CD_Player": 1, "Central_Lock": 1,
    "Powered_Windows": 1, "Power_Steering": 1, "Radio": 1, "Mistlamps": 0,
    "Sport_Model": 0, "Backseat_Divider": 0, "Metallic_Rim": 0, "Radio_cassette": 0,
    "Parking_Assistant": 0, "Tow_Bar": 0
}

def parse_single_response(resp_json):
    # Accept number or {"predicted_price": number} or generic dict with one numeric
    if isinstance(resp_json, (int, float)):
        return float(resp_json)
    if isinstance(resp_json, dict):
        if "predicted_price" in resp_json:
            return float(resp_json["predicted_price"])
        # fallback: first numeric value
        for v in resp_json.values():
            if isinstance(v, (int, float)):
                return float(v)
    return None

def parse_batch_response(resp_json):
    if isinstance(resp_json, list):
        return [float(x) for x in resp_json]
    if isinstance(resp_json, dict):
        if "predictions" in resp_json and isinstance(resp_json["predictions"], list):
            return [float(x) for x in resp_json["predictions"]]
        if "predicted_prices" in resp_json and isinstance(resp_json["predicted_prices"], list):
            return [float(x) for x in resp_json["predicted_prices"]]
    return None

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fill defaults for any missing columns
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, None)
    # Keep only required columns, in model-expected order
    df = df[REQUIRED_COLS]
    # Basic type hygiene (strings for categoricals that are strings in your single form)
    for col in ["Model", "Fuel_Type", "Color"]:
        df[col] = df[col].astype(str)
    # Integers where they should be integers (safe cast with errors='ignore')
    int_like = ["Age_08_04","KM","HP","Doors","Gears","Quarterly_Tax","Weight","CC",
                "Met_Color","Automatic","Mfr_Guarantee","BOVAG_Guarantee","Guarantee_Period",
                "ABS","Airbag_1","Airbag_2","Airco","Automatic_airco","Boardcomputer",
                "CD_Player","Central_Lock","Powered_Windows","Power_Steering","Radio",
                "Mistlamps","Sport_Model","Backseat_Divider","Metallic_Rim","Radio_cassette",
                "Parking_Assistant","Tow_Bar"]
    for col in int_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULTS.get(col, 0)).astype(int)
    return df

def post_json(endpoint: str, payload):
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# -----------------------------
# UI
# -----------------------------
st.title("🚗 Toyota Corolla Price Predictor")
st.markdown("Enter car features below or upload a file for batch predictions.")

model_options = [
    'Other',
    'TOYOTA Corolla 1.4 16V VVT I 3DR TERRA COMFORT 2/3-Doors',
    'TOYOTA Corolla 1.4 16V VVT I 5DR TERRA COMFORT 4/5-Doors',
    'TOYOTA Corolla 1.6 16V VVT I LIFTB LUNA 4/5-Doors',
    'TOYOTA Corolla 1.6 16V VVT I 5DR SOL 4/5-Doors',
    'TOYOTA Corolla 1.4 16V VVT I LIFTB TERRA 4/5-Doors',
    'TOYOTA Corolla 1.6 16V VVT I HATCHB TERRA 2/3-Doors',
    'TOYOTA Corolla 1.4 16V VVT I HATCHB TERRA 2/3-Doors',
    'TOYOTA Corolla 1.6 16V VVT I LIFTB TERRA 4/5-Doors',
    'TOYOTA Corolla 1.6 16V VVT I LIFTB SOL 4/5-Doors',
    'TOYOTA Corolla 1.6 16V VVT I SEDAN TERRA 4/5-Doors',
    'TOYOTA Corolla 1.6 16V VVT I HATCHB G6 2/3-Doors',
    'TOYOTA Corolla 1.6 16V HATCHB LINEA TERRA 2/3-Doors',
    'TOYOTA Corolla 1.6 16V WAGON LINEA TERRA Stationwagen',
    'TOYOTA Corolla 1.3 16V LIFTB LINEA TERRA 4/5-Doors',
    'TOYOTA Corolla 2.0 DSL LIFTB LINEA TERRA 4/5-Doors',
    'TOYOTA Corolla 1.6 16V LIFTB LINEA LUNA 4/5-Doors',
    'TOYOTA Corolla 2.0 DSL HATCHB LINEA TERRA 2/3-Doors',
    'TOYOTA Corolla 1.3 16V HATCHB LINEA TERRA 2/3-Doors',
    'TOYOTA Corolla 1.6 16V HATCHB S 2/3-Doors',
    'TOYOTA Corolla 1.6 16V LIFTB LINEA TERRA 4/5-Doors',
    'TOYOTA Corolla 1.6 16V HATCHB LINEA LUNA 2/3-Doors',
    'TOYOTA Corolla 1.6 16V HATCHB G6 2/3-Doors',
    'TOYOTA Corolla 1.6 16V SEDAN LINEA TERRA 4/5-Doors'
]

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# -----------------------------
# Single prediction tab
# -----------------------------
with tab_single:
    with st.expander("🔧 Engine & Performance", expanded=True):
        age = st.number_input("Age_08_04 (months)", min_value=1, max_value=80, value=40)
        km = st.number_input("KM", min_value=0, max_value=350000, value=3000)
        hp = st.number_input("HP", min_value=69, max_value=192, value=92)
        cc = st.number_input("CC", min_value=1300, max_value=2000, value=1600)

    with st.expander("🚗 Body & Features", expanded=True):
        doors = st.number_input("Doors", min_value=2, max_value=5, value=5)
        gears = st.number_input("Gears", min_value=3, max_value=6, value=5)
        weight = st.number_input("Weight", min_value=1000, max_value=1600, value=1070)
        tax = st.number_input("Quarterly_Tax", min_value=19, max_value=283, value=100)

    with st.expander("🎨 Appearance", expanded=True):
        fuel = st.selectbox("Fuel_Type", ["Petrol", "Diesel", "CNG"])
        color = st.selectbox("Color", ["Black", "Blue", "Red", "Silver", "White", "Grey", "Green", "Other"])
        model = st.selectbox("Model", model_options)

    if st.button("Predict price", use_container_width=True):
        single_payload = {
            "Model": model,
            "Fuel_Type": fuel,
            "Color": color,
            "Age_08_04": age,
            "KM": km,
            "HP": hp,
            "Doors": doors,
            "Gears": gears,
            "Quarterly_Tax": tax,
            "Weight": weight,
            "CC": cc,
            **DEFAULTS
        }
        try:
            resp_json = post_json("/predict", single_payload)
            price = parse_single_response(resp_json)
            if price is not None:
                st.success(f"💰 Predicted Price: ${price:,.2f}")
            else:
                st.error(f"Unexpected response:\n{json.dumps(resp_json, indent=2)}")
        except requests.exceptions.HTTPError as e:
            st.error(f"Server returned {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# -----------------------------
# Batch prediction tab
# -----------------------------
with tab_batch:
    st.markdown("Upload a CSV or Excel with one row per car. You can download a template below.")
    # Create a template users can download
    template_df = pd.DataFrame([{
        "Model": "Other",
        "Fuel_Type": "Petrol",
        "Color": "Black",
        "Age_08_04": 40,
        "KM": 3000,
        "HP": 92,
        "Doors": 5,
        "Gears": 5,
        "Quarterly_Tax": 100,
        "Weight": 1070,
        "CC": 1600,
        **DEFAULTS
    }])[REQUIRED_COLS]
    st.download_button(
        "Download CSV template",
        template_df.to_csv(index=False),
        file_name="toyota_batch_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.subheader("Preview")
        st.dataframe(df_in.head(), use_container_width=True)

        # Validate schema and fill defaults
        missing = [c for c in REQUIRED_COLS if c not in df_in.columns]
        if missing:
            st.warning(f"Missing columns will be auto-filled with defaults: {missing}")

        df_ready = ensure_schema(df_in)
        batch_payload = df_ready.to_dict(orient="records")

        if st.button("Predict batch", type="primary", use_container_width=True):
            try:
                resp_json = post_json("/predict_batch", batch_payload)
                preds = parse_batch_response(resp_json)
                if preds is None:
                    st.error(f"Unexpected response:\n{json.dumps(resp_json, indent=2)}")
                else:
                    if len(preds) != len(df_in):
                        st.warning("Prediction count differs from input rows. Showing aligned results with prepared data.")
                        base_df = df_ready.copy()
                    else:
                        base_df = df_in.copy()
                    out_df = base_df.assign(Predicted_Price=preds)
                    st.subheader("Predictions")
                    st.dataframe(out_df, use_container_width=True)
                    st.download_button(
                        "Download predictions CSV",
                        out_df.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except requests.exceptions.HTTPError as e:
                st.error(f"Server returned {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
