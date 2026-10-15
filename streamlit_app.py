import streamlit as st
import requests
import pandas as pd
import pickle
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Insurance Premium Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS & DATA FROM APP.PY ---
TIER_1_CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
TIER_2_CITIES = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

OCCUPATION_MAP = {
    "Private Job": "private_job",
    "Government Job": "government_job",
    "Business Owner": "business_owner",
    "Freelancer": "freelancer",
    "Student": "student",
    "Retired": "retired",
    "Unemployed": "unemployed"
}

REVERSE_OCCUPATION_MAP = {v: k for k, v in OCCUPATION_MAP.items()}

# Sample Presets for Quick Testing
SAMPLE_PROFILES = {
    "Custom (Manual Input)": None,
    "Healthy Young Student": {
        "age": 21,
        "weight": 62.0,
        "height": 1.74,
        "income_lpa": 3.0,
        "smoker": False,
        "city": "Pune",
        "occupation": "student"
    },
    "Middle-Aged Corporate Professional": {
        "age": 38,
        "weight": 76.0,
        "height": 1.72,
        "income_lpa": 18.5,
        "smoker": False,
        "city": "Bangalore",
        "occupation": "private_job"
    },
    "Senior Smoker Business Owner": {
        "age": 55,
        "weight": 92.0,
        "height": 1.70,
        "income_lpa": 25.0,
        "smoker": True,
        "city": "Mumbai",
        "occupation": "business_owner"
    }
}

# --- CACHED MODEL LOADER (FOR STANDALONE MODE) ---
@st.cache_resource
def load_local_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

local_model = load_local_model()

# --- COMPUTATION HELPERS (MATCHING APP.PY USERINPUT LOGIC) ---
def compute_bmi(weight: float, height: float) -> float:
    if height <= 0:
        return 0.0
    return round(weight / (height ** 2), 2)

def compute_bmi_status(bmi: float) -> tuple[str, str]:
    if bmi < 18.5:
        return "Underweight", "#3B82F6"
    elif 18.5 <= bmi < 25.0:
        return "Normal Weight", "#10B981"
    elif 25.0 <= bmi < 30.0:
        return "Overweight", "#F59E0B"
    else:
        return "Obese", "#EF4444"

def compute_age_group(age: int) -> str:
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"

def compute_lifestyle_risk(smoker: bool, bmi: float) -> str:
    if smoker and bmi > 30:
        return "high"
    elif smoker or bmi > 27:
        return "medium"
    else:
        return "low"

def compute_city_tier(city: str) -> int:
    if city in TIER_1_CITIES:
        return 1
    elif city in TIER_2_CITIES:
        return 2
    return 3

# --- API HEALTH CHECK HELPER ---
def check_api_status(api_url: str) -> bool:
    try:
        res = requests.get(api_url.rstrip("/"), timeout=1.5)
        return res.status_code == 200
    except Exception:
        # Fallback to check docs/openapi
        try:
            res = requests.get(f"{api_url.rstrip('/')}/openapi.json", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False


# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .result-container {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .result-low {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
    }
    .result-medium {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        color: #92400E;
    }
    .result-high {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)


# --- SIDEBAR: SETTINGS & CONNECTION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Settings & Backend")
    st.markdown("Configure prediction source and connection.")

    backend_mode = st.radio(
        "Prediction Mode:",
        ["FastAPI Server (REST API)", "Direct Model (Standalone .pkl)"],
        index=0,
        help="Choose whether to request predictions through the FastAPI app.py endpoint or run directly using model.pkl."
    )

    api_base_url = "http://127.0.0.1:8000"
    if backend_mode == "FastAPI Server (REST API)":
        api_base_url = st.text_input("FastAPI Base URL:", value="http://127.0.0.1:8000")
        
        is_online = check_api_status(api_base_url)
        if is_online:
            st.success("🟢 **FastAPI Server Online**")
        else:
            st.warning("🔴 **FastAPI Server Offline**")
            st.caption("To run the FastAPI backend, execute in your terminal:")
            st.code("uvicorn app:app --reload", language="bash")
            st.caption("*(You can also switch to 'Direct Model' mode above to test without the server)*")

    st.divider()

    st.subheader("⚡ Quick Load Preset")
    preset_choice = st.selectbox(
        "Load sample profile:",
        list(SAMPLE_PROFILES.keys()),
        index=0
    )

    st.divider()
    st.caption("Developed for FastAPI & Scikit-Learn Model Pipeline")


# --- MAIN HEADER ---
st.title("🛡️ Health Insurance Premium Predictor")
st.markdown("Predict the insurance premium risk tier (**Low**, **Medium**, or **High**) using machine learning based on demographic, lifestyle, and financial attributes.")

# --- PREFILL VALUES FROM PRESET IF SELECTED ---
current_preset = SAMPLE_PROFILES[preset_choice]

default_age = current_preset["age"] if current_preset else 28
default_weight = current_preset["weight"] if current_preset else 68.0
default_height = current_preset["height"] if current_preset else 1.75
default_income = current_preset["income_lpa"] if current_preset else 10.0
default_smoker = current_preset["smoker"] if current_preset else False
default_city = current_preset["city"] if current_preset else "Mumbai"
default_occ_key = REVERSE_OCCUPATION_MAP.get(current_preset["occupation"], "Private Job") if current_preset else "Private Job"


# --- INPUT FORM & REAL-TIME DASHBOARD ---
col1, col2, col3 = st.columns([1.1, 1.1, 1.2], gap="large")

with col1:
    st.subheader("👤 Personal & Health")
    
    age = st.number_input(
        "Age (Years):",
        min_value=1,
        max_value=119,
        value=int(default_age),
        step=1,
        help="Valid age range: 1 to 119 years."
    )

    height = st.number_input(
        "Height (Meters):",
        min_value=0.50,
        max_value=2.49,
        value=float(default_height),
        step=0.01,
        format="%.2f",
        help=f"E.g., 1.75 m (~{int(round(float(default_height)*100))} cm)"
    )
    st.caption(f"📏 Equivalent: ~{round(height * 100, 1)} cm / {round(height * 3.28084, 1)} ft")

    weight = st.number_input(
        "Weight (Kilograms):",
        min_value=10.0,
        max_value=250.0,
        value=float(default_weight),
        step=0.5,
        format="%.1f",
        help="Weight of the individual in kg."
    )

    smoker_bool = st.radio(
        "Do you smoke regularly?",
        options=[False, True],
        format_func=lambda x: "🚬 Yes, Smoker" if x else "🚭 No, Non-Smoker",
        index=1 if default_smoker else 0,
        horizontal=True
    )

with col2:
    st.subheader("📍 Location & Career")

    # City list with Tier 1 first, Tier 2, then custom
    all_known_cities = sorted(list(set(TIER_1_CITIES + TIER_2_CITIES)))
    city_options = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune"] + [c for c in all_known_cities if c not in TIER_1_CITIES] + ["Other (Specify below)"]
    
    # Selected city index
    city_index = city_options.index(default_city) if default_city in city_options else city_options.index("Other (Specify below)")
    selected_city_option = st.selectbox("City of Residence:", options=city_options, index=city_index)

    if selected_city_option == "Other (Specify below)":
        city_input = st.text_input("Enter City Name:", value=default_city if default_city not in city_options else "Mysore")
    else:
        city_input = selected_city_option

    occupation_label = st.selectbox(
        "Occupation:",
        options=list(OCCUPATION_MAP.keys()),
        index=list(OCCUPATION_MAP.keys()).index(default_occ_key)
    )
    occupation_value = OCCUPATION_MAP[occupation_label]

    income_lpa = st.number_input(
        "Annual Income (in Lakhs INR / LPA):",
        min_value=0.1,
        max_value=300.0,
        value=float(default_income),
        step=0.5,
        format="%.1f",
        help="Gross annual salary/income in LPA."
    )


# --- REAL-TIME DERIVED METRICS ---
calc_bmi = compute_bmi(weight, height)
bmi_status, bmi_color = compute_bmi_status(calc_bmi)
calc_age_group = compute_age_group(age)
calc_lifestyle_risk = compute_lifestyle_risk(smoker_bool, calc_bmi)
calc_city_tier = compute_city_tier(city_input)

with col3:
    st.subheader("📊 Computed Health Metrics")
    st.caption("Values derived dynamically by FastAPI Pydantic schema:")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Body Mass Index (BMI)", f"{calc_bmi}")
        st.markdown(f"Status: <b style='color:{bmi_color};'>{bmi_status}</b>", unsafe_allow_html=True)
    with m_col2:
        st.metric("Lifestyle Risk", calc_lifestyle_risk.upper())
        st.caption("Based on smoking & BMI")

    st.divider()

    m_col3, m_col4 = st.columns(2)
    with m_col3:
        st.metric("Age Group", calc_age_group.replace("_", " ").title())
        st.caption(f"Bracket for age {age}")
    with m_col4:
        st.metric("City Tier", f"Tier {calc_city_tier}")
        st.caption(f"{city_input.title()}")


# --- PREDICTION ACTION ---
st.write("")
predict_btn = st.button("🔮 Predict Insurance Premium Category", type="primary", use_container_width=True)

# Prepare Payload matching UserInput
payload = {
    "age": int(age),
    "weight": float(weight),
    "height": float(height),
    "income_lpa": float(income_lpa),
    "smoker": bool(smoker_bool),
    "city": city_input.strip(),
    "occupation": occupation_value
}

if predict_btn:
    with st.spinner("Analyzing risk factors and predicting premium..."):
        time.sleep(0.15)
        prediction_category = None
        probabilities = None
        response_data = None
        api_error = None
        start_time = time.time()

        if backend_mode == "FastAPI Server (REST API)":
            predict_endpoint = f"{api_base_url.rstrip('/')}/predict"
            try:
                response = requests.post(predict_endpoint, json=payload, timeout=5)
                latency = round((time.time() - start_time) * 1000, 1)
                if response.status_code == 200:
                    response_data = response.json()
                    prediction_category = response_data.get("insurance_premium_category")
                else:
                    api_error = f"API returned HTTP {response.status_code}: {response.text}"
            except requests.exceptions.ConnectionError:
                api_error = f"Could not connect to FastAPI at {predict_endpoint}. Is the server running?"
            except Exception as e:
                api_error = f"Request error: {str(e)}"

            # Graceful fallback offer if API was unreachable but local model is present
            if api_error and local_model is not None:
                st.warning(f"⚠️ {api_error}")
                st.info("Falling back to local model.pkl for prediction...")
                input_df = pd.DataFrame([{
                    'bmi': calc_bmi,
                    'age_group': calc_age_group,
                    'lifestyle_risk': calc_lifestyle_risk,
                    'city_tier': calc_city_tier,
                    'income_lpa': income_lpa,
                    'occupation': occupation_value
                }])
                prediction_category = local_model.predict(input_df)[0]
                if hasattr(local_model, "predict_proba"):
                    probabilities = dict(zip(local_model.classes_, local_model.predict_proba(input_df)[0]))

        else:
            # Standalone direct model mode
            if local_model is not None:
                input_df = pd.DataFrame([{
                    'bmi': calc_bmi,
                    'age_group': calc_age_group,
                    'lifestyle_risk': calc_lifestyle_risk,
                    'city_tier': calc_city_tier,
                    'income_lpa': income_lpa,
                    'occupation': occupation_value
                }])
                prediction_category = local_model.predict(input_df)[0]
                if hasattr(local_model, "predict_proba"):
                    probabilities = dict(zip(local_model.classes_, local_model.predict_proba(input_df)[0]))
            else:
                api_error = "Could not find model.pkl file in the working directory."

        # Display Results
        if prediction_category:
            st.write("---")
            st.subheader("🎯 Prediction Result")

            badge_class = "result-low"
            icon = "🟢"
            description = "Favorable risk profile! The insured individual is assessed at low actuarial risk, qualifying for standard or discounted premium tiers."
            
            if prediction_category.lower() == "medium":
                badge_class = "result-medium"
                icon = "🟡"
                description = "Moderate risk profile. The insured individual has typical risk metrics and qualifies for baseline standard premium plans."
            elif prediction_category.lower() == "high":
                badge_class = "result-high"
                icon = "🔴"
                description = "Elevated risk profile! Identified risk factors (such as smoking, high BMI, age category, or occupation risk) lead to higher estimated premiums."

            st.markdown(f"""
            <div class="result-container {badge_class}">
                <h2>{icon} Premium Category: {prediction_category.upper()}</h2>
                <p style="font-size: 1.1rem; margin-top: 0.5rem;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

            # If probabilities are available, show visual bar breakdown
            if probabilities is None and local_model is not None:
                input_df = pd.DataFrame([{
                    'bmi': calc_bmi,
                    'age_group': calc_age_group,
                    'lifestyle_risk': calc_lifestyle_risk,
                    'city_tier': calc_city_tier,
                    'income_lpa': income_lpa,
                    'occupation': occupation_value
                }])
                if hasattr(local_model, "predict_proba"):
                    probabilities = dict(zip(local_model.classes_, local_model.predict_proba(input_df)[0]))

            if probabilities:
                st.write("")
                st.write("**Model Class Probabilities:**")
                prob_df = pd.DataFrame({
                    "Category": list(probabilities.keys()),
                    "Probability (%)": [round(v * 100, 1) for v in probabilities.values()]
                }).sort_values(by="Probability (%)", ascending=False)
                
                st.bar_chart(prob_df.set_index("Category"))

        elif api_error:
            st.error(f"❌ Prediction Failed: {api_error}")


# --- DEVELOPER ACCORDION ---
with st.expander("🛠️ API Debug & JSON Payload Inspector"):
    st.write("**Payload sent to `/predict`:**")
    st.json(payload)
    
    st.write("**Derived Features used by the ML Model (`model.pkl`):**")
    st.json({
        "bmi": calc_bmi,
        "age_group": calc_age_group,
        "lifestyle_risk": calc_lifestyle_risk,
        "city_tier": calc_city_tier,
        "income_lpa": income_lpa,
        "occupation": occupation_value
    })

    st.write("**CURL Command Equivalent:**")
    st.code(f"""curl -X POST "{api_base_url.rstrip('/')}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{pd.Series(payload).to_json()}'""", language="bash")
