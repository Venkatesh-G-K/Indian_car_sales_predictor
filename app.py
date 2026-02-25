import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Indian Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.4rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1.05rem; margin: 0.5rem 0 0 0; }

    .price-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 2px solid #e94560;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .price-card .label { color: #a8b2d8; font-size: 1rem; margin-bottom: 0.5rem; }
    .price-card .price { color: #e94560; font-size: 3rem; font-weight: 800; }
    .price-card .lakh  { color: #4ecca3; font-size: 1.3rem; margin-top: 0.3rem; }

    .metric-box {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-box .m-label { color: #a8b2d8; font-size: 0.85rem; }
    .metric-box .m-value { color: #4ecca3; font-size: 1.4rem; font-weight: 700; }

    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c62a47);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(233,69,96,0.4); }

    .tip-box {
        background: #0d1b2a;
        border-left: 4px solid #4ecca3;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #a8b2d8;
        font-size: 0.9rem;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label { color: #a8b2d8 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
BRANDS = sorted([
    'Ashok Leyland','Aston Martin','Audi','BMW','BYD','Bajaj','Bentley','Bugatti',
    'Chevrolet','Citroen','Datsun','Fiat','Force','Force Motors','Ford',
    'Hindustan Motors','Honda','Hyundai','Isuzu','Jaguar','Jeep','Kia',
    'Lamborghini','Land Rover','Lexus','MG','Mahindra','Mahindra Renault',
    'Mahindra Ssangyong','Mahindra-Renault','Maruti Suzuki','Maserati',
    'Mercedes-Benz','Mini','Mitsubishi','Nissan','Opel','Porsche','Renault',
    'Rolls-Royce','Skoda','Ssangyong','Tata','Toyota','Volkswagen','Volvo'
])

TOP_MODELS = [
    'Other','Maruti Swift','Maruti Wagon R','Hyundai i20','Maruti Alto 800',
    'Maruti Swift Dzire','Hyundai Creta','Maruti Baleno','Honda City',
    'Hyundai Grand i10','Kia Seltos','Hyundai i10','Tata Nexon','Tata Tiago',
    'Renault KWID','Maruti Alto K10','Maruti Ertiga','Maruti Celerio',
    'Honda Amaze','Hyundai Venue','Hyundai Verna','Maruti Vitara Brezza',
    'Ford Ecosport','Mahindra XUV500','Hyundai EON','Mahindra Scorpio',
    'Maruti Ciaz','Kia Sonet','Mahindra Bolero','Tata Thar','Mahindra Thar'
]

CITIES = [
    'Other','Mumbai','New Delhi','Pune','Chennai','Bangalore','Hyderabad',
    'Jaipur','Ahmedabad','Kolkata','Gurgaon','Lucknow','Thane','Indore',
    'Nagpur','Surat','Coimbatore','Chandigarh','Bhopal','Kochi','Noida'
]

BODYTYPES = [
    'SUV','Hatchback','Sedan','MUV','Coupe','Minivan','Convertible',
    'Wagon','CompactSuv','SubCompactSuv','FullSizeSuv','SuvCoupe',
    'CompactSedan','StationWagon','MPV_MUV','Pickup Trucks','Truck'
]

FUEL_TYPES    = ['Petrol','Diesel','CNG','Electric','Hybrid','LPG']
TRANSMISSIONS = ['Manual','Automatic']

LUXURY_BRANDS = {"Mercedes-Benz","BMW","Audi","Jaguar","Porsche",
                  "Volvo","Land Rover","Lexus","Bentley","Ferrari",
                  "Rolls-Royce","Lamborghini","Maserati","Aston Martin","Bugatti"}

# ── Load model artifacts ──────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load model, scaler, encoders - works locally and on Streamlit Cloud."""
    base = os.path.dirname(os.path.abspath(__file__))

    # Build exhaustive search list
    search_dirs = [base, os.path.join(base, "ml_output")]
    try:
        for item in os.listdir(base):
            full = os.path.join(base, item)
            if os.path.isdir(full) and "ml_output" in item:
                search_dirs.append(full)
    except Exception:
        pass

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            files = os.listdir(d)
        except Exception:
            continue
        for fname in files:
            if fname.startswith("best_model") and fname.endswith(".pkl"):
                model_path   = os.path.join(d, fname)
                scaler_path  = os.path.join(d, "scaler.pkl")
                encoder_path = os.path.join(d, "encoders.pkl")
                if os.path.exists(scaler_path) and os.path.exists(encoder_path):
                    try:
                        return (joblib.load(model_path),
                                joblib.load(scaler_path),
                                joblib.load(encoder_path),
                                fname.replace("best_model_","").replace(".pkl","").replace("_"," "))
                    except Exception as e:
                        st.error(f"Found files but failed to load: {e}")
                        return None, None, None, None

    # Debug info to diagnose on Streamlit Cloud
    st.warning(f"Searched in: {search_dirs}")
    st.warning(f"Files at root: {os.listdir(base)}")
    try:
        ml_path = os.path.join(base, "ml_output")
        if os.path.isdir(ml_path):
            st.warning(f"Files in ml_output/: {os.listdir(ml_path)}")
    except Exception:
        pass
    return None, None, None, None

# ── Prediction logic ──────────────────────────────────────────
def predict_price(model, scaler, encoders,
                  manufacturing_year, km_driven, fuel_type,
                  transmission_type, brand, city, bodytype,
                  number_of_owners, model_name="Other"):

    car_age     = 2025 - manufacturing_year
    km_per_year = km_driven / max(car_age, 1)
    is_luxury   = int(brand in LUXURY_BRANDS)

    def simplify_fuel(f):
        f = str(f).lower()
        if "electric" in f and "petrol" not in f and "diesel" not in f: return "Electric"
        elif "plug" in f or "hybrid" in f: return "Hybrid"
        elif "cng" in f: return "CNG"
        elif "lpg" in f: return "LPG"
        elif "diesel" in f: return "Diesel"
        return "Petrol"

    row = {
        "manufacturing_year": manufacturing_year,
        "km_driven":          km_driven,
        "fuel_type":          simplify_fuel(fuel_type),
        "transmission_type":  "Manual" if transmission_type == "Manual" else "Automatic",
        "brand":              brand,
        "city":               city,
        "bodytype":           bodytype if bodytype else "Unknown",
        "number_of_owners":   number_of_owners,
        "car_age":            car_age,
        "km_per_year":        km_per_year,
        "is_luxury":          is_luxury,
        "model":              model_name,
    }

    df_row = pd.DataFrame([row])
    for col in ["fuel_type","transmission_type","brand","city","bodytype","model"]:
        if col in encoders:
            le  = encoders[col]
            val = str(df_row[col].iloc[0])
            df_row[col] = le.transform([val])[0] if val in le.classes_ \
                          else (le.transform(["Other"])[0] if "Other" in le.classes_ else 0)

    df_scaled = scaler.transform(df_row)
    return round(float(np.expm1(model.predict(df_scaled)[0])), 2)

def price_range(base_price, pct=12):
    low  = base_price * (1 - pct/100)
    high = base_price * (1 + pct/100)
    return low, high

# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🚗 Indian Used Car Price Predictor</h1>
    <p>AI-powered price estimation trained on 57,000+ real Indian listings</p>
</div>
""", unsafe_allow_html=True)

# Load artifacts
model, scaler, encoders, model_name = load_artifacts()

if model is None:
    st.error("⚠️  Model files not found! Please run `indian_cars_ml_pipeline.py` first to train and save the model, then place the `ml_output_*` folder in the same directory as this app.")
    st.info("Expected files: `best_model_*.pkl`, `scaler.pkl`, `encoders.pkl`")
    st.stop()

st.success(f"✅ Model loaded: **{model_name}**", icon="🤖")

# ── Layout: Form | Results ─────────────────────────────────────
col_form, col_results = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown("### 📋 Car Details")

    c1, c2 = st.columns(2)
    with c1:
        brand = st.selectbox("Brand", BRANDS,
                             index=BRANDS.index("Maruti Suzuki") if "Maruti Suzuki" in BRANDS else 0)
    with c2:
        model_car = st.selectbox("Model (optional)", TOP_MODELS)

    c3, c4 = st.columns(2)
    with c3:
        year = st.selectbox("Manufacturing Year", list(range(2025, 1989, -1)), index=6)
    with c4:
        owners = st.selectbox("Number of Owners", [1,2,3,4,5,6])

    km = st.slider("Kilometres Driven", min_value=500, max_value=300_000,
                   value=45_000, step=500,
                   format="%d km")

    c5, c6 = st.columns(2)
    with c5:
        fuel = st.selectbox("Fuel Type", FUEL_TYPES)
    with c6:
        transmission = st.selectbox("Transmission", TRANSMISSIONS)

    c7, c8 = st.columns(2)
    with c7:
        bodytype = st.selectbox("Body Type", BODYTYPES)
    with c8:
        city = st.selectbox("City", CITIES)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)


with col_results:
    st.markdown("### 💰 Price Estimate")

    if predict_btn:
        with st.spinner("Analyzing..."):
            price = predict_price(
                model, scaler, encoders,
                manufacturing_year=year,
                km_driven=km,
                fuel_type=fuel,
                transmission_type=transmission,
                brand=brand,
                city=city,
                bodytype=bodytype,
                number_of_owners=owners,
                model_name=model_car
            )
            low, high = price_range(price)
            car_age   = 2025 - year
            dep_pct   = min(car_age * 6, 60)

        st.markdown(f"""
        <div class="price-card">
            <div class="label">Estimated Market Price</div>
            <div class="price">₹{price:,.0f}</div>
            <div class="lakh">≈ ₹{price/1e5:.2f} Lakhs</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="tip-box">
            📊 <b>Likely price range:</b> ₹{low/1e5:.1f}L – ₹{high/1e5:.1f}L
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="m-label">Car Age</div>
                <div class="m-value">{car_age} yrs</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="m-label">KM/Year</div>
                <div class="m-value">{km//max(car_age,1):,}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-box">
                <div class="m-label">Est. Depreciation</div>
                <div class="m-value">~{dep_pct}%</div>
            </div>""", unsafe_allow_html=True)

        # Tips
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 💡 Smart Tips")
        tips = []
        if owners > 2:    tips.append("🔴 **3+ owners** significantly reduces resale value. Negotiate hard.")
        if km > 100_000:  tips.append("🟠 **High km** (>1 lakh) — budget extra for maintenance.")
        if car_age > 8:   tips.append("🟠 **Older car** — check for insurance & part availability.")
        if fuel == "Diesel" and km < 50_000:
            tips.append("🟡 Low km Diesel — may not justify the higher initial cost.")
        if fuel == "Electric": tips.append("🟢 **Electric** — lower running costs, check battery health.")
        if brand in LUXURY_BRANDS:
            tips.append("🔵 **Luxury brand** — high maintenance costs; factor in servicing.")
        if not tips: tips.append("✅ Looks like a solid buy! Get an independent inspection before finalizing.")
        for t in tips: st.markdown(t)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #a8b2d8;">
            <div style="font-size: 4rem;">🚗</div>
            <div style="font-size: 1.1rem; margin-top: 1rem;">
                Fill in the car details on the left<br>and click <b>Predict Price</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.85rem'>"
    "Trained on 57,000+ Indian used car listings · Prices are estimates only · "
    "Always verify with a certified mechanic before purchasing"
    "</div>",
    unsafe_allow_html=True
)
