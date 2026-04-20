import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
# Sets wide layout and a custom title/icon
st.set_page_config(page_title="System Phase Predictor", page_icon="🖥️", layout="wide")

# --- LOAD MODEL & SCALER ---
@st.cache_resource # Caches the model so it doesn't reload on every slider click
def load_assets():
    model = joblib.load('system_phase_model.pkl')
    scaler = joblib.load('system_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- SIDEBAR: INPUT CONTROLS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2888/2888698.png", width=80) # Optional generic icon
st.sidebar.title("⚙️ Live Metrics")
st.sidebar.write("Adjust the parameters below to simulate system load.")

cpu_percent = st.sidebar.slider("CPU Percent (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
ram_percent = st.sidebar.slider("RAM Percent (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
process_count = st.sidebar.number_input("Process Count", min_value=0, value=150, step=10)
total_threads = st.sidebar.number_input("Total Threads", min_value=0, value=2000, step=100)
load_index = st.sidebar.slider("Load Index", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
cpu_per_thread = st.sidebar.number_input("CPU per Thread", min_value=0.0, value=0.0150, format="%.4f")
resource_pressure = st.sidebar.number_input("Resource Pressure", min_value=0.0, value=100.0, step=50.0)
rolling_cpu_mean = st.sidebar.slider("Rolling CPU Mean", min_value=0.0, max_value=100.0, value=45.0, step=0.5)

# --- MAIN DASHBOARD AREA ---
st.title("🖥️ System Load Dashboard")
st.markdown("---")

# 1. Top Row: Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Current CPU", value=f"{cpu_percent}%")
col2.metric(label="Current RAM", value=f"{ram_percent}%")
col3.metric(label="Active Processes", value=process_count)
col4.metric(label="Load Index", value=load_index)

st.markdown("---")

# --- PREDICTION LOGIC ---
# Format inputs for the model
input_df = pd.DataFrame([[
    cpu_percent, ram_percent, process_count, total_threads, 
    load_index, cpu_per_thread, resource_pressure, rolling_cpu_mean
]], columns=[
    'cpu_percent', 'ram_percent', 'process_count', 'total_threads', 
    'load_index', 'cpu_per_thread', 'resource_pressure', 'rolling_cpu_mean'
])

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# --- RESULTS & VISUALIZATION ---
col_chart, col_result = st.columns([2, 1])

with col_chart:
    st.subheader("System Footprint")
    # Create a dynamic Radar Chart to visualize the current state
    categories = ['CPU %', 'RAM %', 'Load Index', 'Rolling CPU']
    values = [cpu_percent, ram_percent, load_index, rolling_cpu_mean]
    
    fig = go.Figure(data=go.Scatterpolar(
      r=values + [values[0]], # Close the circle
      theta=categories + [categories[0]],
      fill='toself',
      line_color='#4CAF50' if prediction == 'normal' else ('#FFC107' if prediction == 'moderate' else '#F44336')
    ))
    fig.update_layout(
      polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
      showlegend=False,
      margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_result:
    st.subheader("Predicted Phase")
    st.write("Based on the current metrics, the ML model classifies the system state as:")
    
    # Display highly visible status banners
    if prediction == 'normal':
        st.success("🟢 **NORMAL**")
        st.info("System is operating within safe parameters. No action required.")
    elif prediction == 'moderate':
        st.warning("🟡 **MODERATE**")
        st.info("System load is elevating. Monitor for potential bottlenecks.")
    else:
        st.error("🔴 **HIGH**")
        st.info("Critical load detected! Resources are under heavy pressure.")