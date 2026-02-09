import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import time
import plotly.express as px
from shared import auth, db

# --- Configuration ---
REFRESH_RATE = 2 # Seconds

# --- UI Components ---
def login_screen():
    st.title("🛡️ Sentinel Public Dashboard")
    st.subheader("Login to view crowd safety status")
    
    # Create Tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    # --- Login Tab ---
    with tab1:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if auth.verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    # --- Sign Up Tab ---
    with tab2:
        st.subheader("Create a new account")
        with st.form("signup"):
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submitted_signup = st.form_submit_button("Sign Up")
            
            if submitted_signup:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match!")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    if auth.create_user(new_user, new_pass):
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username already exists.")

def dashboard():
    # --- Sidebar ---
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    # --- Main ---
    st.title("🏙️ Live Crowd Status")
    
    # Live Data Polling
    database = db.get_db()
    latest = database.get_latest_event()
    
    if not latest:
        st.warning("No live data available. System might be offline.")
        return

    # Metrics
    # Define colors for risk
    risk_color = "normal"
    if latest['risk_level'] == "MEDIUM": risk_color = "off" # Orangeish in delta?
    if latest['risk_level'] == "HIGH": risk_color = "inverse" # Red in delta

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", latest['risk_level'], delta="Live", delta_color=risk_color)
    c2.metric("People Count", latest['people_count'])
    c3.metric("Last Update", latest['timestamp'].strftime("%H:%M:%S"))

    # Historical Chart
    st.divider()
    st.subheader("Crowd Density Trend (Last 50 Updates)")
    
    history = database.get_recent_history(limit=50)
    if history:
        df = pd.DataFrame(history)
        fig = px.line(df, x="timestamp", y="people_count", title="People Count over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto-Refresh Logic using empty placeholder sleep hack or st_autorefresh
    # Using simple sleep + rerun for prototype stability
    time.sleep(REFRESH_RATE)
    st.rerun()

def main():
    st.set_page_config(page_title="Sentinel Public Dashboard", page_icon="🏙️")
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if st.session_state.authenticated:
        dashboard()
    else:
        login_screen()

if __name__ == "__main__":
    main()
