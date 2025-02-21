import streamlit as st

# Set page title and layout
st.set_page_config(page_title="LymphFit: Advanced AI for Personalized Lymphatic Exercise Guidance", 
                   layout="wide", initial_sidebar_state='expanded')

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"



# Load the selected page dynamically
if st.session_state.page == "Home":
    st.title("LymphFit: AI-Powered Lymphatic Exercise Guidance")
    st.subheader("Enhancing Recovery Through Intelligent Insights")
    st.write("LymphFit is an advanced AI-driven platform designed to assist breast cancer patients and individuals in their lymphatic recovery journey. Our solution integrates real-time data from EMG sensors, smartwatches, and video analysis to provide personalized exercise guidance and posture correction.")
    
    st.write("### Explore LymphFit")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Go to LymphFit Pose Comparison"):
            st.switch_page("pages/Pose_Comparison.py")
    with col2:
        if st.button("Go to LymphFit Data Visualization"):
            st.switch_page("pages/Data_Visualization.py")
    # with col3:
    #     if st.button("Go to Real-Time Visualization"):
    #         st.switch_page("pages/Real-Time_Visualization.py")
