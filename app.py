# In app.py
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("🚀 NASA Bioscience Explorer")

# Load data (use placeholder until integration)
# df = pd.read_csv('publications.csv')

# --- Sidebar ---
st.sidebar.header("Search & Filter")
search_query = st.sidebar.text_input("Search by concept (e.g., 'muscle atrophy in microgravity')")
search_button = st.sidebar.button("Search")
st.sidebar.markdown("---")
st.sidebar.header("Publication Viewer")
# selected_id = st.sidebar.selectbox("Select a Publication", df['title'])

# --- Main Page ---
st.header("Search Results")
st.write("Results will appear here...")

# Placeholder for selected paper details
st.header("Paper Details")
st.write("Select a paper from the sidebar to see details and a summary.")