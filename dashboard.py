import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("results_loadshift.csv")

st.title("Data Center Sustainability Dashboard")

st.header("Results")
st.metric("Total Energy (MWh)", round(df["grid_import_mwh"].sum(), 2))
st.metric("Total Cost (€)", round(df["cost_eur"].sum(), 2))

fig1 = px.line(df, x="timestamp", y="grid_import_mwh", title="Energy Consumption")
st.plotly_chart(fig1)

fig2 = px.line(df, x="timestamp", y="soc", title="Battery SOC")
st.plotly_chart(fig2)

fig3 = px.line(df, x="timestamp", y="price_eur_per_mwh", title="Price (€/MWh)")
st.plotly_chart(fig3)

fig4 = px.line(df, x="timestamp", y=["critical_load_mw","flexible_load_mw","exec_flex_mw"],
               title="Workload Distribution")
st.plotly_chart(fig4)