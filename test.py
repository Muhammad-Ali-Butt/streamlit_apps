import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.header("This dashboard is for testing purposes only")
st.text("This is a test")

st.header('hello this is only a test')

df = sns.load_dataset('iris')
st.write(df[['species', 'sepal_length', 'petal_length']])

st.bar_chart(df['petal_length'])
st.line_chart(df['petal_length'])
