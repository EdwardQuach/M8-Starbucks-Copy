import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/cleaned_starbucks.csv')

page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis"])
if page == "Home":
    st.title("☕Starbucks Dataset Explorer")
    st.subheader("Welcome to my Starbucks Dataset app!")
    st.write("""This app provides an ainteractive platform to explore the Starbucks dataset. 
    You can visualize the distribution of data, explore relationships between features, and even make predictions on new data! 
    Use the sidebar to navigate through sections.""")

elif page == "Data Overview":
    st.title("Data Overview")
    st.write("Here’s where my data visualizations and dashboards will go.")
    st.dataframe(df)

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA) ")
    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])
    if 'Histograms' in eda_type:
        st.subheader("Histograms - Beverages")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace("_", ' ')}"
            if st.checkbox("Show by Beverage"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color ='Beverage', title = chart_title, barmode = 'overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Box Plots")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='Calories', y=b_selected_col, title=chart_title, color='Calories'))
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_',' ')}vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='Beverage', title = chart_title))
    if 'Count Plots' in eda_type:
        st.subheader("Count Plots -")
        obj_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_col = st.selectbox("Select a categorical variable:", obj_cols)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, color='Beverage', title = chart_title)) 