import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sqlite3
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np



# Function to load data from SQLite
def load_data():
    conn = sqlite3.connect('expenses.db')
    df = pd.read_sql_query("SELECT * FROM monthly_expenses", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Function to get total spending by category
def total_spending_by_category(df):
    return df.groupby('Categories')['Amount'].sum().sort_values(ascending=False)

# Function to get spending by payment mode
def spending_by_payment_mode(df):
    return df.groupby('Payment_mode')['Amount'].sum().sort_values(ascending=False)

# Function to get monthly spending trends
def monthly_spending_trends(df):
    df['Month'] = df['Date'].dt.to_period('M')
    return df.groupby('Month')['Amount'].sum()

# Function to calculate average spending per category
def avg_spending_by_category(df):
    return df.groupby('Categories')['Amount'].mean().sort_values(ascending=False)

# Function for linear regression prediction on monthly spending
def predict_monthly_spending(df):
    df['Month'] = df['Date'].dt.month
    df_grouped = df.groupby('Month')['Amount'].sum().reset_index()

    X = df_grouped['Month'].values.reshape(-1, 1)  # Month number as feature
    y = df_grouped['Amount'].values  # Total amount spent each month

    model = LinearRegression()
    model.fit(X, y)

    # Predict spending for the next month (month 13)
    next_month = np.array([[13]])
    predicted_spending = model.predict(next_month)[0]

    return predicted_spending

# Title of the Streamlit app
st.title("Expense Tracker Dashboard")

# Load data
df = load_data()

# Sidebar for user input
st.sidebar.title("Filters")
category_filter = st.sidebar.selectbox("Select Category", df['Categories'].unique())
payment_mode_filter = st.sidebar.selectbox("Select Payment Mode", df['Payment_mode'].unique())

# Date Range Filter
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

# Display raw data (optional)
st.subheader("Raw Data")
filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
st.write(filtered_df)

# 1. Distribution of Spending Across Categories
st.subheader("Total Spending by Category")
category_spending = total_spending_by_category(filtered_df)
st.bar_chart(category_spending)

# 2. Total Spending by Payment Mode
st.subheader("Total Spending by Payment Mode")
payment_mode_spending = spending_by_payment_mode(filtered_df)
st.bar_chart(payment_mode_spending)

# 3. Monthly Spending Trends
st.subheader("Monthly Spending Trends")
monthly_spending = monthly_spending_trends(filtered_df)
st.line_chart(monthly_spending)

# 4. Spending Filtered by Category (Dynamic Query)
st.subheader(f"Spending in Category: {category_filter}")
category_df = filtered_df[filtered_df['Categories'] == category_filter]
category_spending_filtered = category_df.groupby('Date')['Amount'].sum()
st.line_chart(category_spending_filtered)

# 5. Spending Filtered by Payment Mode (Dynamic Query)
st.subheader(f"Spending by Payment Mode: {payment_mode_filter}")
payment_mode_df = filtered_df[filtered_df['Payment_mode'] == payment_mode_filter]
payment_mode_spending_filtered = payment_mode_df.groupby('Date')['Amount'].sum()
st.line_chart(payment_mode_spending_filtered)

# 6. Interactive Pie Chart for Spending Distribution by Category
st.subheader("Spending Distribution by Category (Pie Chart)")
fig = px.pie(filtered_df, names='Categories', values='Amount', title='Spending Distribution by Category')
st.plotly_chart(fig)

# 7. Interactive Bar Chart for Total Spending by Payment Mode
st.subheader("Total Spending by Payment Mode (Interactive Bar Chart)")
fig = px.bar(payment_mode_spending, x=payment_mode_spending.index, y=payment_mode_spending.values, 
             labels={'x': 'Payment Mode', 'y': 'Total Spending'}, title='Total Spending by Payment Mode')
st.plotly_chart(fig)

# 8. Average Spending per Category
st.subheader("Average Spending by Category")
avg_spending = avg_spending_by_category(filtered_df)
st.bar_chart(avg_spending)

# 9. Predicting Future Spending (Next Month)
predicted_spending = predict_monthly_spending(filtered_df)
st.subheader("Predicted Spending for Next Month")
st.write(f"Predicted spending for next month: ${predicted_spending:.2f}")

# 10. Custom SQL Query
st.subheader("Custom SQL Query")
sql_query = st.text_area("Enter your SQL query below:", "SELECT * FROM expenses LIMIT 10")
if st.button('Run Query'):
    conn = sqlite3.connect('expenses.db')
    try:
        custom_query_result = pd.read_sql_query(sql_query, conn)
        st.write(custom_query_result)
    except Exception as e:
        st.error(f"Error executing query: {e}")
    finally:
        conn.close()