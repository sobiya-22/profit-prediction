# 📦 Install needed
# pip install streamlit pandas scikit-learn matplotlib openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="Profit Predictor", page_icon="📈")
# Load data
@st.cache_data
def load_data():
    master_stock = pd.read_excel('TransconCoData.xlsx', sheet_name='MasterStockSheet')
    customer_orders = pd.read_excel('TransconCoData.xlsx', sheet_name='CustomerOrdersSheet')
    return master_stock, customer_orders

master_stock, customer_orders = load_data()

# Preprocess month column
def convert_month_to_number(month_str):
    return pd.to_datetime('01-' + month_str, format='%d-%b-%y')

customer_orders['month_date'] = customer_orders['month'].apply(convert_month_to_number)
master_stock['month_date'] = master_stock['month'].apply(convert_month_to_number)

# Aggregate monthly profit
monthly_profit = customer_orders.groupby(customer_orders['month_date'].dt.to_period('M'))['profit_loss'].sum().reset_index()
monthly_profit['month_date'] = monthly_profit['month_date'].dt.to_timestamp()

# Prepare model
X = np.arange(len(monthly_profit)).reshape(-1, 1)
y = monthly_profit['profit_loss'].values
model = LinearRegression()
model.fit(X, y)

# Functions
def predict_profit_for_future_month(months_ahead):
    month_index = len(monthly_profit) + months_ahead - 1
    predicted_profit = model.predict(np.array([[month_index]]))[0]
    return predicted_profit

def suggest_quantity_for_target_profit(target_profit_percent, months_ahead):
    latest_month = master_stock.sort_values('month_date', ascending=False).iloc[0]
    purchase_price = latest_month['purchase_price_per_kg']
    sale_price = latest_month['sale_price_per_kg']
    margin_per_kg = sale_price - purchase_price
    profit_percent_per_kg = (margin_per_kg / purchase_price) * 100

    if profit_percent_per_kg <= 0:
        return "⚠️ Current margin is negative or zero! Cannot suggest."

    predicted_profit = predict_profit_for_future_month(months_ahead)
    required_profit = predicted_profit * (target_profit_percent / profit_percent_per_kg)
    quantity_required = required_profit / margin_per_kg

    return {
        "Predicted Profit (₹)": round(predicted_profit, 2),
        "Current Profit % per KG": round(profit_percent_per_kg, 2),
        "Required Additional Quantity (KG)": round(quantity_required, 2),
        "Product to Focus": latest_month['product_type']
    }

# Streamlit UI

st.title("📊 Profit and Sales Predictor App")

st.sidebar.header("🛠️ Controls")
months_ahead = st.sidebar.slider('Months Ahead to Predict', 1, 12, 1)
target_profit_percent = st.sidebar.number_input('Target Profit % (optional)', min_value=0.0, step=1.0)

if st.sidebar.button('Predict'):
    st.subheader(f"🔮 Prediction for {months_ahead} Month(s) Ahead")
    predicted_profit = predict_profit_for_future_month(months_ahead)
    st.success(f"Predicted Profit: ₹{predicted_profit:.2f}")

    if target_profit_percent > 0:
        suggestion = suggest_quantity_for_target_profit(target_profit_percent, months_ahead)
        st.subheader("💡 Suggestion to Achieve Target Profit")
        st.json(suggestion)
    else:
        st.info("No target profit % provided. Showing only prediction.")

    # Show Profit Trend Graph
    fig, ax = plt.subplots()
    ax.plot(monthly_profit['month_date'], monthly_profit['profit_loss'], marker='o')
    ax.set_title('Monthly Profit Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Profit (₹)')
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info('Set the parameters and click Predict to start!')

st.caption("Built with ❤️ using Streamlit")
