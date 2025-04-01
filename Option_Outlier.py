# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:19:22 2025

@author: Hemal
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import matplotlib.pyplot as plt
from datetime import datetime

# Function to calculate implied volatility using the Newton-Raphson method
def calculate_implied_volatility(row, underlying_price, risk_free_rate):
    S = underlying_price
    K = row['Strike Price']
    T = row['Time to Expiry']
    r = risk_free_rate
    option_price = row['Close']
    option_type = row['Option type']

    def f(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'CE':
            model_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'PE':
            model_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return model_price - option_price

    initial_guess = 0.2
    try:
        implied_volatility = newton(f, initial_guess)
    except RuntimeError:
        implied_volatility = np.nan

    return implied_volatility

# Function to calculate Greeks and IV
def calculate_greeks_and_iv(row, underlying_price, risk_free_rate):
    S = underlying_price
    K = row['Strike Price']
    T = row['Time to Expiry']
    r = risk_free_rate
    sigma = row['Implied Volatility']
    option_type = row['Option type']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'CE':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == 'PE':
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    iv = sigma

    return pd.Series([delta, gamma, vega, theta, rho, iv], index=['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'IV'])

# Streamlit app
st.title("Options Greeks and Implied Volatility Calculator")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file
    data = pd.read_csv(uploaded_file)

    # Strip extra spaces from column names
    data.columns = data.columns.str.strip()

    # Convert 'Date' and 'Expiry' columns to datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Expiry'] = pd.to_datetime(data['Expiry'], errors='coerce')

    # Check for any parsing errors
    if data['Date'].isnull().any() or data['Expiry'].isnull().any():
        st.error("Error parsing dates. Please ensure the date format is correct.")
    else:
        # Calculate time to expiration in years
        data['Time to Expiry'] = (data['Expiry'] - data['Date']).dt.days / 365.0

        # Input for risk-free rate
        risk_free_rate = st.number_input("Risk-free rate", value=0.07, format="%.2f")

        # Calculate implied volatility and Greeks
        underlying_price = data['Underlying Value'].iloc[0]
        data['Implied Volatility'] = data.apply(calculate_implied_volatility, axis=1, underlying_price=underlying_price, risk_free_rate=risk_free_rate)
        data[['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'IV']] = data.apply(calculate_greeks_and_iv, axis=1, underlying_price=underlying_price, risk_free_rate=risk_free_rate)

        # Check for significant changes in implied volatility
        data['Previous IV'] = data.groupby('Strike Price')['IV'].shift(1)
        data['IV Change'] = (data['IV'] - data['Previous IV']) / data['Previous IV']
        data['Significant IV Change'] = data['IV Change'].apply(lambda x: 'Yes' if abs(x) >= 0.05 else 'No')

        significant_iv_changes = data[data['Significant IV Change'] == 'Yes']

        # Display the table of significant IV changes
        st.write("Significant IV Changes:")
        st.write(significant_iv_changes[['Date', 'Strike Price', 'Option type', 'IV', 'Previous IV', 'IV Change']])

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['IV'], label='Implied Volatility', alpha=0.7)
        plt.scatter(significant_iv_changes['Date'], significant_iv_changes['IV'], color='orange', label='Significant IV Change', zorder=5)
        plt.xlabel('Date')
        plt.ylabel('Implied Volatility')
        plt.title('Implied Volatility with 1-5% Changes Highlighted')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Save the DataFrame to a CSV file
        output_file_path = 'options_data_with_greeks.csv'
        data.to_csv(output_file_path, index=False)
        st.success(f"DataFrame saved to {output_file_path}")

        # Provide a download link for the CSV file
        with open(output_file_path, "rb") as file:
            btn = st.download_button(
                label="Download CSV",
                data=file,
                file_name="options_data_with_greeks.csv",
                mime="text/csv"
            )
