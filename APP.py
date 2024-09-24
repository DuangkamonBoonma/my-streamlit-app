import joblib
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Condo Price Predictor",
    page_icon=":bar_chart:",
    layout="wide"
)

# Header
st.title("Condo Price Prediction")
st.write("Predict the home sale price based on location and features.")

# Sidebar
st.sidebar.header("User Input")
st.sidebar.text("Adjust the settings below to see how the model predictions change.")

# Input Controls in Sidebar
title = st.sidebar.text_input("Title", "Enter the title here")

# District selection
districts = [
    "ดอนเมือง",
    "พระโขนง",
    "ทุ่งครุ",
    "สวนหลวง",
    "ลาดกระบัง",
    "จตุจักร",
    "ภาษีเจริญ"
]
district_selection = st.sidebar.selectbox("Select District", districts)

detail = st.sidebar.slider("Details Importance (0-100)", min_value=0, max_value=100, value=50)
price = st.sidebar.slider("Price (บาท)", min_value=0, max_value=10000000, value=5000000)

# Creating a DataFrame for the input features
input_data = pd.DataFrame({
    'Title': [title],
    'Details_encoder': [detail],
    'Price (บาท)': [price],
})

# Encode District
label_encoder = LabelEncoder()
label_encoder.fit(districts)  # Fit the encoder on the districts
district_encoded = label_encoder.transform([district_selection])  # Transform the selected district
input_data['District'] = district_encoded  # Add encoded district to input_data

# Ensure the correct column order for the model
input_data = input_data[['Title', 'Details_encoder', 'District', 'Price (บาท)']]  # Ensure this matches the training data

# Prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result")
    st.write(f"The predicted value is: {prediction[0]:.2f}")

# Display a plot for better visualization
st.subheader("Feature Distribution")

# Generating sample data for plotting
data = {
    'Price (บาท)': np.random.randn(100) * 1000000,
    'Details_encoder': np.random.randn(100) * 100
}
df = pd.DataFrame(data)

# Scatter Plot
st.subheader("Scatter Plot of Features")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Price (บาท)', y='Details_encoder', ax=ax, color='blue', alpha=0.6)
ax.set_title('Scatter Plot of Price vs Details Importance')
ax.set_xlabel('Price (บาท)')
ax.set_ylabel('Details Importance')
st.pyplot(fig)


# Additional Information
st.markdown("This app predicts condo prices based on user input.")
