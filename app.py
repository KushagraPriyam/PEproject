import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv("customer_dataset.csv")
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill missing values
    return df

# Feature Engineering
def create_features(df):
    df_features = df.copy()
    df_features['Total_Expenditure'] = df_features['Q1 Expenditure'] + df_features['Q2 Expenditure'] + df_features['Q3 Expenditure'] + df_features['Q4 Expenditure']
    df_features['Expenditure_Variability'] = df_features[['Q1 Expenditure', 'Q2 Expenditure', 'Q3 Expenditure', 'Q4 Expenditure']].std(axis=1)
    df_features['Discount Used'] = df_features['Discount Used'].map({'Yes': 1, 'No': 0})
    return df_features

# Train Model
def train_model(df_features, target='Q4 Expenditure'):
    y = df_features[target]
    numeric_features = ['Age', 'Q1 Expenditure', 'Q2 Expenditure', 'Q3 Expenditure']
    categorical_features = ['Store Type', 'City', 'Gender', 'Age Group', 'Discount Used']

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_categorical = encoder.fit_transform(df_features[categorical_features])
    X = np.hstack((df_features[numeric_features].values, X_categorical))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and encoder
    joblib.dump(model, 'sales_model.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    return model, encoder

# Load Model
def load_model():
    model = joblib.load('sales_model.pkl')
    encoder = joblib.load('encoder.pkl')
    return model, encoder

#**********************************************************************************************************************************************
# Streamlit App
st.title("🛒 Sales Prediction System")
st.write("Predict Q4 expenditure based on previous purchase data.")

# Load Data
file_path = "customer_dataset_modified.csv"
df = load_data(file_path)
df_features = create_features(df)

# Train Model (Only Once)
if "sales_model.pkl" not in st.session_state:
    model, encoder = train_model(df_features)
else:
    model, encoder = load_model()

# User Inputs
st.sidebar.header("Enter Customer Details")
store_type = st.sidebar.selectbox("Store Type", df['Store Type'].unique())
city = st.sidebar.selectbox("City", df['City'].unique())
gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
age_group = st.sidebar.selectbox("Age Group", df['Age Group'].unique())
discount_used = st.sidebar.radio("Discount Used?", ['Yes', 'No'])
age = st.sidebar.slider("Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].mean()))
q1_exp = st.sidebar.number_input("Q1 Expenditure", min_value=0, value=int(df['Q1 Expenditure'].mean()))
q2_exp = st.sidebar.number_input("Q2 Expenditure", min_value=0, value=int(df['Q2 Expenditure'].mean()))
q3_exp = st.sidebar.number_input("Q3 Expenditure", min_value=0, value=int(df['Q3 Expenditure'].mean()))

# Convert inputs into dataframe
input_data = pd.DataFrame([[store_type, city, gender, age_group, discount_used, age, q1_exp, q2_exp, q3_exp]],
                          columns=['Store Type', 'City', 'Gender', 'Age Group', 'Discount Used', 'Age', 'Q1 Expenditure', 'Q2 Expenditure', 'Q3 Expenditure'])

# Encode categorical variables
input_data['Discount Used'] = input_data['Discount Used'].map({'Yes': 1, 'No': 0})
input_encoded = encoder.transform(input_data[['Store Type', 'City', 'Gender', 'Age Group', 'Discount Used']])
input_final = np.hstack((input_data[['Age', 'Q1 Expenditure', 'Q2 Expenditure', 'Q3 Expenditure']].values, input_encoded))

# Make Prediction
predicted_q4 = model.predict(input_final)[0]
st.sidebar.subheader(f"Predicted Q4 Expenditure: ₹{predicted_q4:.2f}")

# Display Data Preview
st.subheader("Customer Purchase Data")
st.write(df.head())

# Visualizations
st.subheader("Expenditure Trends")

# Boxplot for Q1-Q4 expenditures
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df[['Q1 Expenditure', 'Q2 Expenditure', 'Q3 Expenditure', 'Q4 Expenditure']], palette="Set2")
ax.set_title("Quarterly Expenditure Distribution")
st.pyplot(fig)

# Scatter plot of Q3 vs Q4 Expenditure
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=df['Q3 Expenditure'], y=df['Q4 Expenditure'], alpha=0.6)
ax.set_xlabel("Q3 Expenditure")
ax.set_ylabel("Q4 Expenditure")
ax.set_title("Q3 vs Q4 Expenditure")
st.pyplot(fig)

# Footer
st.write("Developed with ❤️ by Kushagra Priyam")

