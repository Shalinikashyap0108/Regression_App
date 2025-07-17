import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels as sm

st.set_page_config(page_title= "Regression Model App", layout = "wide")

st.title("Regression App")

# upload the dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upoad your file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Ames_Housing_Subset.csv")


st.write("### Dataset Preview", df.head())

# feature selection

target = st.sidebar.selectbox("Select target Variable:", df.columns)
features = st.sidebar.multiselect("Select Feature Columns", [col for col in df.columns if col != target])


if not features:
    st.warning("Please select atleast one feature.")
    st.stop()

X = df[features]
y = df[target]

# train/test split
test_size_n = st.sidebar.slider("Test Size (%)", 10, 20, 30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_n/100, random_state=42)


model_name = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Decision Tree", "Random Forest"])

# hyperparameters

if model_name == "Decision Tree":
    params["max depth"] = st.sidebar.slider("Max Depth", 1,5,10,20)
elif model_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("Number of Trees", 10, 50, 100)
    params["max depth"] = st.sidebar.slider("Max Depth", 1,5,10,20)


if st.sidebar.button("Train Model"):
    model = train_model(model_name, X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Evaluation")
    st.write("Mean Absolute Error", mean_absolute_error(y_test, y_pred))
    st.write("Mean Squared Error", mean_squared_error(y_test, y_pred))
    st.write("R-square score", r2_score(y_test, y_pred))

    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importasnce")
        importance = pd.Series(model.feature_importances_, index = features).sort_values(ascending = True)
        st.bar_chart(importance)

    if model == "Linear Regression":
        st.subheader("Summary Statistics")
        X_train_const = sm.add_constant(X_train)
        ols = sm.OLS(y_train, X_train_const).fit()
        st.text(ols.summary)

        st.subheader("Coefficients")
        coef = pd.Series(model.coef_, index=features)
        st.write(coef)
