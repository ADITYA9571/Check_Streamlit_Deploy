import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

st.title("ML Model Comparison App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    dataset = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(dataset.head())

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    model_type = st.selectbox(
        "Choose Model Type",
        ["Regression", "Classification"]
    )

    if st.button("Run Models"):

        if model_type == "Regression":

            results = {}
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Linear Regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results["Linear Regression"] = r2_score(y_test, y_pred)

            # Decision Tree
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results["Decision Tree"] = r2_score(y_test, y_pred)

            # Random Forest
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results["Random Forest"] = r2_score(y_test, y_pred)

            st.write("### Regression Results")
            st.write(results)
            st.success(f"Best Model: {max(results, key=results.get)}")

        else:

            results = {}
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Logistic Regression
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results["Logistic Regression"] = accuracy_score(y_test, y_pred)

            # SVM
            from sklearn.svm import SVC
            model = SVC(kernel="rbf")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results["SVM"] = accuracy_score(y_test, y_pred)

            # Random Forest
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results["Random Forest"] = accuracy_score(y_test, y_pred)

            st.write("### Classification Results")
            st.write(results)
            st.success(f"Best Model: {max(results, key=results.get)}")