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

    # Remove Id column if present
    if "Id" in dataset.columns:
        dataset = dataset.drop("Id", axis=1)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    model_type = st.selectbox(
        "Choose Model Type",
        ["Regression", "Classification"]
    )

    if st.button("Run Models"):
        
        # =========================
        # REGRESSION
        # =========================
        if model_type == "Regression":

            results = {}

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=0
            )

            # Multiple Linear Regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            results["Multiple Linear Regression"] = r2_score(
                y_test, model.predict(X_test)
            )

            # Polynomial Regression
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=4)
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)

            model = LinearRegression()
            model.fit(X_poly_train, y_train)
            results["Polynomial Regression"] = r2_score(
                y_test, model.predict(X_poly_test)
            )

            # Support Vector Regression
            from sklearn.svm import SVR

            sc_X = StandardScaler()
            sc_y = StandardScaler()

            X_train_scaled = sc_X.fit_transform(X_train)
            y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1))

            model = SVR(kernel='rbf')
            model.fit(X_train_scaled, y_train_scaled.ravel())

            y_pred_scaled = model.predict(sc_X.transform(X_test))
            y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1,1))

            results["Support Vector Regression"] = r2_score(
                y_test, y_pred
            )

            # Decision Tree Regression
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(random_state=0)
            model.fit(X_train, y_train)
            results["Decision Tree Regression"] = r2_score(
                y_test, model.predict(X_test)
            )

            # Random Forest Regression
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=10,
                random_state=0
            )
            model.fit(X_train, y_train)
            results["Random Forest Regression"] = r2_score(
                y_test, model.predict(X_test)
            )

            st.write("### Regression Results")
            st.write(results)
            st.success(f"Preferred Regression Model:- {max(results, key=results.get)}")

            
        # =========================
        # CLASSIFICATION
        # =========================
        else:

            results = {}

            # SPLIT ONLY ONCE (VERY IMPORTANT)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.25,
                random_state=0
            )

            # SCALE ONLY ONCE
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # Logistic Regression
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=0)
            model.fit(X_train, y_train)
            results["Logistic Regression"] = accuracy_score(
                y_test, model.predict(X_test)
            )

            # KNN
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train, y_train)
            results["KNN"] = accuracy_score(
                y_test, model.predict(X_test)
            )

            # Linear SVM
            from sklearn.svm import SVC
            model = SVC(kernel="linear", random_state=0)
            model.fit(X_train, y_train)
            results["SVM"] = accuracy_score(
                y_test, model.predict(X_test)
            )

            # Kernel SVM
            model = SVC(kernel="rbf", random_state=0)
            model.fit(X_train, y_train)
            results["Kernal SVM"] = accuracy_score(
                y_test, model.predict(X_test)
            )

            # Naive Bayes
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            model.fit(X_train, y_train)
            results["Naive Bayes"] = accuracy_score(
                y_test, model.predict(X_test)
            )

            # Decision Tree
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(
                criterion="entropy",
                random_state=0
            )
            model.fit(X_train, y_train)
            results["Decision Tree Classification"] = accuracy_score(
                y_test, model.predict(X_test)
            )

            # Random Forest
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=10,
                criterion="entropy",
                random_state=0
            )
            model.fit(X_train, y_train)
            results["Random Tree Classification"] = accuracy_score(
                y_test, model.predict(X_test)
            )

            st.write("### Classification Results")
            st.write(results)
            st.success(f"Preferred Classification Model:- {max(results, key=results.get)}")
