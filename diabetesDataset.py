import joblib
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.write("### Regression Linear On Diabetes Dataset")
dataset = datasets.load_diabetes()
st.write("#### Dataset Features: \n" + str(dataset.feature_names))

x = np.array(dataset.data[:, np.newaxis, 9])

X_train, X_test, y_train, y_test = train_test_split(x, dataset.target, test_size=0.4, random_state=300)

# X_train = np.array(x[:-20])
# y_train = np.array(dataset.target[:-20])
#
# X_test = np.array(x[-20:])
# y_test = np.array(dataset.target[-20:])


model = linear_model.LinearRegression()
model.fit(X_train, y_train)

yPredict = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='r')
ax.plot(X_test, yPredict, color='b', linewidth=2)
st.pyplot(fig)

# Dump diabetes dataset linear regression for use
joblib.dump(model, "diabetes.sav")

st.write("#### Test Diabetes Linear Regression Result: ")

age = st.number_input("Please enter age")
# sex = st.number_input("Please enter sex")
# bmi = st.number_input("Please enter bmi")
# bp = st.number_input("Please enter bp")
age = age * 0.001
# sex = sex * 0.001
# bmi = bmi * 0.001
# bp = bp * 0.001

if st.button("Predict"):
    load_module = joblib.load("diabetes.sav")
    diabetes_value = load_module.predict([[age]])
    st.write("Result is : ", str(diabetes_value))
