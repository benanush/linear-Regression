import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Hours vs Marks - Linear Regression")

st.title("📘 Linear Regression: Hours Studied vs Marks")

# -----------------------------
# Input Data
# -----------------------------
hours = np.array([[1], [2], [3], [4], [5]])
marks = np.array([35, 40, 45, 50, 55])

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(hours, marks)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("🔮 Predict Marks")
study_hours = st.slider("Select study hours", 1, 12, 6)
prediction = model.predict([[study_hours]])

st.success(f"📌 Predicted marks for {study_hours} hours: **{prediction[0]:.2f}**")

# -----------------------------
# Plot
# -----------------------------
st.subheader("📈 Regression Graph")

fig, ax = plt.subplots()

ax.scatter(hours, marks, label="Actual Data")
ax.plot(hours, model.predict(hours), label="Regression Line")
ax.scatter(study_hours, prediction, s=100, label="Your Prediction")

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Model Equation
# -----------------------------
st.subheader("📐 Model Equation")
st.code(f"Marks = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}")
