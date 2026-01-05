import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Hours vs Marks - Linear Regression")

st.title("📘 Linear Regression: Hours Studied vs Marks")

# -----------------------------
# Input Data (from your source)
# -----------------------------
hours = np.array([[1], [2], [3], [4], [5]])
marks = np.array([35, 40, 45, 50, 55])

st.subheader("📊 Training Data")
st.write("Hours Studied:", hours.flatten())
st.write("Marks:", marks)

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(hours, marks)

# -----------------------------
# Prediction Input
# -----------------------------
st.subheader("🔮 Predict Marks")

study_hours = st.slider("Select study hours", 1, 10, 6)
prediction = model.predict([[study_hours]])

st.success(f"📌 Predicted marks for {study_hours} hours: **{prediction[0]:.2f}**")

# -----------------------------
# Plot
# -----------------------------
st.subheader("📈 Regression Graph")

predicted_train = model.predict(hours)

fig, ax = plt.subplots()
ax.scatter(hours, marks, label="Actual Data")
ax.plot(hours, predicted_train, label="Regression Line")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")
ax.set_title("Linear Regression - Hours vs Marks")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Model Info
# -----------------------------
st.subheader("📐 Model Equation")
st.write(f"Marks = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}")
