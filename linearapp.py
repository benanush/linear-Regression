import streamlit as st
import numpy as np
import matplotlib.pyplot as plt  # FIXED: Added .pyplot
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
# Sidebar/Input Logic
# -----------------------------
st.subheader("🔮 Predict Marks")
study_hours = st.slider("Select study hours", 1, 12, 6)
prediction = model.predict([[study_hours]])

st.success(f"📌 Predicted marks for {study_hours} hours: **{prediction[0]:.2f}**")

# -----------------------------
# Plotting Logic
# -----------------------------
st.subheader("📈 Regression Graph")

# Create the figure
fig, ax = plt.subplots()

# 1. Plot the training points
ax.scatter(hours, marks, color='blue', label="Actual Data")

# 2. Plot the regression line
# We create a range of hours to draw a smooth line
line_hours = np.array([[1], [12]])
line_marks = model.predict(line_hours)
ax.plot(line_hours, line_marks, color='red', label="Regression Line")

# 3. Plot the specific prediction point
ax.scatter([[study_hours]], prediction, color='green', s=100, zorder=5, label="Your Prediction")

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Display the plot in Streamlit
st.pyplot(fig)

# -----------------------------
# Model Info
# -----------------------------
st.subheader("📐 Model Equation")
st.code(f"Marks = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}")