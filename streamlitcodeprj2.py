import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("🎓 Career Level Prediction")

# ---------------- LOAD DATA ----------------
data = pd.read_csv(r"c:\Users\Mery\Downloads\education_career_success.csv")

# ---------------- FEATURES ----------------
model_features = [
    "High_School_GPA",
    "SAT_Score",
    "University_GPA",
    "Internships_Completed",
    "Projects_Completed",
    "Certifications",
    "Soft_Skills_Score",
    "Networking_Score",
    "Job_Offers",
    "Starting_Salary",
    "Career_Satisfaction",
    "Years_to_Promotion",
    "Work_Life_Balance"
]

target = "Current_Job_Level"

# ---------------- ENCODING ----------------
le = LabelEncoder()
data[target] = le.fit_transform(data[target])

X = data[model_features]
y = data[target]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- USER INPUT ----------------
st.subheader("Enter Student Details")

student_id = st.text_input("Student_ID")

user_inputs = []
for feature in model_features:
    user_inputs.append(st.text_input(feature))

# ---------------- PREDICTION ----------------
if st.button("Predict Job Level"):
    if "" in user_inputs:
        st.warning("⚠ Please fill all fields")
    else:
        try:
            user_inputs = np.array(user_inputs, dtype=float).reshape(1, -1)
            user_inputs_scaled = scaler.transform(user_inputs)

            prediction = model.predict(user_inputs_scaled)
            job_level = le.inverse_transform(prediction)

            st.success(f"""
            ✅ **Prediction Successful**
            - Student ID: **{student_id}**
            - Current Job Level: **{job_level[0]}**
            """)

        except ValueError:
            st.error("❌ Please enter only numeric values (except Student_ID)")