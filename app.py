import streamlit as st
import pandas as pd

from src.predict import predict_employee


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Employee Attrition Prediction")
st.write("Enter employee information to predict whether the employee is likely to leave.")


# -----------------------------
# Employee Information
# -----------------------------

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    salary = st.number_input(
        "Monthly Salary",
        min_value=0.0,
        value=30000.0
    )

    years = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=50,
        value=5
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4, 5]
    )

    # performance_rating = st.selectbox(
    #     "Performance Rating",
    #     [1, 2, 3, 4, 5]
    # )

    # work_life_balance = st.selectbox(
    #     "Work Life Balance",
    #     [1, 2, 3, 4, 5]
    # )

    # training_hours = st.number_input(
    #     "Training Hours",
    #     min_value=0,
    #     value=20
    # )


with col2:

    # gender = st.selectbox(
    #     "Gender",
    #     ["MALE", "FEMALE"]
    # )

    # department = st.selectbox(
    #     "Department",
    #     ["FINANCE", "HR", "IT", "MARKETING", "SALES"]
    # )

    # job_level = st.selectbox(
    #     "Job Level",
    #     ["Junior", "Mid", "Senior", "Manager"]
    # )

    overtime = st.selectbox(
        "Overtime",
        ["Yes", "No"]
    )

    # remote_work = st.selectbox(
    #     "Remote Work",
    #     ["Yes", "No"]
    # )

    # city = st.selectbox(
    #     "City",
    #     [
    #         "Ahmedabad",
    #         "Delhi",
    #         "Mumbai",
    #         "Rajkot",
    #         "Surat",
    #         "Vadodara"
    #     ]
    # )


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Attrition"):

    employee = pd.DataFrame([{
        "Age": age,
        "Monthly_Salary": salary,
        "Years_At_Company": years,
        "Job_Satisfaction": job_satisfaction,
        # "Performance_Rating": performance_rating,
        # "Work_Life_Balance": work_life_balance,
        # "Training_Hours": training_hours,
        # "Gender": gender,
        # "Department": department,
        # "Job_Level": job_level,
        "Overtime": overtime,
        # "Remote_Work": remote_work,
        # "City": city
    }])

    result, probability = predict_employee(employee)

    st.divider()

    if "leave" in result.lower():

        st.error("⚠️ Employee is likely to leave")

    else:

        st.success("✅ Employee is likely to stay")

    st.metric(
        "Probability of Leaving",
        f"{probability:.2%}"
    )