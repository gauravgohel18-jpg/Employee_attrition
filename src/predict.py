import sys
import joblib
import pandas as pd

from src.logger import logging
from src.exception import CustomException


MODEL_PATH = "models/employee_attrition_model.pkl"


def predict_employee(employee_data):

    try:
        logging.info("Starting prediction")

        # Load trained model
        model = joblib.load(MODEL_PATH)

        # Make prediction
        prediction = model.predict(employee_data)[0]

        # Probability of leaving
        probability = model.predict_proba(employee_data)[0][1]

        if prediction == 1:
            result = "Employee is likely to leave"
        else:
            result = "Employee is likely to stay"

        logging.info("Prediction completed successfully")

        return result, probability

    except Exception as e:

        logging.error("Error occurred during prediction")

        raise CustomException(e, sys)


if __name__ == "__main__":

    employee = pd.DataFrame([{
        "Age": 25,
        "Monthly_Salary": 35000,
        "Years_At_Company": 2,
        "Job_Satisfaction": 2,
        "Performance_Rating": 3,
        "Work_Life_Balance": 2,
        "Training_Hours": 15,
        "Gender": "MALE",
        "Department": "IT",
        "Job_Level": "Junior",
        "Overtime": "Yes",
        "Remote_Work": "No",
        "City": "Surat"
    }])

    result, probability = predict_employee(employee)

    print("Prediction:", result)
    print(f"Probability of leaving: {probability:.2%}")