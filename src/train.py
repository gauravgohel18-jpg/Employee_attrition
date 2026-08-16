import os
import sys
from src.logger import logging
from src.exception import CustomException
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from src.preprocessing import create_preprocessor,evalution_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



def train_model():

    try:

        # load Cleaned data
        logging.info('Started Model Training')

        df = pd.read_csv('data/cleaned_employee_data.csv')

        logging.info('Successfully Read Cleaned data')

        # separate Features 

        selected_features = [
        "Age",
        "Monthly_Salary",
        "Years_At_Company",
        "Job_Satisfaction",
        "Overtime"
         ]

        X = df[selected_features]
        y = df["Left_Company"]

        # Train_test_split the Features

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0,stratify=y)

        logging.info('Train_Test_split Completed')

        # create Preprocessor

        preprocessor = create_preprocessor()

        # Create Random Forest pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(random_state=42))
                ]
            )

        logging.info("Logistic Regression pipeline Successfully Created!")

        # train and evalute that model
        Baseline_model = pipeline.fit(X_train,y_train)

        Baseline_ev_score = evalution_model(Baseline_model,X_test,y_test,'Baseline Logistic Regression')

        
        Baseline_cv_score = cross_val_score(
            Baseline_model,
            X_train,
            y_train,
            cv=5,
            scoring="f1"
            ).mean()


        # Hyperparameter grid
        param_grid = [
        {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1'],
            'model__solver': ['liblinear'],
            'model__class_weight': [None, 'balanced'],
            'model__max_iter': [1000, 2000]
        },
        {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l2'],
            'model__solver': ['lbfgs', 'liblinear'],
            'model__class_weight': [None, 'balanced'],
            'model__max_iter': [1000, 2000]
        }
        ]
        #  GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )

        logging.info("Hyperparameter Tuning Started")

        grid_search.fit(X_train,y_train)
        
        logging.info("Hyperparameter Tuning Ended")

        tuned_model = grid_search.best_estimator_
        Tuned_ev_score = evalution_model(tuned_model,X_test,y_test,'Tuned Logistic Regression')
        tuned_cv_score = grid_search.best_score_

        # comparing both model

        print('Model Comparision') 
        print(f'\n Baseline Logistic Regression CV score: {Baseline_cv_score}')
        print(f'\n Tuned Logistic Regression CV score:{tuned_cv_score}')

        logging.info("Comparining Baseline And Tuned Model")

        a = int(input("\n Enter 1 for Baseline Model And 2 For Tuned Model:"))

        if a == 1:
            best_model = Baseline_model

        elif a == 2:
            best_model = tuned_model

        else:
            print("Invalid Input")
            best_model = None

        if best_model is not None:
            joblib.dump(
                best_model,
                "models/employee_attrition_model1.pkl"
            )

        logging.info('Best Model Saved')
        logging.info('Training Completed!')




    except Exception as e:
        logging.error('Error Occured during Training!')
        raise CustomException(e,sys)


if __name__ == "__main__":
        train_model()