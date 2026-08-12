import sys
import os
import pandas as pd
import numpy as np
import word2number as w2n
from src.exception import CustomException
from src.logger import logging




# ------------------------------------------------------------------------------------------------------------
# Individual cleaning functions
# ------------------------------------------------------------------------------------------------------------

def clean_age(series):
    def clean_age_value(value):

        if pd.isna(value):
            return np.nan
    
        if isinstance(value, (int, float)):
            age = value
    
        else:
            value = value.strip().lower()
    
            try:
                age = pd.to_numeric(value)
            except:
                try:
                    age = w2n.word_to_num(value)
                except:
                    return np.nan
                    
        if age < 1 or age > 100:
            return np.nan
    
        return age

    series = series.apply(clean_age_value)

    series = series.fillna(series.median())

    series = series.astype(int)

    return series




def clean_Gender(series):

    series = series.str.upper()

    series = series.fillna(series.mode()[0])

    return series





def clean_Dep(series):

    series = series.str.strip()

    series = series.str.upper()

    series = series.fillna(series.mode()[0])

    return series



def clean_job_level(series):
    
    series = series.str.strip()
    series = series.str.title()
    return series




def clean_salary(series):

    def clean_salary_value(value):

        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float)):
            salary = value

        value = str(value).strip()

        value = value.replace('₹', '').replace(',', '').strip()

        try:
            salary = pd.to_numeric(value)
        except:
            return np.nan

        if salary < 0:
            return np.nan

        return salary

    series = series.apply(clean_salary_value)

    series = series.fillna(series.median())

    return series.astype(int)




def clean_year_at_company(series):
    def clean_feature_value(value):

        if pd.isna(value):
            return np.nan
    
        if value < 0:
            return np.nan
    
        return value
    series = series.apply(clean_feature_value)
    series = series.fillna(series.median())

    return series.astype(int)




def clean_overtime(series1,series2):

    series1 = series1.str.title()

    series1 =(series1.groupby(series2).transform(lambda x: x.fillna(x.mode()[0])))

    return series1




def clean_job_satisfaction(series1,series2):
    def clean_satisfaction_value(value):

        if pd.isna(value):
            return np.nan
    
        if value < 1 or value > 5:
            return np.nan
    
        return value
    series1 = series1.apply(clean_satisfaction_value)
    series1 = (
        series1.groupby(series2)
               .transform(lambda x: x.fillna(x.median())))
    return series1.astype(int)





def clean_Performance(series):
    def clean_performance_value(value):

        if pd.isna(value):
            return np.nan
            
        if isinstance(value, str):
            value = value.strip().lower()

            if value == 'excellent':
                value = 5
        try:
             value = pd.to_numeric(value)
        except:
            return np.nan

        if value < 1 or value > 5:
            return np.nan

        return value

 
    series = series.apply(clean_performance_value)

    series = series.fillna(series.median())

    return series.astype(int)






def clean_work_life(series1,series2):
    def clean_worklife_value(value):

        if pd.isna(value):
            return np.nan
            
        if value < 1 or value > 5 :
            return np.nan

        return value

    series1 = series1.apply(clean_worklife_value)

    series1 = (series1.groupby(series2).transform(lambda x: x.fillna(x.median())))

    return series1.astype(int)





def clean_Training(series):
    def clean_training_value(value):

        if pd.isna(value):
            return np.nan


        if isinstance(value, (int, float)):
            hour = value

        else:
            value = value.strip().lower()
    
            try:
                hour = pd.to_numeric(value)
            except:
                try:
                    hour = w2n.word_to_num(value)
                except:
                    return np.nan
        if hour < 0:
            return np.nan
            
        return hour

    series = series.apply(clean_training_value)

    series = series.fillna(series.median())

    return series.astype(int)






def clean_city(series):

    series = series.str.strip()
    series = series.str.title()

    series = series.fillna(series.mode()[0])

    return series




def clean_Remote_work(series1,series2):
    
    series1 = series1.str.strip()
    series1 = series1.str.title()

    series1 = ( series1.groupby(series2).transform(lambda  x: x.fillna(x.mode()[0])))

    return series1





def clean_left_company(series):
    series = series.astype('string').str.strip().str.lower()
    series = series.map({
        'yes': 1,
        'no': 0,
        '1': 1,
        '0': 0
    })

    series = series.fillna(series.median())
    
    return series






# -----------------------------------------------------------------------------------------
# Main cleaning function
# -----------------------------------------------------------------------------------------



def clean_data(df):
    
    df.drop(columns=['Employee_ID','Name','Email','Phone','Joining_Date'],inplace=True)
    df.drop_duplicates(inplace=True)
    
    df['Age'] = clean_age(df['Age'])

    df.Gender = clean_Gender(df['Gender'])

    df.Department = clean_Dep(df['Department'])

    df.Job_Level = clean_job_level(df['Job_Level'])

    df.Monthly_Salary = clean_salary(df['Monthly_Salary'])

    df.Years_At_Company = clean_year_at_company(df['Years_At_Company'])

    df.Overtime = clean_overtime(df['Overtime'],df['Job_Level'])

    df.Job_Satisfaction = clean_job_satisfaction(df['Job_Satisfaction'],df['Job_Level'])

    df.Performance_Rating = clean_Performance(df['Performance_Rating'])

    df.Work_Life_Balance = clean_work_life(df.Work_Life_Balance,df.Overtime)

    df.Training_Hours = clean_Training(df['Training_Hours'])

    df.City = clean_city(df['City'])

    df.Remote_Work = clean_Remote_work(df['Remote_Work'],df['City'])

    df.Left_Company = clean_left_company(df['Left_Company'])

    return df








if __name__ == "__main__":


    

    try:

        logging.info('Try to Read Data')

        input_path = "data/employee_final_level2.xlsx"
        output_path = "data/cleaned_employee_data.csv"

        df = pd.read_excel(input_path)

        logging.info('Data Cleaning Started')

        df = clean_data(df)

        df.to_csv(output_path, index=False)

        print("Data cleaning completed successfully!")
        logging.info('Data cleaning completed successfully!')

    except Exception as e:

        logging.error('Data cleaning pipeline failed')
        raise CustomException(e,sys)