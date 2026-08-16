from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



num_features = ['Age','Monthly_Salary','Years_At_Company','Job_Satisfaction']


cat_features = ['Overtime']


def create_preprocessor():

    preprocessor = ColumnTransformer(transformers = [
    ("Num",StandardScaler(),num_features),
    ('Cat',OneHotEncoder(),cat_features)
    ])

    return preprocessor


def evalution_model(model,X_test,y_test,model_name=str):

    y_pred = model.predict(X_test)

    Accuracy= accuracy_score(y_test, y_pred)
    Classification_report = classification_report(y_test,y_pred)
    Confusion_matrix = confusion_matrix(y_test,y_pred)

    print(f'\n {model_name} Performance : ')
    print(f" \n Accuracy: {Accuracy}")
    print(f" \n Classification Report:{Classification_report}")
    print(f"\n Confusion Matrix: {Confusion_matrix}")
    print('-'*100)

    return 