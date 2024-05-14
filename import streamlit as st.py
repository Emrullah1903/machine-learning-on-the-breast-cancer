import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import pandas as pd

@st.cache_data
def load_data():
    data = pd.read_csv('Breast_Cancer.csv')
    return data

def preprocess_data(X):
  
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

def knn_hyperparameter_tuning(X, y):
    '''
    X: independent variable
    y: dependent variable
    return best model, its accuracy, and confusion matrix
    '''
    
    X = preprocess_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNeighborsClassifier()

    params = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    st.write("Model: KNeighborsClassifier")
    st.write("Best Parameters: ", grid.best_params_)
    st.write("Model Accuracy: ", accuracy_score(y_pred, y_test))
    st.write("Model Confusion Matrix: ", confusion_matrix(y_pred, y_test), "\n")
    st.write("Model Classification Report: ", classification_report(y_pred, y_test))
    st.write("-"*50)

    return best_model, accuracy_score(y_pred, y_test), confusion_matrix(y_pred, y_test), classification_report(y_pred, y_test)


def main():
    st.title("Kanser Tahmini")
    st.sidebar.header("Giriş Değerleri")

    clump_thickness = st.sidebar.slider("Clump Thickness", 1, 10, 1)
    uniformity_of_cell_size = st.sidebar.slider("Uniformity of Cell Size", 1, 10, 1)
    uniformity_of_cell_shape = st.sidebar.slider("Uniformity of Cell Shape", 1, 10, 1)
    marginal_adhesion = st.sidebar.slider("Marginal Adhesion", 1, 10, 1)
    single_epithelial_cell_size = st.sidebar.slider("Single Epithelial Cell Size", 1, 10, 1)
    bare_nuclei = st.sidebar.slider("Bare Nuclei", 1, 10, 1)
    bland_chromatin = st.sidebar.slider("Bland Chromatin", 1, 10, 1)
    normal_nucleoli = st.sidebar.slider("Normal Nucleoli", 1, 10, 1)
    mitoses = st.sidebar.slider("Mitoses", 1, 10, 1)


    data = load_data()
    X = data.drop('Class', axis=1)  # Bağımsız değişkenler
    y = data['Class']  # Bağımlı değişken

   
    best_knn_model, knn_accuracy, knn_confusion_matrix, knn_classification_report = knn_hyperparameter_tuning(X, y)

   
    input_data = [[clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape, 
                   marginal_adhesion, single_epithelial_cell_size, bare_nuclei, 
                   bland_chromatin, normal_nucleoli, mitoses]]
    prediction = best_knn_model.predict(input_data)

    st.subheader("Sonuç")
    if prediction[0] == 2:
        st.write("Kanser Değil")
    elif prediction[0] == 4:
        st.write("Kanser")

if _name_ == "_main_":
    main()
