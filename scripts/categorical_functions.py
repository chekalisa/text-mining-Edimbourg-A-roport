import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def cramers_v(a, b):
    '''
    This function calculates Cramér's V statistic in order to measure the association between two categorical variables.

    Arguments:
    a : the first categorical variable, an array structure
       
    b : the second categorical variable, an array structure
        

    Returns:
    float
        The Cramér's V statistic is a value between 0 and 1 that indicates th strength of the association
    '''
    confusion_matrix = pd.crosstab(a, b)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min((k - 1), (r - 1)))

def calculate_cramers_v_matrix(df):
    '''
    This function calculates a matrix of Cramér's V statistics for all pairs of categorical variables in a DataFrame

    Arguments:
    df : the dataframe that contains the categorical variables we want to analyze
        

    Returns:
    pandas DataFrame where each cell (i, j) contains the Cramér's V statistic for the variables in columns i and j of the input DataFrame

    '''
    columns = df.columns
    cramers_v_matrix = pd.DataFrame(index=columns, columns=columns)
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                cramers_v_matrix.loc[col1, col2] = 1.0 
            else:
                cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    cramers_v_matrix = cramers_v_matrix.astype(float)
    return cramers_v_matrix

def plot_cramers_v_heatmap(cramers_v_matrix):
    '''
    This function plots a heatmap of the Cramér's V matrix

    Arguments:
    cramers_v_matrix : pandas DataFrame
        A DataFrame where each cell (i, j) contains the Cramér's V statistic for the variables in columns i and j

    Returns:
    The function displays a heatmap of the Cramér's V matrix.
    '''
    plt.figure(figsize=(30, 22))
    sns.heatmap(cramers_v_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Cramér's V Matrix")
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.show()


def drop_columns(df):
    '''
    Instead of using a function that was extremely time-consumming,  we eliminated iteratively highly correlated variables one by one in order to determine the problematic variables in terms of correlation
    We looked at the most correlated pair of variables and we eliminated one of them - the one that was least correlated with the target variable and then moved to another pair
    This function drops specific columns from the DataFrame and resets the index

    Arguments:
    df : pandas DataFrame

    Returns:
    pandas DataFrame with the specified columns removed and the index reset
    '''
    df.drop(columns=['Luggage porterage','age_36-55','next_flight_Within *1-3 months*','arrdep_Departed and arrived','Private terminal with airside vehicle collection/pick-up','Home collection baggage services','age_56-65','num_trav_3','num_trav_4'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def oversample_data(df, target_column, test_size=0.3, random_state=0):
    '''
    This function oversamples the data using SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.

    Arguments:
    df : the dataframe containing both the features and the target variable
        
    target_column : str, the name of the target variable
    
    Returns:
    tuple
        A tuple containing X_train, X_test, y_train, y_test, os_data_X, os_data_y.
        X_train: the features used for training
        X_test: the features used for testing
        y_train: the target variable used for training
        y_test: the target variable used for testing
        os_data_X: the oversampled features
        os_data_y: the oversampled target variable
    '''
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    
    os = SMOTE(random_state=random_state)
    columns = X_train.columns
    y_train = y_train.values.ravel()
    os_data_X, os_data_y = os.fit_resample(X_train, y_train)

 
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=[target_column])


    print("Length of oversampled data is ", len(os_data_X))
    print("Number of unsatisfied visitors in oversampled data", len(os_data_y[os_data_y[target_column] == 0]))
    print("Number of satisfied visitors", len(os_data_y[os_data_y[target_column] == 1]))
    print("Proportion of unsatisfied visitors in oversampled data is ", len(os_data_y[os_data_y[target_column] == 0]) / len(os_data_X))
    print("Proportion of satisfied visitors in oversampled data is ", len(os_data_y[os_data_y[target_column] == 1]) / len(os_data_X))

    return X_train, X_test, y_train, y_test, os_data_X, os_data_y



def select_features(estimator, X, y):
    '''
    This function selects the optimal number of features using Recursive Feature Elimination with Cross-Validation (RFECV)

    Arguments:
    estimator : object, the machine learning estimator (e.g., classifier) to fit
    X : the data frame containing the features 
    y : the target variable

    Returns:
    Index, the names of the selected features
    '''

    rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5), scoring='f1')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    selected_features = X.columns[rfecv.support_]
    print("Selected features:", selected_features)

    return selected_features


def logit_regression(X, y, selected_features):
    '''
    This function fits a logistic regression model using the selected features and prints the summary of the model

    Arguments:
    X : dataFrame, the feature matrix
    y : series or array-like, the target variable
    selected_features : list of str, the list of selected feature names to be used in the model

    Returns:
    result : statsmodels LogitResults, the result of the fitted logistic regression model
    '''

    
    X_with_constant = sm.add_constant(X[selected_features])
    logit_model = sm.Logit(y, X_with_constant)
    result = logit_model.fit()
    print(result.summary2())
    return result

def evaluate_model(X_test, y_test, selected_features, result):
    '''
    This function evaluates the performance of a fitted logistic regression model on the test set using 5 metrics

    Arguments:
    X_test : DataFrame, the feature matrix for the test set
    y_test : Series or array, the target variable for the test set
    selected_features : list of str, the list of selected feature names to be used in the model
    result : statsmodels LogitResults, the result of the fitted logistic regression model

    Returns:
    tuple, a tuple containing the evaluation metrics: accuracy, precision, recall, F1 score, and ROC AUC score
    '''
    
    X_test_with_constant = sm.add_constant(X_test[selected_features])
    y_pred = result.predict(X_test_with_constant)

   
    threshold = 0.5
    y_pred_binary = (y_pred > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")

    return accuracy, precision, recall, f1, roc_auc


def random_forest(df, target):
    '''
    This function trains and evaluates a Random Forest classifier on the given DataFrame

    Arguments:
    df : DataFrame, the DataFrame containing features and the target variable
    target : str, the name of the target column

    Returns:
    tuple, a tuple containing the trained Random Forest model, the test feature matrix (X_test), the test target variable (y_test), 
    the predicted labels (y_pred), and the predicted probabilities (y_prob)
    '''
    X = df.drop(columns=[target])
    y = df[target]
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    
    return rf_model, X_test, y_test, y_pred, y_prob
