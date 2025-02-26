import pandas as pd

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
TF_IDF=TfidfVectorizer()
#Bag-of-words
from sklearn.feature_extraction.text import CountVectorizer
BOW = CountVectorizer()

#resampling
from imblearn.under_sampling import RandomUnderSampler
under_sampler = RandomUnderSampler()
    
from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

random_forest = RandomForestClassifier()
SVM = SVC()

def raw_step(df,vectorizer, sampler, method):
      """
      This function is for calculating Accuracy and ROC AUC score, Accuracy and ROC AUC gap between test set and train set 
      
      Input:
      df: dataframe used
      vectorizer: Text vectorizer (we expect: TF_IDF and Bag-of-words)
      sampler: method to resample (we expect: oversampling and undersampling)
      method: machine learning classification (we expect: Random Forest and SVM)

      Output: Accuracy and ROC AUC score, Accuracy and ROC AUC gap between test set and train set 
      """
      def text_vector(df,vectorizer):
              """
              This function is for transforming texts into a matrix
              
              Input:
              df: dataframe used
              vectorizer: Text vectorizer (we expect: TF_IDF and Bag-of-words)
              
              Output: A matrix of weight of each word
              """
              X1=vectorizer.fit_transform(df["suggestions_lemm"])
              X2=vectorizer.fit_transform(df["services_lemm"])
              #Concatenate these 2 text columns
              X = pd.concat([pd.DataFrame(X1.toarray()), pd.DataFrame(X2.toarray())],  axis=1)
              return X
      X = text_vector(df,vectorizer)
      
      def sampling(df,X, sampler):
                """
                This function is for resampling the dataset, in order to solve the problem of imbalanced dataset and split the data
                set into train set and test set
              
                Input:
                df: dataframe used
                sampler: method to resample (we expect: oversampling and undersampling)
                
                Output: Dataframe of train set and test set
                """
                X, y =  sampler.fit_resample(X, df['overall_satisfaction_binary'])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #train set and test set
                return X_train, X_test, y_train, y_test
      X_train, X_test, y_train, y_test = sampling(df,X, sampler)
      
      def metrics(method):
                """
                This function is for applying supervised classification and evaluate the model
              
                Input:
                method: machine learning classification (we expect: Random Forest and SVM)
        
                Output: Accuracy and ROC AUC score, Accuracy and ROC AUC gap between test set and train set
                """
                def machine_learning(X_train, X_test, y_train, y_test, method): 
                       """
                       This function is for applying supervised classification and evaluate the model on the test set
                       
                       Input:
                       X_train, y_train: dataframe of train set
                       X_test, y_test: dataframe of test set
                       method: machine learning classification (we expect: Random Forest and SVM)
                       
                       Output: Accuracy and ROC AUC score on test set
                       """
                       method.fit(X_train, y_train)
                       # Predictions
                       y_pred = method.predict(X_test)
                       # Evaluation
                       accuracy = accuracy_score(y_test, y_pred)
                       roc_auc = roc_auc_score(y_test,y_pred)
                       return accuracy, roc_auc
                accuracy_test, roc_auc_test = machine_learning(X_train, X_test, y_train, y_test, method)
                
                def accuracy_trainset(X_train, y_train, method):
                       """
                       This function is for evaluate the model on the train set
                       
                       Input:
                       X_train, y_train: dataframe of train set
                       method: machine learning classification (we expect: Random Forest and SVM)
                       
                       Output: Accuracy and ROC AUC score on train set
                       """
                       y_pred_train = method.predict(X_train)
                       accuracy = accuracy_score(y_train, y_pred_train)
                       roc_auc = roc_auc_score(y_train, y_pred_train)
                       return accuracy,roc_auc
                accuracy_train, roc_auc_train = accuracy_trainset(X_train, y_train, method)
                
                def metrics_gap(accuracy_test,accuracy_train, roc_auc_test, roc_auc_train):
                       """
                       This function is for calculate the gap of metrics between train set and test set
                       
                       Input:
                       accuracy_test: accuracy score calculated previously on test set
                       accuracy_train: accuracy score calculated previously on train set
                       roc_auc_test: ROC AUC score calculated previously on test set
                       roc_auc_train: ROC AUC score calculated previously on train set

                       Output: Accuracy and ROC AUC gap between test set and train set
                       """
                       accuracy_gap = accuracy_test - accuracy_train
                       roc_auc_gap = roc_auc_test - roc_auc_train
                       return accuracy_gap,roc_auc_gap
                accuracy_gap, roc_auc_gap = metrics_gap(accuracy_test,accuracy_train, roc_auc_test, roc_auc_train)
                return accuracy_test, accuracy_gap, roc_auc_test,roc_auc_gap
      accuracy_test, accuracy_gap, roc_auc_test,roc_auc_gap = metrics(method)
      return accuracy_test, accuracy_gap, roc_auc_test,roc_auc_gap
      

def predict_sentiments(df, result):    
    """
    This function is to stock all the results calculated previously in a dataframe, allowing to compare the efficiency of different
    methods more easily

    Input:
    df: dataframe used for calculating metrics
    result: dataframe to stock the results

    Output:
    A dataframe of results
    """
    accuracy_test1, accuracy_gap1, roc_auc_test1,roc_auc_gap1 = raw_step(df,TF_IDF, under_sampler, random_forest)
    accuracy_test2, accuracy_gap2, roc_auc_test2,roc_auc_gap2 = raw_step(df, TF_IDF, under_sampler, SVM)
    accuracy_test3, accuracy_gap3, roc_auc_test3,roc_auc_gap3 = raw_step(df,TF_IDF, over_sampler, random_forest)
    accuracy_test4, accuracy_gap4, roc_auc_test4,roc_auc_gap4 = raw_step(df,TF_IDF, over_sampler, SVM)
    accuracy_test5, accuracy_gap5, roc_auc_test5,roc_auc_gap5 = raw_step(df,BOW, under_sampler, random_forest)
    accuracy_test6, accuracy_gap6, roc_auc_test6,roc_auc_gap6 = raw_step(df, BOW, under_sampler, SVM)
    accuracy_test7, accuracy_gap7, roc_auc_test7,roc_auc_gap7 = raw_step(df,BOW, over_sampler, random_forest)
    accuracy_test8, accuracy_gap8, roc_auc_test8,roc_auc_gap8 = raw_step(df,BOW, over_sampler, SVM)
    result['Vectorizer'] = ["TF-IDF","TF-IDF","TF-IDF","TF-IDF","BOW", "BOW","BOW","BOW"]
    result['Sampler'] = ["Undersampling","Undersampling","Oversampling","Oversampling","Undersampling","Undersampling","Oversampling","Oversampling" ]
    result['Classification'] = ["Random forest", "SVM","Random forest", "SVM","Random forest", "SVM","Random forest", "SVM"]
    result['Accuracy'] = [accuracy_test1,accuracy_test2, accuracy_test3, accuracy_test4, accuracy_test5, accuracy_test6, accuracy_test7,accuracy_test8]
    result['Accuracy gap between train set and test set'] = [accuracy_gap1,accuracy_gap2, accuracy_gap3, accuracy_gap4, accuracy_gap5, accuracy_gap6, accuracy_gap7, accuracy_gap8]
    result['ROC AUC Score'] = [roc_auc_test1,roc_auc_test2, roc_auc_test3, roc_auc_test4, roc_auc_test5, roc_auc_test6, roc_auc_test7,roc_auc_test8]
    result['ROC AUC Score gap between train set and test set'] = [roc_auc_gap1,roc_auc_gap2, roc_auc_gap3, roc_auc_gap4, roc_auc_gap5, roc_auc_gap6, roc_auc_gap7, roc_auc_gap8]


