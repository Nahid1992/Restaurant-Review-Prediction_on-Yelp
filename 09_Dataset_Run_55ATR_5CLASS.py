import argparse
import collections
import csv
#import simplejson as json
import json
import pandas as pd
import matplotlib.pyplot as plt
import operator
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import sklearn.naive_bayes 
from sklearn import naive_bayes, model_selection


result_DF = pd.DataFrame(columns=['Algo_Name','R2-Score', 'MSE', 'Precision','Recall','Accuracy','F1-Score'])

result_Index = 0

#####################################################################################
def kappa(test,pred,algo_name):
    path_name = "Results/09_Dataset_Run_55ATR_5CLASS.txt"
    cm = metrics.confusion_matrix(test, pred)
    kc = metrics.cohen_kappa_score(test, pred)
    text_file = open(path_name,"a")
    text_file.write("\n")
    text_file.write(algo_name + "\n")
    np.savetxt(text_file,cm)   
    text_file.write(str(kc))
    text_file.write("\n")
    text_file.close()
    
def Decision_Tree_Classifier(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Decision Tree Classifier'
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred_dtc=clf.predict(X_test)
#    kappascore=metrics.cohen_kappa_score(y_test, y_pred_dtc) #NOT PRINTED
    PRAF(y_test, y_pred_dtc,num_class,algo_name, reorder=False)  
    kappa(y_test,y_pred_dtc,algo_name)
    
def Random_Forest_Regression(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Random Forest Regression'
    
    
    clf = RandomForestClassifier()
    
    
    
    parameters = {'n_estimators': [4, 6, 10, 50, 100, 200, 500],
                  'max_features': ['log2', 'sqrt'],
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [2, 3, 5, 10],
                  'min_samples_split': [2, 3 ,5],
                  'min_samples_leaf': [1, 3, 5]
                  }
    
#    parameters = {'n_estimators': [4],
#                  'max_features': ['log2', 'sqrt'],
#                  'criterion': ['entropy', 'gini'],
#                  'max_depth': [3],
#                  'min_samples_split': [3],
#                  'min_samples_leaf': [2]
#                  }
    acc_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)
    clf = grid_obj.best_estimator_
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)  
    PRAF(y_test, predictions,num_class,algo_name)
    kappa(y_test,predictions,algo_name)
    
def SGD_Classification(X_train,y_train,X_test,y_test,num_class):
    algo_name = "SGD_Classification"
    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    pred = sgd.predict(X_test)
    PRAF(y_test, pred,num_class,algo_name, reorder=False)  
    kappa(y_test,pred,algo_name)
    
def Multi_Nomial_NB(X_train,y_train,X_test,y_test,num_class):
    algo_name = "Multi Nomial NB"
    clf = naive_bayes.MultinomialNB()
    clf.fit(X_train, y_train)
#    pred = model_selection.cross_val_predict(clf, X_train, y_train)   
    pred = clf.predict(X_test)
    PRAF(y_test, pred,num_class,algo_name, reorder=False)  
    kappa(y_test,pred,algo_name)
    
###########################################################################


def wfile(file,list):
    f=open(file,'w')
    f.writelines(["%s\n" % item  for item in list])
def readStringList(file_name):
    with open(file_name) as file:
        lines = [line.strip() for line in file]
    return lines

def twoClass(YY):
    #YY = map(int,YY)
    for index in range(0,len(YY)):
        if YY[index] <= 3: # Class 1,2,3 = Low = 1
            YY[index] = 0
        else:
            YY[index] = 1 # Class 4,5 = High = 2
    return YY

def threeClass(YY):
    #YY = map(int,YY)
    for index in range(0,len(YY)): # Class 1 = Low = 1
        if YY[index] == 1:
            YY[index] = 0
        elif YY[index]==2 or YY[index]==3: # Class 2,3 = Mid = 2
            YY[index] = 1
        elif YY[index]==4 or YY[index]==5: # Class 4,5 = High = 3
            YY[index] = 2
    return YY

def PRAF(test, pred, num_class, algo_name, reorder=True):
    if reorder == True:    
        for index in range(0,len(pred)):
            if pred[index] < 0:
                pred[index] = 0
            if num_class == 2:
                pred[index] = round(pred[index])
            else:
                pred[index] = (int)(round(pred[index]))

    pred = map(int,pred)                 
    if num_class <3:
        precision = metrics.precision_score(y_true=test, y_pred=pred)
        recall = metrics.recall_score(y_true=test, y_pred=pred)
        f1 = metrics.f1_score(y_true=test, y_pred=pred)
        accuracy = metrics.accuracy_score(y_true=test, y_pred=pred)
    else:
        precision = metrics.precision_score(y_true=test, y_pred=pred,average='macro')
        recall = metrics.recall_score(y_true=test, y_pred=pred,average='macro')
        f1 = metrics.f1_score(y_true=test, y_pred=pred,average='macro')
        accuracy = metrics.accuracy_score(y_true=test, y_pred=pred)
    
    
    r2score=metrics.r2_score(y_true=test, y_pred=pred)     
    mse=metrics.mean_squared_error(y_true=test, y_pred=pred) 
    
    print(algo_name)
    print "R-squared: ",r2score
    print "MSE: ",mse
    print "Precision: ",precision
    print "Recall: ",recall
    print "accuracy: ",accuracy
    print "f1: ",f1 
    print("----------------------------------------------")
    result_DF.loc[algo_name]=[algo_name,r2score,mse,precision,recall,accuracy,f1]

    
def showAttributes(xTicks,selectedSum,selectedList):
    plt.figure(1)  
    plt.title('Number of Attributes: ' + str(55))
    plt.ylabel('Attributes')
    plt.xlabel('Count')
    plt.barh(xTicks,selectedSum)
    plt.yticks(xTicks,selectedList)  
    plt.tight_layout()
    path_name = "Results/09_Dataset_Run_55ATR_5CLASS "
    strplotname = path_name + ".png"
#    plt.savefig(strplotname)
    plt.show()         
    
#ALGORITHMS
def Linear_Regression(X_train,y_train,X_test,y_test,num_class):
    print(" ")
    print("----------------------------------------------")
    algo_name = 'Linear Regression'
    lm = LinearRegression() 
    model = lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)  
    PRAF(y_test, predictions,num_class,algo_name)

def Ridge_Mode(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Ridge Regression'
    ridge_model = RidgeCV(alphas=[0.01,0.05,0.10,0.20,0.50,1])     
    ridge_model.fit(X_train,y_train)       
    y_pred_rm = ridge_model.predict(X_test)    
    PRAF(y_test, y_pred_rm,num_class,algo_name)

def Lasso_Mode(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Lasso Regression'
    lasso_model = LassoCV(alphas=[0.01,0.05,0.10,0.20,0.50,1])
    lasso_model.fit(X_train,y_train)
    y_pred_lm = lasso_model.predict(X_test)
    PRAF(y_test, y_pred_lm,num_class,algo_name)

def Decision_Tree_Regression(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Decision Tree Regression'
    dt_model = DecisionTreeRegressor(random_state=1)
    dt_model.fit(X_train,y_train)
    y_pred_dt = dt_model.predict(X_test)
    PRAF(y_test, y_pred_dt,num_class,algo_name)
    
def Random_Forest_Regression(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Random Forest Regression'
    rf_model = RandomForestRegressor(max_depth=4,n_estimators=100,max_features='sqrt',verbose=1,random_state=1)
    rf_model.fit(X_train,y_train)
    y_pred_rf = rf_model.predict(X_test)  
    PRAF(y_test, y_pred_rf,num_class,algo_name)
    
def Adaboost_Regression(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Adaboost Regression'
    adb_model = AdaBoostRegressor(n_estimators=100,learning_rate=0.01,random_state=1,loss="square") 
    adb_model.fit(X_train,y_train)
    y_pred_adb = adb_model.predict(X_test)
    PRAF(y_test, y_pred_adb,num_class,algo_name)
    
def Gradient_Boosting_Regression(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Gradient Boosting Regression'
    gbm_model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.01,
                                      random_state=1,max_depth=4,max_features="sqrt")
    gbm_model.fit(X_train,y_train)    
    y_pred_gbm = gbm_model.predict(X_test)
    PRAF(y_test, y_pred_gbm,num_class,algo_name)
    
def Decision_Tree_Classifier(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Decision Tree Classifier'
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred_dtc=clf.predict(X_test)
    kappascore=metrics.cohen_kappa_score(y_test, y_pred_dtc) #NOT PRINTED
    PRAF(y_test, y_pred_dtc,num_class,algo_name, reorder=False)     

def Support_Vector_Machine(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Support Vector Machine'
    svm_model_linear = SVC(kernel='linear',C=1).fit(X_train,y_train)
    svm_predictions = svm_model_linear.predict(X_test)
    PRAF(y_test, svm_predictions,num_class,algo_name)
    
if __name__ == '__main__':

    print "09_Dataset_Run_55ATR_5CLASS => New Set + No Cut"

    db = pd.read_csv('sample_100000.csv', dtype=str)
    
    attr_names = readStringList('selected_all_atr.txt')
    encoding_attr_names = readStringList('selected_all_atr_encoding.txt')
#    attr_names = readStringList('selected_columns.txt')
#    encoding_attr_names = readStringList('selected_encoding_columns.txt')    
    print("Actual DB size")
    print(db.shape)    
    db = db[attr_names]
    db = pd.get_dummies(db, columns=encoding_attr_names)    
    ATR_ALL = readStringList('selected_all_atr_encoded_true.txt')
    db = db[ATR_ALL]
    db = db.dropna()

    #Analysis -Nahid
    sumAttr = []  
    sumAttr_name = []  
    for col in db.columns:
        atr_name = col.split(".")[0]
        if atr_name == 'attributes':            
            sumAttr.append(sum(map(int,db[col])))
            sumAttr_name.append(col)
         
    sumAttr_mean = np.mean(sumAttr)
    sumAttr_median = np.median(sumAttr)
    print('Mean = ' + str(sumAttr_mean))
    print('Median = ' + str(sumAttr_median))
    dropList = []
    count = 0
    selectedList = []
    selectedSum = []
    xTicks = []
    count_index = 1
    for col in db.columns:
        atr_name = col.split(".")[0]
        if atr_name == 'attributes':            
            SUM = sum(map(int,db[col]))
            if SUM > sumAttr_mean:
                count = count + 1
                splitText = col.split('_')[0]
                splitText = splitText.split('.')[-1]
                selectedList.append(splitText)
                selectedSum.append(SUM)
                xTicks.append(count_index)
                count_index = count_index+1
            else:
                dropList.append(col)
    print('Attr. Count = ' + str(count))    
    
    showAttributes(xTicks,selectedSum,selectedList)    

    dropList.append('stars_review')
    dfx = db.drop(['stars_review'],axis=1)
    dfy = db['stars_review']
  

#Converting everything to NUMERIC Float/Int
    for col in dfx.columns:
        atr_name = col.split(".")[0]
        if atr_name == 'attributes':            
            dfx[col] = (map(int,dfx[col]))
            #sum_res = sum_res + sum(dfx[col])
        else:
            dfx[col] = (map(float,dfx[col]))
    
    dfy = map(int,dfy)
#-----------------------------------------------------------#    
    
#    dt=db.head(1000)
#    dt.to_csv("sampleFirst1000.csv")
    
    print("Filtered DB size")
    print('dfx Length = ' + str(len(dfx)))
    print('dfy Length = ' + str(len(dfy)))
     
    X_train, X_test, y_train, y_test = train_test_split(dfx,dfy,test_size=0.25)
    
    num_class = 5
#    y_train = threeClass(y_train)
#    y_test = threeClass(y_test)
#    y_train = twoClass(y_train)
#    y_test = twoClass(y_test)
    
    
    breakpoint = 1
    
#Algorithms:
#    Linear_Regression(X_train,y_train,X_test,y_test,num_class)
#    Lasso_Mode(X_train,y_train,X_test,y_test,num_class)
#    Adaboost_Regression(X_train,y_train,X_test,y_test,num_class)
    
    
    
#    Decision_Tree_Regression(X_train,y_train,X_test,y_test,num_class)    
#    Gradient_Boosting_Regression(X_train,y_train,X_test,y_test,num_class)
#    Ridge_Mode(X_train,y_train,X_test,y_test,num_class)
    
#    Support_Vector_Machine(X_train,y_train,X_test,y_test,num_class)
#    Decision_Tree_Classifier(X_train,y_train,X_test,y_test,num_class)
#    Random_Forest_Regression(X_train,y_train,X_test,y_test,num_class)
#    SGD_Classification(X_train,y_train,X_test,y_test,num_class)
#    Multi_Nomial_NB(X_train,y_train,X_test,y_test,num_class)
#
#    result_DF.to_csv('Results/09_Table_Result_55ATR_5CLASS.csv')