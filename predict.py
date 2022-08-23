# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:53:36 2020

@author: rajee
"""
import numpy as np
import sys
import csv
from validate import validate

def compute_likelihood(test_X, classes,model,d):
    likelihood=0
    word=test_X.split()
    for i in word:
        count=0
        frequency=model[classes]
        if i in frequency:
            count=model[classes][i]
        likelihood+=np.log((count+1)/d[classes])
        
    return likelihood


def predic(test_X,model,prior,d,classes):
    
    n=len(test_X)
    pred=np.zeros((n,1))
    for i in range(n):
        best_p=-99999
        best_c=-1
        for c in classes:
            p=compute_likelihood(test_X[i],c,model,d)+np.log(prior[c])
            if p>best_p:
                best_p=p
                best_c=c
        pred[i][0]=best_c
    return pred

def import_data_and_model(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n',dtype=str)
    return test_X


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def predict(test_X_file_path):
    
    
    Y=np.genfromtxt("train_Y_nb.csv",delimiter=',')
    X = np.genfromtxt("train_X_nb.csv",delimiter='\n',dtype=str)
    
    
    model=dict()
    for i in range(len(Y)):
        if Y[i] not in model:
            model[Y[i]]=dict()
        words=X[i].split()
        for j in words:
            if j not in model[Y[i]]:
                model[Y[i]][j]=0
            model[Y[i]][j]+=1
    
    prior_prob=dict()
    l=list(Y)
    n=len(Y)
    classes=list(set(l))
    classes.sort()
    for c in classes:
        prior_prob[c]=l.count(c)/n
    V=[]
    d=dict()
    for i in classes:
        if i not in d:
            d[i]=0
    for i in classes:
        m=model[i]
        d[i]=sum(list(m.values()))
        V+=list(m.keys())
    V=list(set(V))
    
    for c in classes:
        d[c]+=len(V)
    
    
    
    test_X= import_data_and_model(test_X_file_path)
    pred_Y=predic(test_X,model,prior_prob,d,classes)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv")