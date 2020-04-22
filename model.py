#  Copyright (c) 2020, Danish Shakeel
#  https://danishshakeel.me
#  https://github.com/danish17
#  hi@danishshakeel.me

#Import required libraries:
import numpy as np
import pandas as pd
from keras import Sequential,Model
from keras.layers import Dense

#Read the dataset:
df = pd.read_csv('data6.csv')

#View the head of the DF:
print(df.head())

#Basic schema:
print(df.columns)
print(len(df.columns))
print(df.groupby('class').count()['F1']) #Balanced dataset, no need to resample

#Stats:
df.corr()
df.describe()

#Correlation to reduce dimensions:
print("BEFORE FEATURE SELECTION:", df.shape)
correlated_features = set()
correlation_matrix = df.corr()
for i in range(len(correlation_matrix .columns)): #Step forward wrapper method
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.75:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

df.drop(labels=correlated_features, axis=1, inplace=True)
print("AFTER FEATURE SELECTION:", df.shape)

#Create Feature matrix, and target vector:
X = df.iloc[:,0:6].values
y = df.iloc[:,6].values

#Normalization:
from sklearn.preprocessing import normalize
X = normalize(df,norm='l2',axis=1)

#Split into train-test sets:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=12)
print("Training set has {} examples, and test set has {} examples".format(len(x_train),len(x_test)))

#Checking X shape:
print(x_train.shape)

#Creating a Feedforward NN or MLP:
model = Sequential()
model.add(Dense(7, input_dim=7, activation='hard_sigmoid')) #0th layer
model.add(Dense(16, activation='hard_sigmoid')) #1st layer (hidden)
model.add(Dense(1, activation='sigmoid')) #2nd layer

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=500,batch_size=32)

print("TEST ACCURACY: {}".format(model.evaluate(x_test,y_test)[1]))

y_pred = model.predict(x_test)
y_pred[y_pred>0.5] = 1
y_pred[y_pred<=0.5] = 0

#Evaluate:
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) #I got around 90% accuracy

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test,y_pred,average='binary') #R(0.9) P(0.9)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))