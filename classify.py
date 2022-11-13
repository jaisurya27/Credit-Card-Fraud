from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
#import tensorflow as tf
#from tensorflow import keras
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('creditcard.csv')

scaler = StandardScaler()
scaler2 = StandardScaler()
#scaling time
scaled_time = scaler.fit_transform(df[['Time']])
flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]
scaled_time = pd.Series(flat_list1)

scaled_amount = scaler2.fit_transform(df[['Amount']])
flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]
scaled_amount = pd.Series(flat_list2)

df = pd.concat([df, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)
df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.iloc[:, df.columns!='Class']
y = df.iloc[:, df.columns == 'Class']

X_resample, y_resample = SMOTE().fit_resample(X,y.values.ravel())
y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)
X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.25)

#print(len(df.Class==1))

model = Sequential([
    Dense(units=18, input_dim = 30,activation='relu'),
    Dense(units=26,activation='relu'),
    Dropout(0.5),
    Dense(22,activation='relu'),
    Dense(22,activation='relu'),
    Dense(1,activation='sigmoid'),
])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=10,epochs=10)

# score = model.evaluate(X_test, y_test)
# print(score)

# y_pred = model.predict(X_test)
# y_expected = pd.DataFrame(y_test)

# best classification algorithm: Random forester:

clf = RandomForestClassifier(n_estimators = 10) 
clf.fit(X_train, y_train)

# print(classification_report(y_expected, y_pred))
# print('ROC AUC Score: ',roc(y_expected, y_pred))
# cnf_matrix = confusion_matrix(y_expected, y_pred.round())
#print(cnf_matrix)

pickle.dump(model,open('model.pkl','wb'))
pickle.dump(clf,open('clf.pkl','wb'))