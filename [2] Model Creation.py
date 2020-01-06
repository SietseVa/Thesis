'''
This scripts uses the data from previous script to create each model.
For each model a subset or the original features is used, denoted by {model}_cols.
(Except for Neural network, which uses all features)
SVM and NN may take longer as these are more computational intensive.
Furthermore, the NN may not produce the same results as it uses random starting states.
'''


#Import Packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
simplefilter(action='ignore', category=FutureWarning)


#Load in Data
train = pd.read_csv('Train Data.csv')
test = pd.read_csv('Test Data.csv')

#Split into X and y
X_train = train.drop('Response', axis=1)
y_train = train['Response']
X_test = test.drop('Response', axis=1)
y_test = test['Response']


#Logistic Regression
lr_cols = ['Aantal Maanden zelfde abo',
'INT_EVENEMENTEN',
'm12_min_m3_ritten_totaal',
'm3_TE_dalweekend',
'INT_ACTIES',
'SOR',
'WVR']

logreg = LogisticRegression()
logreg.fit(X_train[lr_cols], y_train)
pred = logreg.predict(X_test[lr_cols])
prediction = [round(value) for value in pred]
accuracy_lr = (accuracy_score(y_test, prediction))*100
recall_lr = (recall_score(y_test, prediction))*100
precision_lr = (precision_score(y_test, prediction))*100



#Decision Tree
dt_cols = ['REGIO_RANDSTAD NOORD',
'm1_TE_dalweekend',
'INT_REISINFORMATIE',
'DVR',
'INT_GK_TREINTICKETS',
'Contacten_1mnd',
'm12_min_m3_TE_dalweekend',
'Aantal Maanden zelfde abo']

dt_classifier=DecisionTreeClassifier(random_state=12345, max_depth=5)
dt_classifier.fit(X_train[dt_cols], y_train)
pred = dt_classifier.predict(X_test[dt_cols])
prediction = [round(value) for value in pred]
accuracy_dt = (accuracy_score(y_test, prediction))*100
recall_dt = (recall_score(y_test, prediction))*100
precision_dt = (precision_score(y_test, prediction))*100



#Random Forest
rf_cols = ['m1_ritten_totaal',
'INT_EVENEMENTEN',
'INT_NS_BELEVING',
'INT_INTERNATIONAAL',
'OV_FIETS_ABO_KLANT',
'DVR',
'm1_TE_spits_ochtend',
'm12_min_m3_TE_dalweekend',
'NSF_Basis',
'INT_VOORSTELLING',
'ContactenTel_2mnd',
'TVR',
'SEX',
'REGIO_RANDSTAD NOORD',
'INT_FESTIVAL',
'INT_SPORT_BEWEGEN',
'm3_TE_dalweekend',
'M12_VOORKEUR_REISTIJDSTIP_Week- en weekenddal',
'Aantal Maanden zelfde abo',
'INT_REISINFORMATIE',
'Contacten_1mnd',
'INT_GK_TREINTICKETS']

random_classifier = RandomForestClassifier(n_jobs=100, max_depth=5, random_state=12345)
random_classifier.fit(X_train, y_train)
y_pred = random_classifier.predict(X_test)
prediction = [round(value) for value in y_pred]
accuracy_rf = (accuracy_score(y_test, prediction)) * 100
recall_rf = (recall_score(y_test, prediction)) * 100
precision_rf = (precision_score(y_test, prediction))*100



#SVM <---- Slower than other models
svm_cols = ['Contacten_3mnd', 'INT_LEZEN', 'PC_FIRST_STATION_AFSTAND_CAT',
       'INT_DIERENTUIN', 'BCB', 'WVR', 'M12_VOORKEUR_REISTIJDSTIP_Dalweekend',
       'BCT', 'INT_GK_TREINTICKETS', 'm12_min_m3_ritten_totaal', 'DVD',
       'REGIO_RANDSTAD ZUID', 'DVR', 'INT_VOORSTELLING', 'm1_TE_dalweekend',
       'm1_ritten_totaal', 'm2_TE_dalweekend', 'INT_NS_BELEVING',
       'Aantal Maanden zelfde abo', 'm3_TE_dalweekend', 'SOR',
       'm1_TE_spits_ochtend', 'M12_VOORKEUR_REISTIJDSTIP_Week- en weekenddal',
       'INT_FESTIVAL', 'TVR', 'OV_FIETS_ABO_KLANT', 'm3_TE_spits_ochtend',
       'ContactenTel_2mnd', 'INT_EVENEMENTEN', 'm2_TE_spits_avond',
       'M12_TE_SPITS_AVOND_TOTAAL', 'INT_REISINFORMATIE',
       'REGIO_RANDSTAD NOORD', 'Contacten_1mnd', 'm12_min_m3_TE_dalweekend',
       'INT_ACTIES']

svm_classifier=SVC(probability=True, gamma=0.0001, C=3)
svm_classifier.fit(X_train[svm_cols],y_train)
y_pred=svm_classifier.predict(X_test[svm_cols])
prediction = [round(value) for value in y_pred]
accuracy_svm = (accuracy_score(y_test, prediction))*100
recall_svm = (recall_score(y_test, prediction))*100
precision_svm = (precision_score(y_test, prediction))*100



#Neural Network <- Also slow + may not produce same results as it uses randomized states
ann_classifier=Sequential()
ann_classifier.add(Dense(neuron,activation='relu', input_dim=X_train.shape[1]))
for i in range(40):
    ann_classifier.add(Dense(8,activation='relu'))
ann_classifier.add(Dense(1,activation='sigmoid'))
ann_classifier.compile(optimizer='adagrad',loss='binary_crossentropy', metrics = ['accuracy'])
ann_classifier.fit(X_train,y_train,batch_size=8,epochs=200, verbose=False)
y_pred_proba = ann_classifier.predict(X_test)
y_pred = (y_pred_proba>.5).astype('int')
accuracy_ann = (accuracy_score(y_test, y_pred))*100
recall_ann = (recall_score(y_test, y_pred)) * 100
precision_ann = (precision_score(y_test, y_pred)) * 100