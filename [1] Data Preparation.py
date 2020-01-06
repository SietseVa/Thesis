'''
This script deals with all the data cleaning and data preparation of the original dataset.
Irrelevant entries (such as NaN values) and irrelevant variables (that only have 1 value) are removed.
Furthermore, each variable is encoded in the right way, and the dataset is normalized.
The scripts uses VIF's in order to remove variables that cause multi-collinearity.
The dataset is then balanced in such a way that the minority and majority class are distributed more equally.
Finally, the scripts saves a training and test sample for further use in the models.
'''



#Import Packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import random

label_encoder=LabelEncoder()

#VIF Feature removal
def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

df = pd.read_csv('Dataset Final.csv', index_col='Hashed NSR')

#Delete irrelevant columns
del df['JAARMAAND']

#Delete missing values
df.dropna(inplace=True)

#Rewrite 'Reisfrequentie' to numeric categories
mapper = {'Onbekend': 0, 'Inactief': 0, '1-2 dagen per jaar': 1, '3-5 dagen per jaar': 2, '6-12 dagen per jaar': 3, '1-3 dagen per maand': 4, '1-3 dagen per week': 5, '4 dagen per week of vaker': 6}
df.replace(mapper, inplace=True)


'''
    Recode Variables to correct data types
'''

#Recode abo vars
abos = ['OFI', 'SOR', 'AVD',
       'AVR', 'DVD', 'DVR', 'KVR', 'TVR', 'VDU', 'WVR', 'BCB', 'BCD', 'BCT',
       'BCS', 'NSF_Basis', 'WVD']
for col in abos:
    df[col] = df[col].apply(lambda x: 'No' if x == 0 else 'Yes')


#Recode INT vars
int = ['INT_BEURS', 'INT_EVENEMENTEN',
       'INT_FESTIVAL', 'INT_CONCERT', 'INT_VOORSTELLING', 'INT_KUNST_CULTUUR',
       'INT_DIERENTUIN', 'INT_SPORT_BEWEGEN', 'INT_ETEN_DRINKEN', 'INT_HOTEL',
       'INT_STEDEN', 'INT_GK_TREINTICKETS', 'INT_INTERNATIONAAL',
       'INT_REISINFORMATIE', 'INT_NS_BELEVING', 'INT_KETENINFORMATIE',
       'INT_ZAKELIJK', 'INT_ACTIES', 'INT_LEZEN']
for col in int:
    df[col] = df[col].apply(lambda x: 'No' if x == 0 else 'Yes')

#Recode Binary vars to same type (Consistency across variables)
df['Response'] = df['Response'].apply(lambda x: 'No' if x == 0 else 'Yes')
df['OV_FIETS_ABO_KLANT'] = df['OV_FIETS_ABO_KLANT'].apply(lambda x: 'No' if x == 0 else 'Yes')

#Express numeric vars as numeric
df[['Age', 'RELATIEDUUR_CAT', 'PC_FIRST_STATION_AFSTAND_CAT']] = df[['Age', 'RELATIEDUUR_CAT', 'PC_FIRST_STATION_AFSTAND_CAT']].apply(pd.to_numeric, downcast='integer')
df[['AANTAL_ABOOS_NU_GELDIG']] = df[['AANTAL_ABOOS_NU_GELDIG']].apply(pd.to_numeric, downcast='integer')

# Remove non relevant data values
df = df[df.REGIO.isin(['NOORDOOST', 'ZUID', 'RANDSTAD ZUID', 'RANDSTAD NOORD'])]
df = df[df.SEX.isin(['M', 'V'])]
df = df[df.KLASSE.isin(['2', '1'])]

# Remove variables that have only 1 value
categorical_var=[i for i in df.columns if df[i].dtypes=='object']
for z in categorical_var:
    if len(df[z].unique()) == 1:
        del df[z]


## Label encode variables

#Encode binary vars
for x in [i for i in df.columns if len(df[i].unique())==2]:
    df[x]= label_encoder.fit_transform(df[x])

#Encode categorical vars
old = df.columns
df= pd.get_dummies(df, columns= [i for i in df.columns if df[i].dtypes=='object'],drop_first=True)


#Split into X and y
X = df.drop('Response', axis=1)
y=df['Response']

#Remove multicollinear variables
X = calculate_vif_(X)



#Splt data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Scale data
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_train=pd.DataFrame(X_train,columns=X.columns)
X_test=sc.transform(X_test)

X_train=pd.DataFrame(X_train,columns=X.columns)
y_train=pd.DataFrame(y_train, columns=['Response'])


X_test=pd.DataFrame(X_test, columns=X.columns)
y_test=pd.DataFrame(y_test, columns=['Response'])
X_train['Response'] = y_train.values
X_test['Response'] = y_test.values


#Undersample the train data (NOT THE TEST DATA)
train_class_0 = X_train[X_train.Response == 0]
train_class_1 = X_train[X_train.Response == 1]
s = np.floor(len(train_class_1)*(7/5)) #This ensures a distribution of approximately 60:40
s = s.astype(np.int64)
train_class_0_under = train_class_0.sample(s)
X_train = pd.concat([train_class_0_under, train_class_1], axis=0)
random.shuffle(X_train)
random.shuffle(X_test)


#Save data to new csv's for further use
X_train.to_csv('Train Data.csv')
X_test.to_csv('Test Data.csv')