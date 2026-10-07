import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

Dataset = pd.read_csv('Data.csv')

x = Dataset.iloc[:,:-1].values

y = Dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])


from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

x[:,0] = labelencoder_x.fit_transform(x[:,0])

labelencoder_y= LabelEncoder()

y = labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test  = train_test_split(x, y,test_size=0.2, train_size=0.8, random_state=0)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)



print(x_test.shape)
print(y_test.shape)


print(Dataset.columns)
print(Dataset.head())