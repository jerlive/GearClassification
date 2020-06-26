# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if "Processed Data" in dirname:
            print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
df=pd.DataFrame()
for no in range(10,29):
    temp=pd.read_csv("/kaggle/input/cartripsdatamining/TripData/Processed Data/fileID"+str(no)+"_ProcessedTripData.csv",header= None)
    df=df.append(temp)

# %% [code]
df.head()

# %% [markdown]
# Data Labelling and Identification

# %% [code]
df.columns=['Time','Vehicle Speed','SHIFT','Engine Load','Total Acceleration','Engine RPM','Pitch','Lateral Acceleration','Passenger Count','Car Load','AC Status','Window Opening','Radio Volume','Rain Intensity','Visibility','Driver Wellbeing','Driver Rush']

# %% [code]
df.head()

# %% [code]
df.corr(method='pearson')

# %% [markdown]
# Deleting Unncessary Features

# %% [code]
df=df.drop(['Time','Engine Load','Total Acceleration','Pitch','Lateral Acceleration','Passenger Count','Car Load','AC Status','Window Opening','Radio Volume','Rain Intensity','Visibility','Driver Wellbeing','Driver Rush'],axis=1) 

# %% [markdown]
# Deleting Rows where the Gear is in Neutral Position

# %% [code]
newdf = df[df.SHIFT != 0]
newdf = newdf.reset_index(drop=True)
df=newdf

# %% [code]
df.describe()

# %% [markdown]
# Scatter Plot Analysis

# %% [code]
import matplotlib.pyplot as plt
count=1
x="Vehicle Speed"
y="Engine RPM"
colors=['red','green','blue','brown','black','yellow','orange']

for i in range(1,6):
    plt.scatter(df[x][df.SHIFT==i],df[y][df.SHIFT==i],c=colors[i],label=i,alpha=0.8)


plt.gca().update(dict(title='SCATTER', xlabel=x, ylabel=y,))
plt.legend(title='GEAR')
plt.show()

# %% [markdown]
# Transfering the label feature towards the end.

# %% [code]
temp=df.SHIFT
df=df.drop("SHIFT",axis=1)
df.insert(len(df.columns),"SHIFT",temp)
df

# %% [markdown]
# Separating data from the labels.

# %% [code]
X = df[df.columns[:-1]]
y = df.SHIFT

# %% [markdown]
# Splitting the data to test and train data

# %% [code]
x_train, x_test, y_train, y_test=train_test_split(X,                                                            y, train_size=1093902, test_size=364634, random_state=0)

# %% [markdown]
# Finding all models to determine optimum value for K

# %% [code]
i=1
n_neighbors=[]
while (i<40):
    n_neighbors.append(i)
    i+=2
print(n_neighbors)
modellist=[]

for j in n_neighbors:
    model = KNeighborsClassifier(n_neighbors=j)
    model.fit(x_train, y_train)
    modellist.append(model)
scorelist=[]
knnlist = modellist
for k in range(len(n_neighbors)):
    scorelist.append(knnlist[k].score(x_test, y_test))
kdf=pd.DataFrame(scorelist,n_neighbors)
kdf.plot.line()

# %% [markdown]
# Maximum accuracy is at k=1 with Accuracy at 99.9989%

# %% [code]
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
accuracy=model.score(x_test, y_test)
print("Model Accuracy : "+str(accuracy*100)+" %")

# %% [markdown]
# Interactive Console for Prediction

# %% [code]
speed=input("Enter Current Speed : ")
speed=float(speed)/1.609
rpm=input("Enter Current RPM : ")
test=pd.DataFrame([speed,rpm])
predictions=model.predict(test.values.reshape(1, -1))
print("Predicted Gear State : "+str(int(predictions)))

# %% [code]
df.corr(method='pearson')

# %% [code]
