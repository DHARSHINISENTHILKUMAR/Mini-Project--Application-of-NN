# Mini-Project--Application-of-NN


(Expected the following details in the report )
## Project Title:
Stock market prediction
## Project Description 
We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.
## Algorithm:

    1.import the necessary pakages.
    2.install the csv file
    3.using the for loop and predict the output
    4.plot the graph
    5.analyze the regression bar plot
##Google Colab Link:
 https://colab.research.google.com/drive/1rknMNlbLphgS6ObhFSfGUWMfB-S_kPIE?usp=sharing
## Program:
 ```
   import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
```
###install the csv file
```
df = pd.read_csv('/content/Tesla.csv')
df.head()
df.shape
df.describe()
df.info()
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
df.head()
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1)
df.isnull().sum()
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.distplot(df[col])
plt.show()
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.boxplot(df[col])
plt.show()
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
features = df[['open-close', 'low-high']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
models = [LogisticRegression(), SVC(
kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

print(f'{models[i]} : ')
print('Training Accuracy : ', metrics.roc_auc_score(
	Y_train, models[i].predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(
	Y_valid, models[i].predict_proba(X_valid)[:,1]))
print()
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()
```
## Output:
![image](https://user-images.githubusercontent.com/113699377/205443461-5218992d-41a9-4fbe-8db9-8b6b626b6b6a.png)
![image](https://user-images.githubusercontent.com/113699377/205443479-0768f81d-8b72-4c77-bf86-514665ea7f13.png)
![image](https://user-images.githubusercontent.com/113699377/205443497-422634e6-02d6-4f11-9ac6-8da637c23835.png)
![image](https://user-images.githubusercontent.com/113699377/205443506-501a8dab-040f-449e-83e6-96ffb304ae56.png)
![image](https://user-images.githubusercontent.com/113699377/205443527-1b5acf16-62db-403f-a20c-491a86d9d0df.png)
![image](https://user-images.githubusercontent.com/113699377/205443534-99e99725-64e2-45a6-b8c6-86a582d3aca3.png)
![image](https://user-images.githubusercontent.com/113699377/205443543-d86fb2d9-d17c-47df-b384-ffc76b24bbdc.png)
![image](https://user-images.githubusercontent.com/113699377/205443557-815cea2c-dd1a-498a-b3f4-656d92befd0c.png)
![image](https://user-images.githubusercontent.com/113699377/205443570-d405bd33-6dd1-429d-9b7e-d340e5030867.png)
![image](https://user-images.githubusercontent.com/113699377/205443585-3cb21117-300d-4eab-b99e-ba6133afd2ee.png)
![image](https://user-images.githubusercontent.com/113699377/205443594-69a2c2a2-d2a9-4270-8f30-76dbb382a630.png)
![image](https://user-images.githubusercontent.com/113699377/205443603-b8aa617a-fabe-4d9e-b1bd-a8619582ac74.png)
![image](https://user-images.githubusercontent.com/113699377/205443611-209d57eb-cf4c-475d-a9dc-bd340f865dd1.png)
![image](https://user-images.githubusercontent.com/113699377/205443625-199455ac-b5bb-4d24-b8b7-2096ef9b582d.png)
![image](https://user-images.githubusercontent.com/113699377/205443633-0961c810-4116-409a-9fa4-d17d43b01f5f.png)

## Advantage :
```
Python is the most popular programming language in finance. Because it is an object-oriented and open-source language, it is used by many large corporations, including Google, for a variety of projects. Python can be used to import financial data such as stock quotes using the Pandas framework.
```
## Result:
Thus, stock market prediction is implemented successfully.
