import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv('input.csv')
df = pd.DataFrame(data,columns= ['kasus_covid', 'rawat_inap', 'isolasi_mandiri', 'margin_error', 'fatal_rate'])

X = df[['kasus_covid', 'rawat_inap', 'isolasi_mandiri', 'margin_error']]
y = df['fatal_rate']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("=========================================================================================")
print(data)
print("=========================================================================================")

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print(classification_report(y_test, y_pred))
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.title('Confusion Matrix pada Algoritma Logistic Regression\n dalam kasus Covid-19 DKI Jakarta per-Januari 2021')
plt.show()