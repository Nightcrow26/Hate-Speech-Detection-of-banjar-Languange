# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import f1_score

df = pd.read_csv('banjar.csv')
df.columns = ['Text', 'Label']

# remove missing values
df = df.dropna()

# encode target label
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# establish input and output
X = list(df['Text'])
y = list(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
tfidf_train = tfidf.fit_transform(X_train)
#Fitting and transforming input data
tfidf_train = tfidf.transform(X_train)
tfidf_test = tfidf.transform(X_test)

svc = LinearSVC()
svc.fit(tfidf_train, y_train)#TFIDF

y_pred = svc.predict(tfidf_test)

# find f-1 score
score = f1_score(y_test, y_pred, average='micro')
print('F-1 score : {}'.format(np.round(score,4)))

print(metrics.classification_report(y_test, y_pred))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

test_text=tfidf.transform(["tambuk siapa nyawa ngini"])

print("\n")
print("Prediction:")
print(svc.predict(test_text))
