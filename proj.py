# reading data

import pandas as pd
import nltk
from sklearn import metrics
import os
import joblib

datasur = pd.read_csv(r'C:\\Users\\gessn\\OneDrive\\Pulpit\\Nauka\\python\\projekt\\labeledtext.csv')
dataright = datasur[datasur['label']=='right']
dataleft = datasur[datasur['label']=='left']
dataright=dataright.head(10)
dataleft=dataleft.head(10)
data=dataright._append(dataleft)
data.columns = ['title', 'text', 'label']
data=data.dropna()


import nltk

text = list(data['text'])

# preprocessing 

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    r = re.sub('[^a-zA-Z]', ' ', text[i])

    r = r.lower()

    r = r.split()

    r = [word for word in r if word not in stopwords.words('english')]

    r = ' '.join(r) #nie do końca rozumiem, dlaczego to jest potrzebne, a nie może być po prostu lista słów

    corpus.append(r)


data["text"]= corpus

data.head()
#print (data.head())
# stworzenie dwóch setów

X = data['text']

y = data['label']


# rozdział datasetu

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)




print('Training Data :', X_train.shape)

print('Testing Data : ', X_test.shape)

# BOW

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train) 
print (X_train_cv)

X_train_cv.shape

# trenowanie

from sklearn.linear_model import LogisticRegression
# jakie argumenty i wartości do nich przypisane będą odpowiednie?
lr = LogisticRegression(solver='liblinear', random_state=0)

lr=lr.fit(X_train_cv, y_train)
print (lr)

# transform X_test using CV

X_test_cv = cv.transform(X_test)


# predykcje

predictions = lr.predict(X_test_cv)
predictionsy = lr.predict_proba(X_test_cv)
print(predictions)
print (predictionsy)
#jak zwrócić wartości liczbowe przewidywań?

# confusion matrix

df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['left','right'], columns=['left','right'])
print (df)


#obliczyć accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,predictions)
print (score)

# Create the 'trained_model' folder if it doesn't exist
if not os.path.exists('trained_model'):
        os.makedirs('trained_model')

# Save the trained pipeline to the 'trained_model' folder
pipeline_filename = os.path.join('trained_model', 'lr_predictions.pkl')
joblib.dump(lr, pipeline_filename)
print(f"Trained pipeline saved as {pipeline_filename}")
