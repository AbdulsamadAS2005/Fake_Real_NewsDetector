import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

# def transform_text(text):
#     text=text.lower()
#     text=text.split()
#     cleaned_text=[]
#     for word in text:
#         word="".join(char for char in word if char not in string.punctuation)
#         if (word):
#             word="".join(char for char in word if char not in ['0','1','2','3','4','5','6','7','8','9'])
#             if (word):
#                 if word not in stop_words:
#                     cleaned_text.append(word)
#     return " ".join(cleaned_text)

import re
def fast_transform(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    return ' '.join([word for word in text.split() if word not in stop_words])

Fake_news=pd.read_csv("Fake.csv")
True_news=pd.read_csv("True.csv")

Fake_news['Label']=1
True_news['Label']=0

data=pd.concat([True_news,Fake_news],ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

data['title'] = data['title'].astype(str).apply(fast_transform)
data['text'] = data['text'].astype(str).apply(fast_transform)
data['subject'] = data['subject'].astype(str).apply(fast_transform)

data['combined'] = data['title'] + ' ' + data['subject'] + ' ' + data['text']

tfidf = TfidfVectorizer(max_features=5000)  # Limit features to reduce memory
X = tfidf.fit_transform(data['combined'])  # This is a sparse matrix (efficient)
Y=data['Label']

# for i in range(1,1000):
#     xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.25,random_state=i)

#     from sklearn.linear_model import LogisticRegression
#     lr=LogisticRegression()

#     lr.fit(xtrain,ytrain)

#     from sklearn.metrics import confusion_matrix,recall_score
#     recall=recall_score(ytest,lr.predict(xtest))*100
#     if(recall>rec):
#         rec=recall
#         random=i
    

# print(rec,random)

X_small=X
Y_Small=Y
xtrain,xtest,ytrain,ytest=train_test_split(X_small,Y_Small,test_size=0.25,random_state=42)

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score
from sklearn.svm import SVC
svc=SVC(C=10,kernel="linear")
gnb=GaussianNB()
bnb=BernoulliNB()
mnb=MultinomialNB()

# from sklearn.model_selection import GridSearchCV
# ds={
#     "kernel": ['linear'], 
#     "C": [0.1, 1, 10]

# }
# gs=GridSearchCV(SVC(),param_grid=ds,scoring='f1', verbose=1, n_jobs=-1)
# gs.fit(xtrain,ytrain)
# print(gs.best_params_)

svc.fit(xtrain,ytrain)
print(svc.score(xtrain,ytrain)*100,svc.score(xtest,ytest)*100)
print(precision_score(ytest,svc.predict(xtest)))
print(recall_score(ytest,svc.predict(xtest)))
print(f1_score(ytest,svc.predict(xtest)))
print(confusion_matrix(ytest,svc.predict(xtest)))