import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

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

Fake_news=pd.read_csv("Fake.csv")
True_news=pd.read_csv("True.csv")

Fake_news['Label']=1
True_news['Label']=0

data=pd.concat([True_news,Fake_news],ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

# data['title']=data['title'].apply(transform_text)
# data['text']=data['text'].apply(transform_text)
# data['subject']=data['subject'].apply(transform_text)

sb.scatterplot(x="title",y="Label",data=data)
plt.show()