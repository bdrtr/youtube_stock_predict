# %%
import nltk
from simplemma import text_lemmatizer
import re
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import numpy as np 
import pandas as pd 
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# %%
PATH='/home/bedir/Documents/vsCode3/python/python3/youutbeapi/stopwords.txt'
"""
stop_words = None
with open(PATH, "r") as stop_file:
    stop_words = set(stop_file.read().splitlines())

# %%
def clean_text(text):
    text = text.replace("Â", "a")
    text = text.replace("â", "a")
    text = text.replace("î", "i")
    text = text.replace("Î", "ı")
    text = text.replace("İ", "i")
    text = text.replace("I", "ı")
    text = text.replace(u"\u00A0", " ")
    text = text.replace("|", " ")

    text = re.sub(r"@[A-Za-z0-9]+", " ", text)
    text = re.sub(r"(.)\1+", r"\1\1", text)
    text = re.sub(r"https?:\/\/\S+", " ", text)
    text = re.sub(r"http?:\/\/\S+", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"#(\w+)", " ", text)
    text = re.sub(r"^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^A-Za-zâîığüşöçİĞÜŞÖÇ]+", " ", text)
    text = re.sub(r"((https://[^\s]+))", " ", text)

    text = " ".join(text.lower().strip().split())
    text = text_lemmatizer(text, lang="tr")

    return " ".join([word for word in text if word not in stop_words])




    

# %%
sentence = 'Ben bu hisse adamı batırır oldukça kötü sakın almayın aldırmayın çok pişman olursunuz şimdiden söylüyorum YTD'
new_sentence = clean_text(sentence)
#tokens = nltk.word_tokenize(new_sentence)

new_sentence

# %%


# %%
def etiketle_yorum(yorum):
    if 'Olumlu' in yorum:
        return 1
    elif 'Olumsuz' in yorum:
        return -1
    else:
        return 0

etiketle_yorum(yorumlar)
yorumlar['etiketli'] = yorumlar['Durum'].apply(etiketle_yorum)
yorumlar

"""
class Cleaning_Model:
    def __init__(self,path_stopword):
        self.path_stopword = path_stopword
        self.stopwords = None
        self.get_stop_words()


    def get_stop_words(self):
        with open(self.path_stopword, "r") as stop_file:
            self.stopwords = set(stop_file.read().splitlines())

    def clean_text(self,text):
        text = text.replace("Â", "a")
        text = text.replace("â", "a")
        text = text.replace("î", "i")
        text = text.replace("Î", "ı")
        text = text.replace("İ", "i")
        text = text.replace("I", "ı")
        text = text.replace(u"\u00A0", " ")
        text = text.replace("|", " ")

        text = re.sub(r"@[A-Za-z0-9]+", " ", text)
        text = re.sub(r"(.)\1+", r"\1\1", text)
        text = re.sub(r"https?:\/\/\S+", " ", text)
        text = re.sub(r"http?:\/\/\S+", " ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"#(\w+)", " ", text)
        text = re.sub(r"^\x00-\x7F]+", " ", text)
        text = re.sub(r"[^A-Za-zâîığüşöçİĞÜŞÖÇ]+", " ", text)
        text = re.sub(r"((https://[^\s]+))", " ", text)

        text = " ".join(text.lower().strip().split())
        text = text_lemmatizer(text, lang="tr")

        return " ".join([word for word in text if word not in self.stopwords])

    def etiketle_yorum_tur1(self,yorum):
        if 'Olumlu' in yorum:
            return 1
        elif 'Olumsuz' in yorum:
            return -1
        else:
            return 0
        
    def clean(self,row):
        if isinstance(row['Görüş'], str):
            satirlar = row['Görüş'].split('\n')
            for i in range(len(satirlar)):
                satirlar[i] = self.clean_text(satirlar[i])

            return " ".join(satirlar).lower().translate(str.maketrans("", "", string.punctuation))
        else:
            return ""




# %%
"""
def clean(row):
    if isinstance(row["Görüş"], str):
        satirlar = row["Görüş"].split('\n')
        for i in range(len(satirlar)):
          satirlar[i] = clean_text(satirlar[i])

        return " ".join(satirlar).lower().translate(str.maketrans("", "", string.punctuation))
    else:
        return ""
"""
yorumlar = pd.read_csv('/home/bedir/Documents/vsCode3/python/python3/youutbeapi/magaza_yorumlari_duygu_analizi.csv',encoding='utf-16')

CleanModel = Cleaning_Model(PATH)

yorumlar['clean_comment'] = yorumlar.apply(lambda row: CleanModel.clean(row), axis=1)
yorumlar['etiketli'] = CleanModel.etiketle_yorum_tur1(yorumlar)

# %%

X = yorumlar.clean_comment.to_numpy()
y = yorumlar.etiketli.to_numpy()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
"""
X_test = vectorizer.transform(X_test)
# %%
model = MultinomialNB()
model.fit(X_train, y_train)

# %%
model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, y_train)
# %%
clf_SVM = SVC(kernel="linear")
clf_SVM.fit(X_train, y_train)

clf_SVM.score(X_test, y_test)

# %%

clf_SVM2 = SVC(kernel="rbf")
clf_SVM2.fit(X_train, y_train)

clf_SVM2.score(X_test, y_test)

# %%
clf_best = SVC(kernel='rbf',C=100,gamma=0.001)
clf_best.fit(X_train, y_train)

# %%
model.score(X_train, y_train)

# %%
model.score(X_test, y_test)

# %%
predictions_train = model.predict(X_train)
print("Train F1:", f1_score(y_train, predictions_train,average='weighted'))

predictions_test = model.predict(X_test)
print("Test F1:", f1_score(y_test, predictions_test,average='weighted'))


print("DT train accuracy:", model_DT.score(X_train, y_train))
print("DT test accuracy:", model_DT.score(X_test, y_test))

predictions_train = model_DT.predict(X_train)
print("DT Train F1:", f1_score(y_train, predictions_train,average='weighted'))

predictions_test = model_DT.predict(X_test)
print("DT Test F1:", f1_score(y_test, predictions_test,average='weighted'))

# %%
vec_sen = vectorizer.transform([new_sentence])
vec_sen

# %%
sentence = 'aslında iyi bir yorum yapmak istiyodum fakat bu hissenin belirsizliği beni korkutuyo'
#tokens = nltk.word_tokenize(new_sentence)

new_sentence = clean_text(sentence)


vec_sen = vectorizer.transform([new_sentence])

new_sentence

# %%
print("bayes ",model.predict(vec_sen))
print("d_tree ",model_DT.predict(vec_sen))
print("svm ",clf_SVM.predict(vec_sen))
print("svm_rbf ",clf_SVM2.predict(vec_sen))

# %%


"""

