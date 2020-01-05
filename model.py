import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

np.random.seed(500)

Corpus = pd.read_csv(r"./data.csv", encoding='latin-1')

## Pre-processing

# Remove blank rows
Corpus['text'].dropna(inplace=True)

# Lower case
Corpus['text'] = [entry.lower() for entry in Corpus['text']]

# Tokenise
Corpus['text'] = [word_tokenize(entry) for entry in Corpus['text']]

# Prune useless tokens and stem
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Corpus['text']):
    final_words = []
    stemmed_word = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_final = stemmed_word.lemmatize(word,tag_map[tag[0]])
            final_words.append(word_final)
    Corpus.loc[index,'text_final'] = str(final_words)

# Prepare train and test data sets

train_x, test_x, train_y, test_y = model_selection.train_test_split(Corpus['text_final'], Corpus['label'], test_size=0.3)

# Encode

Encoder = LabelEncoder()
train_y = Encoder.fit_transform(train_y)
test_y = Encoder.fit_transform(test_y)

# Using TF-IDF to vectorise text data

tfidf_vect = TfidfVectorizer(max_features=500)
tfidf_vect.fit(Corpus['text_final'])
train_x_tfidf = tfidf_vect.transform(train_x)
test_x_tfidf = tfidf_vect.transform(test_x)
# # Output the learned vocabulary
# print("\n\nLEARNED VOCAB:")
# print(tfidf_vect.vocabulary_)
# # Output the vectorised data
# print("\n\nVECTORS:")
# print(train_x_tfidf)

## Classification using SVM

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_x_tfidf, train_y)

predictions = SVM.predict(test_x_tfidf)

print("SVM accuracy score: ", accuracy_score(predictions, test_y))