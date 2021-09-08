import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle
import spacy
import re

nlp = spacy.load('en_core_web_sm')
def preprocess_text(text):
  text = text.lower()
  text = re.sub(r'coronavirus|covid 19|covid', 'covid-19', text)
  text = re.sub(r'\W+', ' ', text)
  text = nlp(text)
  words = [token.lemma_ for token in text if token.is_stop is False]
  return ' '.join(words)

intents = json.load(open('data/intents.json'))['intents']

index = 0
responses = dict()
data = pd.DataFrame(columns=['text', 'intent'])
for intent in intents:
  tag = intent['tag']
  if tag not in responses.keys():
    responses[tag] = intent['responses']
  
  for pattern in intent['patterns']:
    data.loc[index] = [pattern, tag]
    index = index + 1

data['text'] = data['text'].map(preprocess_text)

x = data.values[:, 0]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)
x = x.toarray()

y = data.values[:, -1]
le = LabelEncoder()
le.fit(y)

label_mapping = dict(zip(le.transform(le.classes_), le.classes_))

y = le.transform(y)

classifier = DecisionTreeClassifier()

classifier.fit(x, y)

with open('model/classifier.pickle', 'wb') as f:
  pickle.dump(classifier, f)

with open('model/label_mapping.pickle', 'wb') as f:
  pickle.dump(label_mapping, f)

with open('model/vectorizer.pickle', 'wb') as f:
  pickle.dump(vectorizer, f)

with open('model/intent_responses.pickle', 'wb') as f:
  pickle.dump(responses, f)










