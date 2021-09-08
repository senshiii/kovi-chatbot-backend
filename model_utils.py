import pickle
import spacy
import re
import json
import random
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

class DTCModel:

  def __init__(self):
    # Loading Intent Response Mapping
    intent_res = pickle.load(open('model/intent_responses.pickle', 'rb'))
    
    # Loading Decision Tree Classifier
    classifier = pickle.load(open('model/classifier.pickle', 'rb'))
    
    # Loading Label Mapping
    label_mapping = pickle.load(open('model/label_mapping.pickle', 'rb'))

    # Loading Vectorizer
    vectorizer = pickle.load(open('model/vectorizer.pickle', 'rb'))

    # Loading Entities for Custom NER
    entities = json.load(open('data/entities.json', encoding='utf8'))['patterns']
    
    # Adding Entities to NER Pipeline
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(entities)

    # print('Intent Response ', intent_res, end='\n')
    # print('Vectorizer ', vectorizer, end='\n')
    # print('Classifier ', classifier, end='\n')
    # print('Label Mapping ', label_mapping, end='\n')
    
    self.intent_res = intent_res
    self.classifier = classifier
    self.label_mapping = label_mapping
    self.vectorizer = vectorizer
    self.entities = entities


  def preprocess_text(self, text):
    text = text.lower()
    text = re.sub(r'coronavirus|covid 19|covid', 'covid-19', text)
    text = re.sub(r'\W+', ' ', text)
    text = nlp(text)
    words = [token.lemma_ for token in text if token.is_stop is False]
    return ' '.join(words)

  def classify_query(self, chat_message):
    chat_message = self.preprocess_text(chat_message)
    test_data = [chat_message]
    text_pred = self.vectorizer.transform(test_data)
    label_index = self.classifier.predict(text_pred)[0]
    label = self.label_mapping[label_index]
    print('Label Index = {} Label = {}'.format(label_index, label), end='\n')
    
    res = None
    no_of_responses = len(self.intent_res)
    if no_of_responses > 1:
      res = random.choice(self.intent_res[label])
    elif no_of_responses == 1:
      res = self.intent_res[label][0]

    return (label, res)

  def recognize_entites(self, chat_message):
    doc = nlp(chat_message)
    ents = displacy.parse_ents(doc)['ents']
    self.ent_dict = dict()
    for e in ents:
      if e['label'] not in self.ent_dict.keys():
        self.ent_dict[e['label']] = chat_message[e['start']:e['end']]

    print('Ents: ', ents, end='\n')
    print('Entity Dict: ', self.ent_dict)

  # Get Priority Keyword from entities
  def extract_keyword(self):
    if self.ent_dict is None:
      self.recognize_entites()
    keys = self.ent_dict.keys()
    if 'VAX' in keys:
      return self.ent_dict['VAX']
    elif 'LOCKDOWN' in keys:
      return self.ent_dict['LOCKDOWN']
    elif 'COVID' in keys:
      return self.ent_dict['COVID']
    else:
      return "Covid-19"

  # Get Location
  def get_location(self):
    if self.ent_dict is None:
      self.recognize_entites()
    if "GPE" in self.ent_dict.keys():
      return self.ent_dict['GPE']
    else:
      return "World"
