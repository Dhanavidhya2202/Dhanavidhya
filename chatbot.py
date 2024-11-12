import io
import random
import string
import json
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

# Load JSON data
with open('chatbot.json', 'r', encoding='utf8') as fin:
    data = json.load(fin)

# Extract data from JSON
knowledge_base = data['knowledge_base']
greeting_inputs = tuple(data['greetings']['inputs'])
greeting_responses = data['greetings']['responses']
fallback_response = data['fallback_response']

# Tokenization
sent_tokens = nltk.sent_tokenize(" ".join(knowledge_base))
word_tokens = nltk.word_tokenize(" ".join(knowledge_base))

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting Matching
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)

# Generate Response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = fallback_response
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

# Main Loop
flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while flag:
    user_response = input().lower()
    if user_response != 'bye':
        if user_response in ('thanks', 'thank you'):
            flag = False
            print("ROBO: You are welcome..")
        elif greeting(user_response) is not None:
            print("ROBO: " + greeting(user_response))
        else:
            print("ROBO: " + response(user_response))
    else:
        flag = False
        print("ROBO: Bye! Take care..")
