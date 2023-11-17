import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def text_transformer(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)


  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text :
    y.append(ps.stem(i))
  return " ".join(y)

tfidf = pickle.load(open('vectorizer2.pkl','rb'))
model = pickle.load(open('model2.pkl','rb'))

st.title("Email/Sms-spam-classifier")

input_sms = st.text_input("Enter sms text")

if st.button('Predict'):
  #preprocess
  transformed_sms = text_transformer(input_sms)
  #vectorize
  vector_input = tfidf.transform([transformed_sms])
  #predict
  result =model.predict(vector_input )[0]
  #display
  if result == 1:
      st.header("spam")
  else:
      st.header("not_spam")
