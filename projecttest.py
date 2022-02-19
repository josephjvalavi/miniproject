import nltk 
import pickle
import streamlit as st
from  nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


def process_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().replace('\n',' ').replace('\r','').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]','',text)

    tokens=word_tokenize(text)
    filtered_sentence=[]
    for f in tokens:
      if f not in stop_words:
        filtered_sentence.append(f) 
    text = " ".join(filtered_sentence)
    return(text)
#s=process_text("")
tfidf = pickle.load(open(r'C:\Users\ASUS\Desktop\project models\vectorizer.pkl','rb'))
model = pickle.load(open(r'C:\Users\ASUS\Desktop\project models\model.pkl','rb'))

st.title("news classifier")
#print(vector_input)

input_text = st.text_area("Enter the message")
if st.button('Predict'):
    s=process_text(input_text)
    vector_input=tfidf.transform([s])
    
    result = model.predict(vector_input)[0]
    if result==[0]:
      st.header("buisness")
    elif result==[1]:
      st.header("entertainment")
    elif result==[2]:
      st.header("politics")
    elif result==[3]:
      st.header("sports")
    else:
      st.header("technology")
    