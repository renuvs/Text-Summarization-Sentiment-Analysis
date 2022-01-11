# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:46:16 2021

@author: Admin
"""

import streamlit as st
import PyPDF2
import re

import pandas as pd
from afinn import Afinn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer



def getData(pdfreader):
    no_pages = pdfreader.getNumPages()
    corpus = ''
    for i in range(0, no_pages):
        page = pdfreader.getPage(i)
        corpus += page.extractText()
        
    return corpus


def cleanData(textCorpus):
    textCorpus = textCorpus.replace("'s",'') # replaces apostrophe s
    textCorpus = textCorpus.replace('\n','') # replaces newline character
    textCorpus = re.sub(r'\([^()]*\)','',textCorpus) # removes text inside brackets including brackets
    textCorpus = re.sub(r'(http|https|www)\S+', '', textCorpus) # replaces www.digitalsherpa.com,http://www.articlesbase.com/technology
    textCorpus = re.sub(r'\<.+\>','',textCorpus) # replaces <link rel=ﬂcanonicalﬂ href=ﬂﬂ />
    textCorpus = re.sub(r'\s+',' ',textCorpus) # replaces more than 2 spaces with 1 space
    textCorpus = textCorpus.lower() # converts the text to lower
    
    return textCorpus

def getTokenizeSentences(stringCorpus):
    sentences = sent_tokenize(stringCorpus)
    return sentences

def stemming_tokenizer(str_input):
    ps = PorterStemmer()
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [ps.stem(word) for word in words]
    return words

def getSummary(df_tfidf_topscore_data):
    summary_tfidf = ''
    for i in range(len(df_tfidf_topscore_data)):
        summary_tfidf += df_tfidf_topscore_data['sentence'][i]
    return summary_tfidf

#sentiment analysis
def summarysentiment(textsummary):
    list_assert_words = []
    afn = Afinn()
    scores = afn.score(textsummary)
    listAssertWords = afn.find_all(textsummary)
    #converting list to string
    strAssertWords = ', '
    strAssertWords = strAssertWords.join(listAssertWords)
    
    sentiment = ''
    if scores > 0:
        sentiment = 'Positive'
    elif scores < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return (sentiment)


st.title('Text Summarization')

uploaded_file = st.file_uploader('Choose your .pdf file', type='pdf')

if uploaded_file is not None:
    #st.write('file uploaded successfully')
    #attributes/methods
    #st.write(dir(uploaded_file))
        
    if uploaded_file.type == 'application/pdf':
        
        #file_details = {'File Name':uploaded_file.name,'File Type':uploaded_file.type,'Size':uploaded_file.size}   
        #st.write(file_details)
        #st.info(uploaded_file.name)
    
        #creating pdf filereader object
        pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
        textPdf = getData(pdf_reader)
        #length of text from pdf = 21932
        #st.write(len(textPdf))
        #st.write(textPdf)
        
        #cleaning the text in corpus
        corpus = cleanData(textPdf)
        #length of corpus after cleaning = 21309
        #st.write(len(corpus))
        
        sentences = getTokenizeSentences(corpus)
        #after sentence tokenization = 169 sentences
        #st.write(len(sentences))
        
        #removes stopwords, stems the words and gives tfidf score of words in each sentence
        vectorizer = TfidfVectorizer(analyzer='word', stop_words = 'english', tokenizer = stemming_tokenizer, ngram_range=(2,2))
        tfidf = vectorizer.fit_transform(sentences)
        #st.write(tfidf.toarray())
        
        #creating table of tfidf scores
        df = pd.DataFrame(tfidf.toarray(), columns= vectorizer.get_feature_names())
        #adding sentences in dataframe
        df['sentence'] = sentences
        df['sentence'] = df['sentence'].str.capitalize() 
        # adding tfidf scores row-wise 
        df['sentence score'] = df.sum(axis=1)
        #sorting on the basis of sentence score
        df.sort_values(by='sentence score', ascending=False, inplace=True)
        #st.write(df[['sentence','sentence score']])
        
        summary_length  = 4
        df_tfidf_topscore = df[['sentence','sentence score']].head(summary_length)
        df_tfidf_topscore = df_tfidf_topscore.reset_index(drop=True)
        
        summary = getSummary(df_tfidf_topscore)
        summary = summary.replace('.','. ')
        #st.write(summary)
        st.subheader("Summary:")
        st.success(summary)
        
        #sentiment analysis
        sentiment_summary = summarysentiment(summary)        
        st.subheader("Summary Sentiment:")
        st.info(sentiment_summary)       
        
    else:
        st.error('Only Pdf Uploads are Allowed')
        
        
        



