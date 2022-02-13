#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dan Haines
"""
# ----------------------------------------------------------------------------
# This document is meant to be a model for sentiment analysis
# ----------------------------------------------------------------------------

# import relevant libraries
import pandas, os, nltk

# download 'punkt', 'wordnet', and 'averaged_perceptron_tagger' from nltk
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# change the directory so that it points to the .csv storage location
os.chdir("Documents/Python/ST542")

# import the .csv file that contains text-based description data
data = pandas.read_csv("vaccine microchipyoutube_videos.csv")

# point to the videoID and description data columns                       
data_trim = data[['videoID', 'description']]

# create a function that will set 'NA' type values to blank strings and run 
# the word_tokenize() function.
def custom_tokenize(text):
    if pandas.isnull(text):
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.tokenize.word_tokenize(text)

# apply the 'custom_tokenize' function
data_trim['tokenized'] = data_trim.description.apply(custom_tokenize)

# add a part-of-speech tag to the tokenized data
data_trim['pos'] = data_trim.tokenized.apply(nltk.tag.pos_tag)

# lemmatize the data 
def lemmatize_sentence(tokens):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in tokens:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

# run the lemmatize_sentence() function




