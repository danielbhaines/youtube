Youtube Classification with Zero-Shot Analysis
================

- [Introduction](#introduction)
- [Required Packages](#required-packages)
- [Importing Data from YoutubeAPI](#importing-data-from-youtubeapi)
  - [Make_CSV() Function](#make\_csv-function)
  - [Get_CC() Function](#get\_cc-function)
  - [Get_LongDesc() Function](#get\_longdesc-function) 
- [Fitting the Model](#fitting-the-model)
  - [ZS_Classify() Function](#zs\_classify-function)
- [Cleaning the Video Data](#cleaning-the-video-data)
- [Frequency and Sentiment Analysis](#frequency-and-sentiment-analysis)

# Introduction

The growth of video platforms such as YouTube has created an outlet for educators and content creators to express their unfiltered views to millions of users at the click of a button. In recent years, this growth has also given rise to conspiracy theorists, many of whom have developed significant followings. Without any available filters to determine the quality of a particular video, the onus of deciphering fact from fiction is put on the end-user. Out of this pitfall arises the question: can statistical methods be used to automatically rank the videos from a particular search based on some measurable quality?

The motivation for this project is a continuation of a studyA from fall 2020 by Dr. John Muth, Jimmy Hickey, Russell Sui, Yiming Wang, and David Elsheimer. In this work, the consultants attempted to derive the quality of YouTube videos based on the video’s comments. This project seeks to expand that project using natural language processing (NLP) to analyze the videos’ keywords, descriptions, and closed caption transcripts. To do this, we will posit a machine learning model to categorize the videos from a particular search.

A machine learning approach would utilize one of two basic frameworks: supervised learning or unsupervised learning. In supervised learning, a known truth would be used to categorize our data in order to create a training set. When comparing reality to conspiracy theory, this sounds like a simple task. However, truth can often be subjective, and our own inherent biases could influence the model’s classification. On the other hand, unsupervised learning uses no such class labels and instead infers natural structures that exist within the data set. In this respect, unsupervised learning is more tailored to this project’s goals.
The particular unsupervised method we will explore is zero-shot learningB. Zero-shot learning performs categorization without the need for a training set or class definitions. Instead, the user states a set of candidate labels, which the model compares to the input vector of text. The model then associates a probability with each label, based on the degree to which the input text linguistically entails the candidate labels. The label corresponding to the highest predicted probability of entailment is the predicted class.

Our end goal is to create a model that will rank the informational quality of YouTube videos from a particular search. A particular drawback of this project is that it may prove challenging to generalize to less divisive searches. Topics such as ‘5G’ or ‘Covid-19’ have ample conspiracy theories surrounding them, many of which are characterized by unique buzzwords or phrases. As a result, machine learning methods based on NLP may produce very distinct groupings, for example, those based on science and those based on conspiracy theory. A future direction for this project might be to optimize this model to rank quality for less polarized topics

# Required Packages
```python
import numpy as np
import os
import csv
import json
import requests
import re
import pandas
import nltk
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
```

# Importing Data from YoutubeAPI

## Make_CSV() function
```python
make_csv(my_keyword, youtube_api_key)
```

The `Make_CSV` function pulls data from the YoutubeAPI and outputs it to a 
CSV file. It takes on two arguments:

  - `my_keyword`: Must be passed as a string. This is the search term or 
    phrase that will be passed on the API.
  - `youtube_api_key`: Must be passed as a string. See 
    https://developers.google.com/youtube/v3/quickstart/python for instructions 
    on obtaining an API key.

## Get_CC() function
```python
get_cc(my_keyword)
```

The `Get_CC` function takes the outputted CSV from the `Make_CSV()` function 
and uses it to pass a list of `VideoID`'s and obtain their closed-captioning 
information from the YoutubeTranscriptAPI. It takes on a single argument:

  - `my_keyword`: Must be passed as a string. This is the same keyword that was 
    used to produce the CSV in `Make_CSV`. It is used to point the function to the 
    CSV file created by that function. 

## Get_LongDesc() function
```python
get_longdesc(my_keyword)
```

The `Get_LongDesc` function takes the outputted CSV from `Make_CSV()` function and 
uses it to pass a list of `VideoID`'s and obtain their long description information 
from the YoutubeAPI. It takes on a single argument:

  - `my_keyword`: Must be passed as a string. This is the same keyword that was 
    used to produce the CSV in `Make_CSV`. It is used to point the function to the 
    CSV file created by that function. 

# Fitting the Model

## ZS_Classify() function
```python
zs_classify(textVec, candidate_labels)
```

The `zs_classify` function takes a column of data from a pandas dataframe and 
candidate labels for classification and fits the data to a zero-shot model. It 
takes on two arguments:

  - `textVec`: Takes on a column from the pandas dataframe created from data 
    retrieved during the `Make_CSV()` data pull (ex. `df['LongDescription']`). 
  - `candidate_labels`: Takes on a sorted list of potential candidate labels. 
    In our case, to classify to 'science' or 'conspiracy', we would pass 
    `['science','conspiracy']`.

# Cleaning the Video Data

## Custom_Tokenize() function
```python
custom_tokenize(text)
```

The `custom_tokenize` function is a wrapper for the `word_tokenize()` function 
from the `NLTK` library that first changes `NA` values to blank strings before 
tokenizing the rows of the dataframe. It takes on a single argument:

  - `text`: Takes on a column from a pandas dataframe consisting of information 
    relating to a particular `VideoID` (ex. keywords, long description, etc.).

## Lemmatize_Sentence() function
```python
lemmatize_sentence(tokens)
```
The `lemmatize_sentence` function first adds a part-of-speech tag. It then takes 
the tokenized words (output from `custom_tokenize()`) and their associated parts 
of speech and applies the `lemmatize()` function from `NLTK`. This reduces different 
forms of a word to its lemma for better frequency analysis. For example, 'built', 
'building', and 'builds', when all categorized as verbs will be changed to their 
lemma, 'build'. It takes on a single argument:

  - `tokens`: Takes on the pandas dataframe column of tokenized words outputted by 
    the `custom_tokenize()` function.


