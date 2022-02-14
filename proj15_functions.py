#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:41:55 2021
@author: dhaines
"""

import numpy as np
import csv
import json
import requests
import re
import pandas
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

# request a search term --- have to insert dependencies inside of each function ---
my_keyword = input("Enter search keyword: ")

# define the Google API key --- have to insert dependencies inside of each function ---
youtube_api_key = input("Enter Google API key: ")


#The code and function below retrieve the videos from YouTube    
#=============================================================================
def make_csv(my_keyword):    
    # request a search term --- have to insert dependencies inside of each function ---
    my_keyword = input("Enter search keyword: ")
    
    # define the Google API key --- have to insert dependencies inside of each function ---
    youtube_api_key = input("Enter Google API key: ")    
    
    # construct the api url from the user defined term and key    
    base = "https://www.googleapis.com/youtube/v3/search?"
    fields = "&part=snippet&q="
    api_key = "&key=" + youtube_api_key
    api_url = base + fields + my_keyword +'&type=video'+'&maxResults=50' +api_key
    api_response = requests.get(api_url)
    videos = json.loads(api_response.text)
    
    # write the returned data pull to a .csv file
    with open("%syoutube_videos.csv" % my_keyword, "w") as csv_file:
        csv_writer = csv.writer(csv_file,dialect='excel')
        csv_writer.writerow(["videoID",
                            "publishedAt",
                            "title",
                            "description"])        
                            
        has_another_page = True
        while has_another_page:
            if videos.get("items") is not None:
                for video in videos.get("items"):
                    a=video["id"]["videoId"]
                    b=video["snippet"]["publishedAt"]
                    c=video["snippet"]["title"]
                    d=video["snippet"]["description"]
                    c=c.replace(',', '')
                    d=d.replace(',', '')
                    video_data_row=[a,b,c,d]
                    
                    # remove special characters from the returned data
                    for i in range(len(video_data_row)):
                        video_data_row[i] = re.sub(r'[^\x00-\x7f]',r'', video_data_row[i])
                    csv_writer.writerow(video_data_row)
                    
            if "nextPageToken" in videos.keys():
                next_page_url = api_url + "&pageToken="+videos["nextPageToken"]
                next_page_posts = requests.get(next_page_url)
                videos = json.loads(next_page_posts.text)            
                        
            else:
                print("no more videos!")
                has_another_page = False
                

#=============================================================================

### Now we retrieve the closed caption for videos for which it exists and 
### append it to our data.

### function accesses output from make_csv file and appends cc data
def get_cc(my_keyword):

    # download the CC data from the google API using the 'videoid' vector from 
    # the df dataframe. we will populate the cc vector with captions and 
    # append it to our dataframe as the column 'transcript' if it doesn't 
    # exist, the field is filled with None

    cc = []
    for item in df['videoID']:
        try:
            tx = YouTubeTranscriptApi.get_transcript(item)
            transcript = ''
            for value in tx:
                for key,val in value.items():
                    if key == 'text':
                        transcript += ' ' + val                   
            l = transcript.splitlines()
            final_tran = " ".join(l)                  
            cc.append(final_tran)
        except Exception:
            cc.append(None)
    return cc
    


#=============================================================================
# Now we are going to retrieve and append the long descriptions of the videos 
# if they exist

def get_longDesc(my_keyword):
    longDesc = []
    i = 0 
    for item in df['videoID']:
        try:
            my_videoID = item
            base = "https://www.googleapis.com/youtube/v3/videos?"
            #fields = "&part=snippet&channelId="
            fields = "&part=snippet&id="
            api_key = "&key=" + youtube_api_key
            api_url = base + fields + my_videoID +api_key
            api_response = requests.get(api_url)
            videos = json.loads(api_response.text)
            longDesc.append(videos['items'][0]['snippet']['description'])
            i = i +1
            print(i)
        except Exception:
            longDesc.append(None)
            i = i +1
            print(i)
    return longDesc

## execute intial funtion to get initial yt data
make_csv(my_keyword)

# establish a variable with the name of the .csv file to be imported
df_keyword = my_keyword + 'youtube_videos.csv'

# import the created .csv file as pandas df
df = pandas.read_csv(df_keyword, encoding = 'latin1')

# retrieve cc data 
cc1 = get_cc(my_keyword)

# retrieve long description data
ld1 = get_longDesc(my_keyword)

# append cc and ld data to the original dataframe
df['Transcript'] = cc1
df['LongDescription'] = ld1

# write complete data to file, libreOffice compains about the csv file, but 
# excel doesn't
df.to_csv('YTdata_complete.csv', columns=list(df))


##============================================================================================
##function to apply zero-shot classification 

# import the full .csv file
df1 = pandas.read_csv('YTdata_complete.csv', encoding = 'latin1')

# pipeline in our classifier from the transformers library
classifier = pipeline("zero-shot-classification")  

# define our classification function
def zs_classify(textVec, candidate_labels):

    ##intialize lists for most probable labels and corresponding scores
    lb = []
    scr = []
    
    # initialize df2 to store scores for each video
    df2 = pandas.DataFrame(columns=candidate_labels)
    
    for sequence in textVec:
        if sequence != None and sequence is not np.nan:
            # perform classification
            out = classifier(sequence, candidate_labels)
            #sort scores according to labels in alphabetical order to keep for data
            new = dict(zip(df2.columns, [x for _,x in sorted(zip(out['labels'],out['scores']))]))
            #Store scores
            df2 = df2.append(new, ignore_index=True)
            #keep top scoring label and corresponding score
            lb.append(out.get('labels')[0])
            scr.append(out.get('scores')[0])
        else:
            new = dict(zip(df2.columns, [None]*len(candidate_labels)))
            df2 = df2.append(new, ignore_index=True)
            lb.append(None)
            scr.append(None)
            #return df of classification scores and top ranking label/score
    return df2, lb, scr 

# establish candidate labels to classification
candidate_labels = sorted(["science","conspiracy theory", "education", "religion", "politics"])

# 
# tV = df[]
# score_mat, labels, scores = zs_classify(tV, candidate_labels)
# score_mat.to_csv('vaccine_microchip_probscores.csv', columns=list(score_mat))

