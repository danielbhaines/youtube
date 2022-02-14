Youtube Classification with Zero-Shot Analysis
================

- [Required Packages](#required-packages)
- [Importing Data from YoutubeAPI]()
- [Cleaning the Video Data]()
- [Frequency and Sentiment Analysis]()


# Required Packages
```python
import numpy as np
import os
import csv
import json
import requests
import re
import pandas
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

# Cleaning the Video Data

# Frequency and Sentiment Analysis
