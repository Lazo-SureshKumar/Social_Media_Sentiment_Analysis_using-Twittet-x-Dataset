from textblob import TextBlob
import csv
import os 
import numpy as np
import pandas as pd
UPLOAD_FOLDER = 'uploads'

def get_tweet_sentiment():
    df = pd.read_csv(UPLOAD_FOLDER+'/clean.csv')
    sentiments = []
    senti =[]
    s =[]
    for tweet in df['tweet']:
        i=1
        if i>0:
            blob = TextBlob(tweet)
            polarity = blob.sentiment.polarity
            if polarity>0:
                sentiment = "Positive"
                senti.append(["Positive"])
            elif polarity < 0:
                sentiment = "Negative"
                senti.append(["Negative"])
            else:
                sentiment = "Neutral"
                senti.append(["Neutral"])
            sentiments.append({'tweet':tweet,'polority':polarity,'sentiment':sentiment})
    with open(UPLOAD_FOLDER+'/cleaned.csv','w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["tweet","polarity","sentiment"])
        for s in sentiments:
            csvwriter.writerow([s['tweet'],s['polority'],s['sentiment']])
    return sentiments
def get_sentiment():
    df = pd.read_csv(UPLOAD_FOLDER+'/clean.csv')
    tweets = df['tweet']
    total = []
    for tweet in df['tweet']:
        i=1
        if i>0:
            blob = TextBlob(tweet)
            polarity = np.array([blob.sentiment.polarity])
            subjectivity = np.array([blob.sentiment.subjectivity])
    average_polartity = np.mean(polarity)
    average_sujectivity = np.mean(subjectivity)
    total.append({'avg_polarity':average_polartity,'avg_subjectivity':average_sujectivity})
    return total
    
     

if __name__ == "__main__":
    sentiments = get_tweet_sentiment()