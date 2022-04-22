# -*- coding: utf-8 -*-
"""FinalDef_new.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d6L8_h77hE5jzxy7VARYxglTt-f_qb3M
"""

import numpy as np
import pandas as pd
import warnings
import re 
from tqdm import tnrange, tqdm_notebook, tqdm
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

import json
import os
import random
import subprocess
import time
from datetime import date, datetime, timedelta
import requests

from twython import Twython
from time import sleep
import io

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

"""# Part 1 Twitter """

def weightavg(group,avg_name,wei_name):
  a=group[avg_name]
  w=group[wei_name]+1
  try:
    return (a*w).sum()/w.sum()
  except:
    return a.mean()

def extract(NUMBER_OF_QUERIES,ACCESS_TOKEN,query):
    data = {"statuses": []}
    next_id = "" #"1147236962945961984"
    APP_KEY ='vJB7L6fhV3hYPQjXdgSDtzWdy'  #'mPQKoRwd2Pb9qpQyQmyG5s8KR'
    APP_SECRET =  'dUTeph2pJCaojtpuiv7M7UDLeEiuR6qTBhD0fOzdaTuOE8xTZF' #'HLvIhusvfzDLKaRXY8CnZGP143kp3E3f2KqQBIEMfVL5mOxZjq'
    since_id= ''
    
    # Extract data
    twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)
    for i in tqdm(range(NUMBER_OF_QUERIES)):
        if not next_id:
            data = twitter.search(q=query, lang='en', result_type='recent', count="100",tweet_mode='extended',since_id=since_id) # Use since_id for tweets after id
#           print(data)
        elif since_id:
            data["statuses"].extend(twitter.search(q=query, lang='en', result_type='mixed', count="100",max_id=next_id,tweet_mode='extended')["statuses"])
        else:
            data["statuses"].extend(twitter.search(q=query, lang='en', result_type='mixed', count="100", max_id=next_id,tweet_mode='extended')["statuses"])
        if len(data["statuses"]) > 1:
            next_id = data["statuses"][len(data["statuses"]) - 1]['id']
            
    d = pd.DataFrame([[s["id"], s["full_text"].replace('\n','').replace('\r',''), s["user"]["name"], s["user"]["followers_count"], s["retweet_count"], s["favorite_count"], s["created_at"]] for s in data["statuses"]], columns=('ID', 'Text', 'UserName', "UserFollowerCount", 'RetweetCount', 'Likes', "CreatedAt"))
    return d

def preTwitter(d):
    # Clean
    for i,s in enumerate(tqdm(d['Text'])):
        text = d.loc[i, 'Text']
        text = text.replace("#", "")
        text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text, flags=re.MULTILINE)
        text = re.sub('@\\w+ *', '', text, flags=re.MULTILINE)
        d.loc[i, 'Text'] = text

    d.drop(d[d['CreatedAt']=='CreatedAt'].index,inplace=True)
    d['time'] = d['CreatedAt'].apply(lambda x: pd.to_datetime(x))
    d['time'] = d['time'].apply(lambda x: x.replace(second=0))
    d['time'] = d['time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    d['RetweetCount']=d['RetweetCount'].astype(int)
    d.drop(['ID',"UserName","UserFollowerCount","Likes","CreatedAt"],axis=1,inplace=True)
  
    # Add sentiment score
    sia = SIA()
    d['score'] = d['Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

    score1=d.groupby('time')['score'].mean()
    s1=pd.DataFrame({'time':score1.index, 'avg':score1.values})

    score2=d.groupby("time").apply(weightavg,"score","RetweetCount")
    s2=pd.DataFrame({'time':score2.index, 'wgtavg':score2.values})

    result=pd.merge(s1,s2,on='time')
    result['open_time'] = result['time'].apply(lambda x: pd.to_datetime(x))
    result=result[['open_time','avg','wgtavg']]

    return result

def ExtractTwitter(t):    
    # Define the currency
    CURRENCY = "bitcoin"
    CURRENCY_SYMBOL = "BTC"
    query = '#%s OR #%s'%(CURRENCY,CURRENCY_SYMBOL)

    # Authentication
    APP_KEY ='vJB7L6fhV3hYPQjXdgSDtzWdy'  #'mPQKoRwd2Pb9qpQyQmyG5s8KR'
    APP_SECRET =  'dUTeph2pJCaojtpuiv7M7UDLeEiuR6qTBhD0fOzdaTuOE8xTZF' #'HLvIhusvfzDLKaRXY8CnZGP143kp3E3f2KqQBIEMfVL5mOxZjq'
    twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()
    twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)
    twitter.get_application_rate_limit_status()['resources']['search']

    # If it's the first time executing this function, extract around 60 mins of data
    if ExtractTwitter.Flag == 0:
        NUMBER_OF_QUERIES = 150
        ExtractTwitter.Flag = 1
    else: # If it's not the first time, extract arount 1 min of data
        NUMBER_OF_QUERIES = 1
    
    # Extract
    d = extract(NUMBER_OF_QUERIES,ACCESS_TOKEN,query)
    print(d)
    # Preprocess
    d = preTwitter(d)
    print(d)
    if NUMBER_OF_QUERIES == 150:
        d = d[d['open_time'] >= t]
        ExtractTwitter.tweets = d
        print(ExtractTwitter.tweets)
    else:
        d = d.iloc[-1]
        x = ExtractTwitter.tweets.append(d)
        ExtractTwitter.tweets = x.iloc[1: , :]
        print(ExtractTwitter.tweets)
        
    return ExtractTwitter.tweets

ExtractTwitter.Flag = 0
ExtractTwitter.tweets = pd.DataFrame(columns = ['open_time', 'avg', 'wgtavg'])

"""# Part 2 Price"""

API_BASE = 'https://api.binance.com/api/v3/'

LABELS = [
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'close_time',
    'quote_asset_volume',
    'number_of_trades',
    'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume',
    'ignore'
]

def set_dtypes(df):

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    #df = df.set_index('open_time', drop=True)
    print(df.columns)

    df = df.astype(dtype={
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64',
        'quote_asset_volume': 'float64',
        'number_of_trades': 'int64',
        'taker_buy_base_asset_volume': 'float64',
        'taker_buy_quote_asset_volume': 'float64'
        #'ignore': 'float64'
    })
    return df

def quick_clean(df):

    # drop dupes
    dupes = df['open_time'].duplicated().sum()
    if dupes > 0:
        df = df[df['open_time'].duplicated() == False]

    # sort by timestamp, oldest first
    df.sort_values(by=['open_time'], ascending=False)
    
    df.drop(['close_time','ignore'],axis=1,inplace=True)


    return df

def get_batch(symbol, interval='1m', start_time=0, limit=1000):

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'limit': limit
    }
    try:
        # timeout should also be given as a parameter to the function
        response = requests.get(f'{API_BASE}klines', params, timeout=30)
    except requests.exceptions.ConnectionError:
        print('Connection error, Cooling down for 5 mins...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)
    
    except requests.exceptions.Timeout:
        print('Timeout, Cooling down for 5 min...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)
    
    except requests.exceptions.ConnectionResetError:
        print('Connection reset by peer, Cooling down for 5 min...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)

    if response.status_code == 200:
        return pd.DataFrame(response.json(), columns=LABELS)
    print(f'Got erroneous response back: {response}')
    return pd.DataFrame([])

def ExtractPrice(t):
    base='BTC'
    quote='BUSD'
    t=time.strptime(t,'%Y-%m-%d %H:%M:%S')
    timeStamp = int(time.mktime(t))*1000

    new_batch = get_batch(symbol=base+quote,
            interval='1m',
            start_time=timeStamp)
    new_batch = quick_clean(new_batch)
    new_batch=set_dtypes(new_batch)
    return new_batch

"""# Part 3 Preprocess"""

def Preprocess(t):
    tweets_raw_file=ExtractTwitter(t)
    pricedf=ExtractPrice(t)  
    data=pd.merge(tweets_raw_file,pricedf,how='inner',on='open_time')
    return data

"""# Part4 Final Def"""

#Run!
#t为需要预测的时间-60min,秒数=0。字符串格式。(也可以直接为timestamp及其他格式,看twitter要求)
#t为utc0时区=time.time()=东8区时间-8hours <binance api要求> 
#当前时间为'2022-04-20 14:00:00'则：t='2022-04-20 13:00:00'
t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 60 * 60 - 8 * 60 * 60))
print(t)
data=Preprocess(t)

#data.to_csv('data.csv',index=False)
data.head()
