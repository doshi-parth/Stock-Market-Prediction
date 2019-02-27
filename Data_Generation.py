""" This is the code for Data Generation from Pkl file to data.csv.
	Data Labelling is done using price differences. 1 if price increases, 0 if price decreases due to headlines respectively.
    Developed by Parth Rajendra Doshi 1215200012"""

import pandas as pd
import numpy as np
df=pd.read_pickle('Pickled ten year filtered data (Articles + DJIA).pkl')					#Reading Pickle File
df['prices']=df['adj close'].apply(np.int64)												#Conert Float to Int
df['articles']=df['articles'].map(lambda x: x.lstrip('.-'))									#Eliminate punctuations
articles=df['articles'].values																#Numpy Array of Headlines
date_time_stamp=df.index																	#Date of Headlines
date_time_stamp=date_time_stamp.strftime('%Y-%m-%d')										#Numpy of Date
prices=df['prices'].values																	#Numpy of Prices
label=np.zeros(len(prices)-1)																#Initial Label Array
for i in range(len(prices)-1):																#Labelling of Headlines
	if prices[i]-prices[i+1]>0:																#If Price of Stock decreases next day, then headlines are labelled 0 or negative
		label[i]=0
	if prices[i]-prices[i+1]<0:																#If Price of Stock increases next day, then headlines are labelled 1 or positive
		label[i]=1
date_time_stamp=date_time_stamp[0:len(label)]												#Removal of last element from date
articles=articles[0:len(label)]																#Removal of last element from articles
m=np.stack((date_time_stamp, label, articles), axis=-1)										#Stacking of all three numpy arrays
df=pd.DataFrame(m)																			#Conversion into Pandas Dataframe
df.to_csv("data.csv",header=['Date','Label','Headlines'],index=False)						#Generation of data.csv

