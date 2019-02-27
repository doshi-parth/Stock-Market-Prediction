import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
# Read in the data
def preprocessing(A):															                            
    B=A.iloc[:,2]                                                                                           
    B.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)                                  
    C=B.values
    return C


news=pd.read_csv('data.csv')                                                                                
news_date=news['Date']
trainingset=news[news_date<'20150101']                                                                       
testset=news[news_date>'20141231']
#trainingset_label=trainingset['Label'].values.astype(np.int64)                                                
#testset_label=testset['Label'].values.astype(np.int64)                                                  
                                                                              
headlines_train=preprocessing(trainingset)                                                                   
headlines_test=preprocessing(testset)
vectorizer=CountVectorizer(ngram_range=(1,1),lowercase=True,stop_words='english',min_df=5)                  
news_training_values=vectorizer.fit_transform(headlines_train)                                  
news_test_values=vectorizer.transform(headlines_test)
#print(news_test_values.shape)
accuracy_scores=[]

decisiontreesize=100
for i in range(decisiontreesize):
	basicmodel = RandomForestClassifier(n_estimators=i+1, criterion='entropy',max_features='auto')
	basicmodel = basicmodel.fit(news_training_values, trainingset["Label"])
	predictions=basicmodel.predict(news_test_values)
	accuracy_scores.append((accuracy_score(testset["Label"], predictions)))

print(np.max(np.asarray(accuracy_scores))) 
iterations=range(1,decisiontreesize+1)
graphs, axes = plt.subplots()                                                                                 
axes.plot(iterations,accuracy_scores)
axes.set(xlabel='No of Trees', ylabel='Accuracy(%)',title='RandomForest : No of Decision Trees VS. Predicted Accuracy')
axes.grid()
graphs.savefig("RandomForest_Data.png")                                                                                   
plt.show()

