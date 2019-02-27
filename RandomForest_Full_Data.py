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
data = pd.read_csv('Full_Data.csv', encoding = "ISO-8859-1")
data.head(1)
trainingset = data[data['Date'] < '20150101']
testingset = data[data['Date'] > '20141231']

cleanInfo= trainingset.iloc[:,2:27]
cleanInfo.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)


list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
cleanInfo.columns= new_Index
cleanInfo.head(5)


for index in new_Index:
    cleanInfo[index]=cleanInfo[index].str.lower()
cleanInfo.head(1)
headlines = []
for row in range(0,len(cleanInfo.index)):
    headlines.append(' '.join(str(x) for x in cleanInfo.iloc[row,0:25]))
basicvectorizer = CountVectorizer(ngram_range=(1,1),min_df=5)
training = basicvectorizer.fit_transform(headlines)


newshead = []
for row in range(0,len(testingset.index)):
    newshead.append(' '.join(str(x) for x in testingset.iloc[row,2:27]))

accuracy_scores=[]

decisiontreesize=100
for i in range(decisiontreesize):
	testmodel = RandomForestClassifier(n_estimators=i+1, criterion='entropy',max_features='auto')
	testmodel = testmodel.fit(training, trainingset["Label"])
	tests = basicvectorizer.transform(newshead)
	predictions = testmodel.predict(tests)
	accuracy_scores.append(accuracy_score(testingset["Label"], predictions))

print(np.max(np.asarray(accuracy_scores))) 
iterations=range(1,decisiontreesize+1)
graphs, axes = plt.subplots()                                                                                 
axes.plot(iterations,accuracy_scores)
axes.set(xlabel='No of Trees', ylabel='Accuracy(%)',title='RandomForest : No of Decision Trees VS. Predicted Accuracy')
axes.grid()
graphs.savefig("RandomForest_Data2.png")                                                                                   
plt.show()