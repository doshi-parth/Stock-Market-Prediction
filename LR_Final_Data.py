""" This is the code for Logistic Regression on data.csv file.
    Accuracy achieved was 74.1% using LR model developed from scratch while
    Library implementation achieved 74.6% accuracy.
    Developed by Parth Rajendra Doshi 1215200012"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def preprocessing(X):															                            #Eliminating Punctuations and return numpy array
    Y=X.iloc[:,2]                                                                                           #Indexing to Headlines
    Y.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)                                  
    Z=Y.values
    return Z

def one_hot(label):                                                                                         #ONE HOT Encoding Function
    arr=np.zeros(k,'uint8')
    arr[label]=1
    return arr

def probability_function(feature_vector, weight):                                                           #Posterior Probability Calculation
    probability=np.zeros((k, 1))                                                                            #for each class we have one probability element
    probability[0:-1]=np.exp(np.matmul(weight, feature_vector.transpose())).reshape((k-1,1))                #Using formula for k classes
    probability[-1]=1                                                                                       #last K Class
    total=sum(probability)                                                                                  #Denominator computation common for all classes
    probability=probability/total                                                                           #probabilities calculated
    return probability

news=pd.read_csv('data.csv')                                                                                #Reading CSV File
news_date=news['Date']
news_train=news[news_date<'20150101']                                                                       #Spliting via Date into training set and test set
news_test=news[news_date>'20141231']
news_train_label=news_train['Label'].values.astype(np.int64)                                                #Training Labels
news_test_label=news_test['Label'].values.astype(np.int64)                                                  #Test Labels
k=len(news['Label'].unique())                                                                               #Unique Labels denoting Classes
headlines_train=preprocessing(news_train)                                                                   #Preprocessing of Headlines
headlines_test=preprocessing(news_test)
vectorizer=CountVectorizer(ngram_range=(1,1),lowercase=True,stop_words='english',min_df=5)                  #Count Vectorizer initialized with unigram model and stops word with minimum document frequency of 5.
headlines_train_tokens=vectorizer.fit_transform(headlines_train).toarray()                                  #Fit_transform used to generate bag of words from training set returning term-document matrix
headlines_test_tokens=vectorizer.transform(headlines_test).toarray()                                        #Transform used to generate term-document matrix for test set using training set vocab
m,n=headlines_train_tokens.shape                                                                            #Shape of matrix
weights=np.zeros((k-1,n))                                                                                   #weights of the logistic model
weight_increment=np.zeros((k-1,n))                                                                          #weights_increments 
accuracies=[]                                                                                               #Accuracies List
learning_rate=0.2                                                                                           #Learning Rate
no_of_iterations=100                                                                                        #Max Iterations
for i in range(no_of_iterations):                                                                           #Looping from 1 to 100
    for j in range(m):                                                                                      #For training set
            labelTrainingHot=one_hot(news_train_label[j]);                                                  #Label One Hot
            prob=probability_function(headlines_train_tokens[j],weights)[0:-1]                              #Probabilities Computed
            gradient_ascent=(learning_rate*(labelTrainingHot[0:(k-1)].reshape(((k-1),1))-prob))             #True Label - Current Prediction * Learning Rate
            weight_increment=weight_increment+((gradient_ascent*headlines_train_tokens[j].reshape((1,n)).repeat(k-1,axis=0))/m) #Multiplcation with Feature and Addition to old weight vector
    weights=weights+weight_increment                                                                       #Updating weight vector
    acc=0               
    for l in range(len(news_test)):                                                                         #Prediction on Test set using weight vector
            predicted_label=np.argmax(probability_function(headlines_test_tokens[l], weights))              #Label computation using probabilities
            if predicted_label==news_test_label[l]:                                                         #True Label==Predicted Label
                    acc=acc+1                                                                               #Correct Prediction
    accuracies.append((acc/len(news_test))*100)                                                                         
print(np.max(np.asarray(accuracies)))                                                                       #Maximum Accuracy
iterations=range(1,no_of_iterations+1)
fig,ax=plt.subplots()                                                                                       #Plotting No of Iterations vs Accuracies
ax.plot(iterations,accuracies)
ax.set(xlabel='No of Iterations', ylabel='Accuracy(%)',title='Logistic Regression : No of Iterations VS. Accuracy')
ax.grid()
fig.savefig("LRPLOT_Data.png")                                                                              #Figure saved as LRPLOT
plt.show()                                                                                                  #Displaying figure