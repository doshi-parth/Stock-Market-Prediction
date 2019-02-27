import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('Full_Data.csv', encoding = "ISO-8859-1")
data.head(1)
train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']
sdata = train.iloc[:, 2:27]
sdata.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

list1= [i for i in range(25)]
index1=[str(i) for i in list1]
sdata.columns= index1
sdata.head(5)

for index in index1:
    sdata[index]=sdata[index].str.lower()
sdata.head(1)

labels = []

for row in range(0, len(sdata.index)):
    labels.append(' '.join(str(x) for x in sdata.iloc[row, 0:25]))

test_labels = []
for row in range(0,len(test.index)):
    test_labels.append(' '.join(str(x) for x in test.iloc[row, 2:27]))

accuracy = []

for i in range(1,4):
    vector2 = CountVectorizer(ngram_range=(1,i))
    training2 = vector2.fit_transform(labels)
    svm_model2 = svm.LinearSVC(C = 0.3,class_weight='balanced')
    svm_model2 = svm_model2.fit(training2, train["Label"])
    testing2 = vector2.transform(test_labels)
    pred2 = svm_model2.predict(testing2)
    pd.crosstab(test["Label"], pred2, rownames=["Actual"], colnames=["Predicted"])
    accuracy.append(accuracy_score(test["Label"], pred2))
    print(accuracy_score(test["Label"], pred2) * 100)

k_list = [1,2,3]
print(accuracy)
plt.plot(k_list, accuracy)
plt.xlabel("N-grams")
plt.ylabel("Accuracy")
plt.title("SVM")

plt.show()





