import matplotlib.pyplot as plt
import pickle
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

#read features and labels from file
with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)
with open('features.pickle', 'rb') as f:
    features = pickle.load(f)
print('read pickles from files')
#labels=labels[:20000]
#features=features[:20000]

#print(len(labels))

features = np.array(features)
labels = np.array(labels)

labelstrain=[]
labelsclassify=[]
featuresstrain=[]
featuresclassify=[]

for i in range(len(labels)):
    if i%100>10:
        labelstrain.append(labels[i])
        featuresstrain.append(features[i])
    else:
        labelsclassify.append(labels[i])
        featuresclassify.append(features[i])
print('splitted into training and classification data')

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(featuresstrain, labelstrain)
print('finished training')
# Now predict the value of the digit on the second half:
expected = list(labelsclassify)
predicted = list(classifier.predict(featuresclassify))

print(expected[:20])
print(predicted[:20])

n_rights=0
for i in range(len(expected)):
    if expected[i]==predicted[i]:
        n_rights+=1
print(n_rights/len(expected))
