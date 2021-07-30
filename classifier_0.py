from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

print(label_names)

print(labels[0])
print(feature_names[0])
print(features[0])

# splitting the data into training and testing data
# here, 40% of the data is used for testing
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.40, random_state=42)

# initialize the Naive Bayes model
gnb = GaussianNB()

# fit the model to the data
model = gnb.fit(train, train_labels)

# perform predictions on the test data
preds = gnb.predict(test)
print(preds)

# find accuracy of prediction
print(accuracy_score(test_labels,preds))
