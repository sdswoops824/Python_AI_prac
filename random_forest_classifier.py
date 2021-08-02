# import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np


# import dataset and arrange it
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# fit the model
forest = RandomForestClassifier(n_estimators=50, random_state=0)
forest.fit(X_train, y_train)

# print the accuracy on the training data
print('Accuracy on the training subset:(:.3f)', format(forest.score(X_train, y_train)))
print('Accuracy on the testing subset:(:.3f)', format(forest.score(X_test, y_test)))

# visualizing the feature weights
n_features = cancer.data.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()