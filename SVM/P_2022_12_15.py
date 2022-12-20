# Práctica 15-12-2022
# Suárez Pérez Juan Pablo

# Import the libraries needed
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

# The model to predict with SVM
class SVM:
    
    # Global Attributes
    c_negative = list()
    c_positive = list()
    c = list()
    c_norm = 0
        
    # Method to fit the model
    def fit(self, X_train, y_train):
        # List of positives and negatives instance
        positives = list()
        negatives = list()
        for i in range(len(y_train)):
            if y_train[i] == 1:
                positives.append(X_train[i])
            else:
                negatives.append(X_train[i])
        positives = np.array(positives)
        negatives = np.array(negatives)
        # Create the vector c positive an negative
        self.c_positive = np.mean(positives, 0)
        self.c_negative = np.mean(negatives, 0)
        self.c_positive = np.array(self.c_positive)
        self.c_negative = np.array(self.c_negative)
        # Create the vector c and his norm
        self.c = np.array(self.c_positive + self.c_negative) / 2
        self.c_norm = np.linalg.norm(self.c)
    
    # Method to predict new instances
    def predict(self, X_test):
        y_predict = list()
        for x in X_test:
            proyection = np.dot(x, self.c) / self.c_norm
            if proyection > self.c_norm:
                y_predict.append(1)
            else:
                y_predict.append(0)
        return y_predict

# Get data
df = pd.read_csv('./heart.csv', sep = ',', engine = 'python')
X = df.drop(['target'], axis = 1).values
y = df['target'].values

# Generate Data Test and Data Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Create the model and fit the model with the train set
svm = SVM()
svm.fit(X_train, y_train)

print('C Positive:')
print(svm.c_positive)

print('C Negative:')
print(svm.c_negative)

print('C:')
print(svm.c)

print('C Norm:')
print(svm.c_norm)

# Predict the instance of test set
y_predict = svm.predict(X_test)

results = list()
for i in range(len(y_test)):
    results.append([y_test[i], y_predict[i]])
print(tabulate(results, headers = ['Y Test', 'Y Predict'], tablefmt = 'github'))

# Report of the model
target_names = list(map(str, [0, 1]))
print(classification_report(y_test, y_predict, target_names=target_names))
cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot()
plt.show()