# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights and bias.
2. compute predictions using sigmoid function.
3. Update weight using gradient descent.
4. classify outputs using threshold (0.5).
   
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Poornima J
RegisterNumber:  212225040303
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Generate dataset
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 2. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. Initialize parameters
weights = np.zeros(2)
bias = 0
lr = 0.1
epochs = 1000

# 4. Gradient Descent
for _ in range(epochs):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)

    dw = (1/len(X)) * np.dot(X.T, (y_pred - y))
    db = (1/len(X)) * np.sum(y_pred - y)

    weights -= lr * dw
    bias -= lr * db

# 5. Predictions
probabilities = sigmoid(np.dot(X, weights) + bias)
y_pred_final = (probabilities >= 0.5).astype(int)

# 6. Classification Metrics
print("Accuracy:", accuracy_score(y, y_pred_final))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred_final))
print("\nClassification Report:\n", classification_report(y, y_pred_final))

# 7. Plot graph
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

# Decision boundary
x_vals = np.array([min(X[:, 0]), max(X[:, 0])])
y_vals = -(weights[0]*x_vals + bias) / weights[1]

plt.plot(x_vals, y_vals, color='black')

plt.title("Logistic Regression (Gradient Descent)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

```

## Output:
Accuracy: 0.99

Confusion Matrix:

 [[48  0]
 
 [ 1 51]]

Classification Report:

               precision    recall  f1-score   support
           0       0.98      1.00      0.99        48
           1       1.00      0.98      0.99        52
    accuracy                           0.99       100
   macro avg       0.99      0.99      0.99       100
weighted avg       0.99      0.99      0.99       100

<img width="565" height="453" alt="download" src="https://github.com/user-attachments/assets/f2032b31-2dae-430a-bbe7-0c0e172bd6f1" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

