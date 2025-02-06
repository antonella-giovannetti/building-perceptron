from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
class PerceptronModel:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  
        self.max_iter = max_iter  
        self.weights = None  
        self.bias = None  

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.max_iter):
            for idx, sample in enumerate(X):
                linear_output = np.dot(sample, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                # Corriger ici
                update = self.learning_rate * (y[idx] - y_predicted)  
                self.weights += update * sample
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)  

    def evaluate(self, y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return accuracy, report, cm
