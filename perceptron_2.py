from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class PerceptronModel:# perceptron avec sklearn

    def __init__(self, max_iter=500, random_state=42):
        self.model = Perceptron(max_iter=max_iter, random_state=random_state)

    def fit(self, X_train, y_train):#entrainement du modèle
        self.model.fit(X_train, y_train)

    def pred(self, X_test):#prédiction
        return self.model.predict(X_test)

    def evale(self, y_test, y_pred):#évaluation du modèle

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, report, cm