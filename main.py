from prog import load_data
from perceptron_2 import PerceptronModel

def main():
    X_train, X_test, y_train, y_test = load_data()
    perceptron = PerceptronModel(max_iter=500, random_state=42)

    perceptron.fit(X_train, y_train)
    y_pred = perceptron.pred(X_test)
    accuracy, report, cm = perceptron.evale(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)


if __name__ == "__main__":
    main()