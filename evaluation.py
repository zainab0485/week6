
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
   y_pred = model.predict(X_test)

   print("Accuracy:")
   print(accuracy_score(y_test, y_pred))

   print("\nClassification Report:")
   print(classification_report(y_test, y_pred))

   print("\nConfusion Matrix:")
   print(confusion_matrix(y_test, y_pred))