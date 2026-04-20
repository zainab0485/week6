import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test, model_name):
   y_pred = model.predict(X_test)

   accuracy = accuracy_score(y_test, y_pred)

   print(f"\n{model_name} Accuracy:")
   print(accuracy)

   print(f"\n{model_name} Classification Report:")
   print(classification_report(y_test, y_pred))

   print(f"\n{model_name} Confusion Matrix:")
   print(confusion_matrix(y_test, y_pred))

   disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
   plt.title(f"{model_name} Confusion Matrix")

   file_name = model_name.replace(" ", "_").lower() + "_confusion_matrix.png"
   plt.savefig(file_name)
   plt.close()

   return accuracy