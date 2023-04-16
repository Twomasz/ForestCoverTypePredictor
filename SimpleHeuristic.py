import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from SupportScript import get_data, SimpleHeuristic


forests = get_data()

myHeuristic = SimpleHeuristic()

myHeuristic.compute_weights(forests)

X = forests.drop(['CoverType'], axis=1)
y = forests['CoverType']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

y_pred = np.array([])

for index, row in X_test.iterrows():

    y_pred = np.append(y_pred, myHeuristic.single_predict(row))


# === EVALUATE ===

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[i+1 for i in range(7)])
plt.figure(figsize=(12, 10))
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('True', fontsize=18)
plt.show()


# === SAVE HEURISTIC ===

myHeuristic.save_weights('models/SimpleHeuristicWeights.npy')
