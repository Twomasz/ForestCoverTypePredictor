import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from SupportScript import get_data
import pickle


forests = get_data()

X = forests.drop('CoverType', axis=1).to_numpy()
y = forests['CoverType'].to_numpy()


# MUST STRATIFY Y, BECAUSE WE HAVE UNBALANCED DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)


clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


# === EVALUATE ===

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[i+1 for i in range(7)])
plt.figure(figsize=(12, 10))
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('True', fontsize=18)
plt.show()


# === SAVE MODEL ===

filename = 'models/RandomForest.pickle'
pickle.dump(clf, open(filename, 'wb'))
