import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from SupportScript import get_data
import pickle


forests = get_data()

# === PREPROCESS DATA ===

X = forests.drop(columns=['CoverType'])
y = forests['CoverType'].to_numpy()

ct = pickle.load(open('models/ColumnTransformer.pickle', "rb"))

X_tran = ct.transform(X)

# we have unbalanced data
X_train, X_test, y_train, y_test = train_test_split(X_tran, y, stratify=y, test_size=0.5, random_state=42)

# === MAKE PREDICTIONS ===

clf = KNeighborsClassifier(n_neighbors=3, n_jobs=2)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test[:20000])


# === EVALUATE ===

print(classification_report(y_test[:20000], y_pred))

cm = confusion_matrix(y_test[:20000], y_pred, labels=[i+1 for i in range(7)])
plt.figure(figsize=(12, 10))
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('True', fontsize=18)
plt.show()


# === SAVE MODEL ===

filename = 'models/NearestNeighbors.pickle'
pickle.dump(clf, open(filename, 'wb'))
