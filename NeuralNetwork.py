import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from SupportScript import get_data
import pickle

forests = get_data()

# === PREPROCESS DATA ===

X = forests.drop(columns=['CoverType'])
y = forests['CoverType'].to_numpy()
y = y - 1  # start indexing from 0

ct = pickle.load(open('models/ColumnTransformer.pickle', "rb"))

X_tran = ct.transform(X)

# we have unbalanced data
X_train, X_test, y_train, y_test = train_test_split(X_tran, y, stratify=y, test_size=0.5, random_state=42)

# === DETERMINE MODEL ===
model = Sequential()
model.add(Dense(108, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # prevent overfitting
model.add(Dense(54, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_test, y_test, epochs=30, batch_size=108)


# === MAKE PREDICTIONS ===

predictions = model.predict(X_test)
y_pred = np.zeros(predictions.shape[0])

for i, pred_for_each_class in enumerate(predictions):
    y_pred[i] = np.argmax(pred_for_each_class)

# === EVALUATE ===

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[i+1 for i in range(7)])
plt.figure(figsize=(12, 10))
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('True', fontsize=18)
plt.show()


# === SAVE MODEL ===

filename = 'models/NeuralNetwork.pickle'
pickle.dump(model, open(filename, 'wb'))
