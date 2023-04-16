from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from SupportScript import get_data, get_soil_types, get_wilderness_areas, get_quantitative_cols
import pickle

forests = get_data()

X = forests.drop(columns=['CoverType'])
y = forests['CoverType'].to_numpy()

# we have unbalanced data, so must stratify y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)

ct = ColumnTransformer([('scale_num_data', StandardScaler(), get_quantitative_cols()),
                        ('pass_dummy_data', 'passthrough', get_wilderness_areas() + get_soil_types())])

# fit transformer for only train data
# I will use same data for kNN and NN algorithms
ct.fit(X_train)

# === SAVE TRANSFORMER ===

filename = 'models/ColumnTransformer.pickle'
pickle.dump(ct, open(filename, 'wb'))
