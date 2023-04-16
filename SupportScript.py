import pandas as pd
import numpy as np


def get_quantitative_cols():

    quantitative_cols = ['Elevation',
                         'Aspect',
                         'Slope',
                         'HydroHorDist',
                         'HydroVertDist',
                         'RoadHorDist',
                         'Hillshade_9am',
                         'Hillshade_Noon',
                         'Hillshade_3pm',
                         'FirePtsHorDist']
    return quantitative_cols


def get_wilderness_areas():

    wilderness_areas = ['Rawah_WA',
                        'Neota_WA',
                        'ComanchePeak_WA',
                        'CacheLaPoudre_WA']
    return wilderness_areas


def get_soil_types():
    soil_types = [f'SoilType_{i}' for i in range(40)]

    return soil_types


def get_data():
    """
    Creating df with column names to better insights in data
    :return: proper df
    """
    forests = pd.read_csv('data/covtype.data', header=None)

    output = ['CoverType']

    forests.columns = get_quantitative_cols() + get_wilderness_areas() + get_soil_types() + output

    return forests


class SimpleHeuristic:
    """
    IDEA:
    Compute metrics for each forest type:
    - For quantitative columns metric tell us how far from the middle is that feature.
        formula: 1 / (abs(x - x_mean) / x_std)
        (I used reciprocal number because metric should be highest for the smallest distance)

    - For dummy columns metric tell us how often given feature occurs with each forest type
        formula: x * sum(feature) / count(i_forest_type)

    At the end metrics are scaled to (0, 1) range to eliminate biases

    Sum of these metrics should give us the highest value for correct forrest type
    """
    def __init__(self):
        self.means = None
        self.stds = None
        self.frequencies = None

    def compute_weights(self, forests_df):

        CLASSES_LEN = 7
        QUANTITATIVE_ATTRIBUTES_LEN = len(get_quantitative_cols())
        OTHER_ATTRIBUTES_LEN = len(get_wilderness_areas()) + len(get_soil_types())

        means = np.zeros((CLASSES_LEN, QUANTITATIVE_ATTRIBUTES_LEN))
        stds = np.zeros((CLASSES_LEN, QUANTITATIVE_ATTRIBUTES_LEN))
        frequencies = np.zeros((CLASSES_LEN, OTHER_ATTRIBUTES_LEN))

        for i in range(CLASSES_LEN):
            forest_i = forests_df.where(forests_df['CoverType'] == i + 1).dropna()

            FOREST_TYPE_QUANTITY = forest_i.shape[0]

            # === COMPUTE WEIGHTS FOR FIRST 10 COLS ===

            quantitative_part = forest_i[get_quantitative_cols()]

            means[i, :] = quantitative_part.mean().to_numpy()
            stds[i, :] = quantitative_part.std().to_numpy()

            # === COMPUTE WEIGHTS FOR OTHER COLS ===

            not_quantitative_part = forest_i[get_wilderness_areas() + get_soil_types()]

            frequencies[i, :] = not_quantitative_part.sum().to_numpy() / FOREST_TYPE_QUANTITY

            # === ASSIGNMENTS ===

            self.means = means
            self.stds = stds
            self.frequencies = frequencies

    def save_weights(self, path_to_numpy_file: str):
        weights = np.array([])

        weights = np.append(weights, self.means)
        weights = np.append(weights, self.stds)
        weights = np.append(weights, self.frequencies)

        with open(f'{path_to_numpy_file}', 'wb') as f:
            np.save(f, weights)

    def load_weights(self, path_to_numpy_file: str):
        with open(f'{path_to_numpy_file}', 'rb') as f:
            weights = np.load(f)

            self.means = weights[:70].reshape(7, -1)
            self.stds = weights[70:140].reshape(7, -1)
            self.frequencies = weights[140:].reshape(7, -1)

    def single_predict(self, sample_features):
        # === SPLIT INTO QUANTITATIVE AND NOT QUANTITATIVE COLUMNS ===

        pivot = len(get_quantitative_cols())
        first_part_sample = sample_features[:pivot].to_numpy()
        second_part_sample = sample_features[pivot:].to_numpy()

        # === COMPUTE METRICS ===

        first_metric = 1 / (np.abs(first_part_sample - self.means) / self.stds)
        second_metric = self.frequencies * second_part_sample

        all_metrics = np.concatenate((first_metric, second_metric), axis=1)

        # === MIN-MAX SCALE ===

        mins = all_metrics.min(axis=0)
        maxs = all_metrics.max(axis=0)

        maxs_mins = maxs - mins
        maxs_mins[maxs_mins == 0] = 1  # fix columns with all 0 values

        all_metrics_scaled = (all_metrics - mins) / maxs_mins

        # === FINAL RESULTS ===

        predictions = np.sum(all_metrics_scaled, axis=1)
        predicted_class = np.argmax(predictions) + 1  # plus offset (cover type numbers begin from 1)

        return predicted_class
