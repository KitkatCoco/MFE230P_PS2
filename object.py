import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer

class DecisionTree:
    def __init__(self, max_depth=10, min_gain=0.01, max_leaf=100):
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.max_leaf = max_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return self._create_leaf(y)

        best_split = self._find_best_split(X, y)
        if best_split['gain'] < self.min_gain:
            return self._create_leaf(y)

        left_indices = X[:, best_split['feature']] <= best_split['threshold']
        right_indices = X[:, best_split['feature']] > best_split['threshold']

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _find_best_split(self, X, y):
        best_split = {'gain': -1}
        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature], y, threshold)
                if gain > best_split['gain']:
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'gain': gain
                    }
        return best_split

    def _information_gain(self, feature_values, y, threshold):
        parent_variance = self._variance(y)
        left_indices = feature_values <= threshold
        right_indices = feature_values > threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])

        var_left = self._variance(y[left_indices])
        var_right = self._variance(y[right_indices])

        child_variance = (n_left / n) * var_left + (n_right / n) * var_right
        ig = parent_variance - child_variance
        return ig

    def _variance(self, y):
        return np.var(y)

    def _create_leaf(self, y):
        return np.mean(y)

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature = tree['feature']
        threshold = tree['threshold']
        if x[feature] <= threshold:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_gain=0.01, max_leaf=100):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.max_leaf = max_leaf
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in tqdm(range(self.n_trees), desc="Training Trees"):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_subset, y_subset = X[indices], y[indices]
            tree = DecisionTree(max_depth=self.max_depth, min_gain=self.min_gain, max_leaf=self.max_leaf)
            tree.fit(X_subset, y_subset)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in tqdm(self.trees, desc="Making Predictions")])
        return np.mean(tree_predictions, axis=0)

def grid_search(X, y):
    param_grid = {
        'max_depth': [5, 10, 15],
        'min_gain': [0.01, 0.05, 0.1],
        'max_leaf': [50, 100, 150]
    }
    best_params = None
    best_mse = float('inf')

    for max_depth in param_grid['max_depth']:
        for min_gain in param_grid['min_gain']:
            for max_leaf in param_grid['max_leaf']:
                model = RandomForest(n_trees=10, max_depth=max_depth, min_gain=min_gain, max_leaf=max_leaf)
                kf = KFold(n_splits=5)
                mse_scores = []

                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    mse_scores.append(mean_squared_error(y_val, y_pred))

                avg_mse = np.mean(mse_scores)
                print(f"Params: max_depth={max_depth}, min_gain={min_gain}, max_leaf={max_leaf} - MSE: {avg_mse}")

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = (max_depth, min_gain, max_leaf)

    print(f"Best Params: max_depth={best_params[0]}, min_gain={best_params[1]}, max_leaf={best_params[2]} - MSE: {best_mse}")
    return best_params

def cross_validation(X, y, best_params):
    model = RandomForest(n_trees=10, max_depth=best_params[0], min_gain=best_params[1], max_leaf=best_params[2])
    kf = KFold(n_splits=5)
    mse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, y_pred))
        print(f"Fold MSE: {mse_scores[-1]}")

    avg_mse = np.mean(mse_scores)
    print(f"Average MSE: {avg_mse}")

    # Save the model to disk
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return avg_mse

def preprocess_data(df, continuous_features, n_bins=10):
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df[continuous_features] = kbd.fit_transform(df[continuous_features])
    return df

if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('train_data2.csv', index_col=0)
    continuous_features = ['macro_state_1', 'macro_state_2']
    df = preprocess_data(df, continuous_features, n_bins=10)
    X = df.drop(columns=['outcome']).values
    y = df['outcome'].values

    # Perform grid search for hyperparameter tuning
    best_params = grid_search(X, y)

    # Perform 5-fold cross-validation with the best parameters
    cross_validation(X, y, best_params)

    # Load the model from disk
    with open('random_forest_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Predict on the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = loaded_model.predict(X_test)

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')
