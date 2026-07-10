import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score


class Node:
    def __init__(self, X, y, depth):
        if len(y) == 0:
            raise ValueError("y must not be empty")
        
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.depth = depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None


class Criterion(ABC):

    @abstractmethod
    def impurity(self, y):
        pass

    def split_score(self, y_left, y_right):
        n_left = len(y_left) 
        n_right = len(y_right)
        n = n_left + n_right
        
        return (n_left / n * self.impurity(y_left)
                + n_right / n * self.impurity(y_right))


class GiniCriterion(Criterion):
    def __init__(self, classes):
        self.classes_ = classes

    def _get_counts(self, y):
        return np.sum(y == self.classes_[:, None], axis=1)

    def impurity(self, y):
        counts = self._get_counts(y)
        total = np.sum(counts)
        return 1 - np.sum((counts / total) ** 2)
    
    def impurity_from_counts(self, counts):
        n = np.sum(counts)
        return 1 - np.sum((counts / n) ** 2)
    
    def split_score_counts(self, left_counts, right_counts):
        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)
        n = n_left + n_right
        
        return (n_left / n * self.impurity_from_counts(left_counts)
                + n_right / n * self.impurity_from_counts(right_counts))


class MSECriterion(Criterion):

    def impurity(self, y):
        return np.mean((y - np.mean(y))**2)


class DecisionTreeBase(ABC):
    def __init__(self, criterion=None, max_depth=10):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.root = Node(X, y, depth=0)
        self._build_tree(self.root)

        return self

    def _build_tree(self, node):
        if node.depth >= self.max_depth or len(np.unique(node.y)) == 1:
            node.value = self._make_leaf_value(node)

        else:
            node.feature_index, node.threshold = self._find_best_split(node)

            if node.feature_index is None or node.threshold is None:
                node.value = self._make_leaf_value(node)
            else:
                mask = node.X[:, node.feature_index] >= node.threshold

                X_left = node.X[~mask]
                y_left = node.y[~mask]

                X_right = node.X[mask]
                y_right = node.y[mask]

                node.left = Node(X_left, y_left, node.depth + 1)
                node.right = Node(X_right, y_right, node.depth + 1)

                self._build_tree(node.left)
                self._build_tree(node.right)

    @abstractmethod
    def _find_best_split(self, node):
        pass

    @abstractmethod
    def _make_leaf_value(self, node):
        pass

    def _through_the_tree(self, x_row):
        node = self.root
        value = node.value

        while value is None:
            if x_row[node.feature_index] >= node.threshold:
                node = node.right
            else:
                node = node.left
            value = node.value
        
        return value


class my_DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, max_depth=10, max_features=None, random_state=None):
        super().__init__(max_depth=max_depth)

        self.max_features = max_features
        self.rng = np.random.default_rng(seed=random_state)
        self.classes_ = None

    def fit(self, X, y, classes=None):
        y = np.asarray(y)

        if classes is None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = classes

        self.class_to_index = {
            c : i for i, c in enumerate(self.classes_)
        }
        self.criterion = GiniCriterion(self.classes_)
        return super().fit(X, y)
    
    def _find_best_split(self, node):
        best_score = np.inf
        best_feature_index = None
        best_threshold = None

        n_features = node.X.shape[1]

        if self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
            feature_indices = self.rng.choice(n_features, size=n_selected, replace=False)
        else:
            feature_indices = np.arange(n_features)

        for j in feature_indices:
            order = np.argsort(node.X[:, j])
            x_sort = node.X[order, j]
            y_sort = node.y[order]

            left_counts = np.zeros(len(self.classes_))
            right_counts = np.sum(y_sort == self.classes_[:, None], axis=1)

            for i in range(len(order) - 1):
                ind = self.class_to_index[y_sort[i]]
                left_counts[ind] += 1
                right_counts[ind] -= 1
                if x_sort[i] == x_sort[i + 1]:
                    continue
                
                score = self.criterion.split_score_counts(left_counts, right_counts)
                if score < best_score:
                    best_score = score
                    best_feature_index = j
                    best_threshold = (x_sort[i] + x_sort[i + 1]) / 2
        
        return best_feature_index, best_threshold

    def _make_leaf_value(self, node):
        return np.sum(node.y == self.classes_[:, None], axis=1) / len(node.y)

    def predict_proba(self, X):
        X = np.asarray(X)
        proba = []
        for x_row in X:
            proba.append(self._through_the_tree(x_row))
        
        proba = np.asarray(proba)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        max_indices = np.argmax(proba, axis=1)
        return self.classes_[max_indices]


class my_DecisionTreeRegressor(DecisionTreeBase):
    def __init__(self, max_depth=10, max_features=None, random_state=None):
        super().__init__(criterion=MSECriterion(), max_depth=max_depth)

        self.max_features = max_features
        self.rng = np.random.default_rng(seed=random_state)
    
    def _find_best_split(self, node):
        best_score = np.inf
        best_feature_index = None
        best_threshold = None

        n_features = node.X.shape[1]

        if self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
            feature_indices = self.rng.choice(n_features, size=n_selected, replace=False)
        else:
            feature_indices = np.arange(n_features)

        for j in feature_indices:
            feat_sort = np.sort(np.unique(node.X[:, j]))
            thresholds = []

            for i in range(len(feat_sort) - 1):
                thresholds.append((feat_sort[i] + feat_sort[i + 1]) / 2)

            for threshold in thresholds:
                mask = node.X[:, j] >= threshold
                if np.sum(mask) == 0 or np.sum(~mask) == 0:
                    continue
                y_left = node.y[~mask]
                y_right = node.y[mask]
                score = self.criterion.split_score(y_left, y_right)
                if best_score > score:
                    best_score = score
                    best_feature_index = j
                    best_threshold = threshold
        
        return best_feature_index, best_threshold

    def _make_leaf_value(self, node):
        return np.mean(node.y)

    def predict(self, X):
        X = np.asarray(X)
        pred = []

        for x_row in X:
            pred.append(self._through_the_tree(x_row))

        pred = np.asarray(pred)
        return pred


class my_ExtraTreeClassifier(my_DecisionTreeClassifier):
    def __init__(self, max_depth=10, n_times=30, random_state=21):
        super().__init__(max_depth=max_depth)
        self.n_times = n_times
        self.rng = np.random.default_rng(seed=random_state)

    def _find_best_split(self, node):
        best_score = np.inf
        best_feature_index = None
        best_threshold = None

        for _ in range(self.n_times):
            feature_index = self.rng.integers(low=0, high=len(node.X[0,:]))
            x_min = np.min(node.X[:, feature_index])
            x_max = np.max(node.X[:, feature_index])
            threshold = self.rng.uniform(x_min, x_max)
            mask = node.X[:, feature_index] >= threshold
            y_left = node.y[~mask]
            y_right = node.y[mask]
            if len(y_left)==0 or len(y_right)==0:
                continue
            score = self.criterion.split_score(y_left, y_right)

            if score < best_score:
                best_score = score
                best_feature_index = feature_index
                best_threshold = threshold

        return best_feature_index, best_threshold


class my_ExtraTreesClassifier:
    def __init__(self, n_estimators=25, max_depth=5, n_times=30, random_state=21):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_times = n_times
        self.rng = np.random.default_rng(seed=random_state)
        self.trees = [my_ExtraTreeClassifier(max_depth=self.max_depth, n_times=self.n_times, random_state=self.rng.integers(1, 1_000_000)) for _ in range(self.n_estimators)]


    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        for i in range(self.n_estimators):
            self.trees[i].fit(X, y, self.classes_)

        return self
    
    def predict_proba(self, X):
        X = np.asarray(X)

        proba = []
        for i in range(self.n_estimators):
            proba.append(self.trees[i].predict_proba(X))

        proba = np.asarray(proba)

        proba = np.mean(proba, axis=0)

        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        max_indices = np.argmax(proba, axis=1)
        return self.classes_[max_indices]


class my_RandomForestClassifier:
    def __init__(self, max_depth=10, random_state=21, n_estimators=25, max_features='sqrt'):
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed=random_state)
        self.classes_ = None
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = [my_DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, random_state=self.rng.integers(0, 1_000_000)) for _ in range(self.n_estimators)]
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        for i in range(self.n_estimators):
            indices = self.rng.choice(np.arange(len(X)), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            self.trees[i].fit(X_bootstrap, y_bootstrap, self.classes_)

        return self
    
    def predict_proba(self, X):
        X = np.asarray(X)

        proba = []
        for i in range(self.n_estimators):
            proba.append(self.trees[i].predict_proba(X))

        proba = np.asarray(proba)

        proba = np.mean(proba, axis=0)

        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        max_indices = np.argmax(proba, axis=1)
        return self.classes_[max_indices]


class my_GradientBoostingClassifier:
    def __init__(self, max_depth=3, number_of_trees=100, max_features=None, random_state=21, learning_rate=0.1, threshold=0.5):
        self.max_depth = max_depth
        self.max_features = max_features
        self.number_of_trees = number_of_trees
        self.rng = np.random.default_rng(seed=random_state)
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.trees = []

    def _sigmoid(self, a):
        a = np.clip(a, -500, 500)
        return 1 / (1 + np.exp(-a))

    def _nll(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.mean(- y * np.log(p) - (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        pred = np.zeros(len(y))

        for i in range(self.number_of_trees):

            prob = self._sigmoid(pred)
            grad = y - prob
            tree = my_DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.max_features, random_state=self.rng.integers(1, 1_000_000))

            tree.fit(X, grad)

            update = tree.predict(X)

            pred += self.learning_rate * update

            self.trees.append(tree)

        return self
    
    def predict_proba(self, X):
        X = np.asarray(X)

        pred = np.zeros(X.shape[0])

        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)

        prob = self._sigmoid(pred)
        return np.column_stack((1 - prob, prob))

    
    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]

        return np.where(prob >= self.threshold, 1, 0)
    


def tune_and_evaluate(model, param_space, X_train, y_train, X_valid, y_valid, search_type='random', n_iter=50):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=21
    )

    if search_type == 'random':
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            random_state=21,
            n_jobs=-1,
            refit=True
        )

    elif search_type=='grid':
        search = GridSearchCV(
            estimator=model,
            param_grid=param_space,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            refit=True
        )
    
    else:
        raise ValueError("search_type must be 'random' or 'grid'")
    
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    train_proba = best_model.predict_proba(X_train)[:, 1]
    valid_proba = best_model.predict_proba(X_valid)[:, 1]

    train_gini = 2 * roc_auc_score(y_train, train_proba) - 1
    valid_gini = 2 * roc_auc_score(y_valid, valid_proba) - 1

    result = {
        'model' : model.__class__.__name__,
        'best_params': search.best_params_,
        'best_cv_gini' : 2 * search.best_score_ - 1,
        'train_gini' : train_gini,
        'valid_gini' : valid_gini,
        'overfit_gap' : train_gini - valid_gini,
        'best_model' : best_model
    }

    return result