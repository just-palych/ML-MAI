import numpy as np
from collections import Counter



def find_best_split(feature_vector, target_vector):
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)
    num_samples = len(feature_vector)

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    split_points = np.where(sorted_features[1:] != sorted_features[:-1])[0]
    if len(split_points) == 0:
        return None, None, None, None

    cumulative_samples_left = np.arange(1, num_samples + 1)
    cumulative_class1_left = np.cumsum(sorted_targets)
    cumulative_class0_left = cumulative_samples_left - cumulative_class1_left

    total_class1 = cumulative_class1_left[-1]
    total_class0 = cumulative_class0_left[-1]

    cumulative_samples_left = cumulative_samples_left[:-1]
    cumulative_class1_left = cumulative_class1_left[:-1]
    cumulative_class0_left = cumulative_class0_left[:-1]

    cumulative_samples_right = num_samples - cumulative_samples_left
    cumulative_class1_right = total_class1 - cumulative_class1_left
    cumulative_class0_right = total_class0 - cumulative_class0_left

    class1_rate_left = cumulative_class1_left / cumulative_samples_left
    class0_rate_left = cumulative_class0_left / cumulative_samples_left
    class1_rate_right = cumulative_class1_right / cumulative_samples_right
    class0_rate_right = cumulative_class0_right / cumulative_samples_right

    gini_left = 1 - class1_rate_left**2 - class0_rate_left**2
    gini_right = 1 - class1_rate_right**2 - class0_rate_right**2

    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    weighted_ginis = -cumulative_samples_left / num_samples * gini_left - cumulative_samples_right / num_samples * gini_right

    filtered_ginis = weighted_ginis[split_points]
    filtered_thresholds = thresholds[split_points]
    best_split_index = np.argmax(filtered_ginis)

    return filtered_thresholds, filtered_ginis, filtered_thresholds[best_split_index], filtered_ginis[best_split_index]

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and node.get("depth", 0) >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                categories = np.unique(sub_X[:, feature])
                clicks = {}
                counts = {}

                for category in categories:
                    mask = sub_X[:, feature] == category
                    counts[category] = np.sum(mask)
                    clicks[category] = np.sum(sub_y[mask])

                sorted_categories = sorted(categories,
                                        key=lambda x: clicks[x]/counts[x] if counts[x] > 0 else 0)
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}


                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat in categories_map
                                    if categories_map[cat] < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        left_mask = split
        right_mask = ~split

        if (self._min_samples_leaf is not None and
            (np.sum(left_mask) < self._min_samples_leaf or
             np.sum(right_mask) < self._min_samples_leaf)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["left_child"]["depth"] = node.get("depth", 0) + 1
        node["right_child"]["depth"] = node.get("depth", 0) + 1

        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"])
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
