import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from typing import List


def entropy(q):
    return -(q*np.log2(q) + (1-q)*np.log2(1-q))


def remainder(label, feature, _df_copy):
    total_rows_count = len(_df_copy.index)
    values = _df_copy[feature].unique()
    rem = 0
    for value in values:
        rows = _df_copy.loc[_df_copy[feature] == value]
        rows_count = len(rows.index)
        p = len(rows.loc[rows[label] == 1].index)
        rem = rem + (rows_count/total_rows_count)*entropy(p/rows_count)
    return rem


def info_gain(_df, label, feature):
    return entropy(len(_df.loc[_df[label] == 1]) / len(_df.index)) - remainder(label, feature, _df)


def filter_df_with_info_gain(_df, label, num_col, number_of_features):
    # Binning continuous columns in 2 buckets
    _df_copy = _df.copy()
    for column in num_col:
        _df_copy[column] = pd.qcut(_df_copy[column], 2, labels=[-1, 1], duplicates='drop')

    info_gain_list = []
    for col in _df.drop([label], axis=1).columns.to_list():
        info_gain_list.append({"feature": col, "info_gain": info_gain(_df_copy, label, col)})

    # Keeping a info gain sorted list, from max to min
    info_gain_list = sorted(info_gain_list, key=lambda d: d['info_gain'],  reverse=True)
    top_features = list(map(lambda x: x["feature"], info_gain_list[:number_of_features]))
    top_features.append(label)

    _df = _df[top_features]
    cate_col = _df.select_dtypes(include=['object']).columns.tolist()
    num_col = _df.select_dtypes(exclude=['object']).columns.tolist()
    num_col.remove(label)

    print(top_features)
    print()
    return _df, cate_col, num_col


def one_hot_encode_categorical(_df, cate_col):
    # One-hot-encoding categorical columns
    _df = pd.get_dummies(_df, columns=cate_col)
    return _df


def standardize_numerical(_df, num_col):
    # Standardize
    scaler = StandardScaler()
    _df[num_col] = scaler.fit_transform(_df[num_col])
    print("mean", _df[num_col].mean(), "std", _df[num_col].std())
    print()
    return _df


class FirstPreprocessor:
    def __init__(self, filename, label, positive_value, negative_value, useless_fields, numeric_categorical_list):
        self.filename = filename
        self.label = label
        self.positive_value = positive_value
        self.negative_value = negative_value
        self.useless_fields = useless_fields
        self.numeric_categorical_list = numeric_categorical_list

    def first_draft(self):
        # Reading the data info dataframe
        _df = pd.read_csv(self.filename)
        # Useless field drop
        _df = _df.drop(self.useless_fields, axis=1)
        # Replacing spaces and empty strings with NaNs
        _df = _df.replace(r'^\s+$', np.nan, regex=True)
        # Data type fix of non label columns
        _df = _df.astype({"TotalCharges": float})
        # Checking the number of NaNs is each column
        print(_df.isna().sum())
        print()
        return _df

    def label_encoding(self, _df):
        # Label encoding
        _df[self.label].replace({self.positive_value: "1", self.negative_value: "-1"}, inplace=True)
        # Parsing strings to float
        _df = _df.astype({self.label: 'int64'})
        print(_df.info())
        print()
        return _df

    def get_column_type_categorized(self, _df):
        # Primarily separating numerical and categorical
        cate_col = _df.select_dtypes(include=['object']).columns.tolist()
        num_col = _df.select_dtypes(exclude=['object']).columns.tolist()
        # Removing Label and transferring Numeric Categorical
        for elem in [self.label] + self.numeric_categorical_list:
            num_col.remove(elem)
        # cate_col.extend(["SeniorCitizen"])
        cate_col.extend(self.numeric_categorical_list)
        # Checking
        print("Numerical", len(num_col))
        for column in num_col:
            print(len(_df[column].unique()), column)
        print()
        print("Categorical", len(cate_col))
        for column in cate_col:
            print(len(_df[column].unique()), column)
        print()
        print()
        return cate_col, num_col

    def preprocess(self):
        df = self.first_draft()
        df = self.label_encoding(df)
        categorical_col, numerical_col = self.get_column_type_categorized(df)

        # Replacing NaNs with mean, mode
        df[categorical_col] = df[categorical_col].fillna(df.mode().iloc[0])
        df[numerical_col] = df[numerical_col].fillna(df.mean())
        print()

        df, categorical_col, numerical_col \
            = filter_df_with_info_gain(df, label=self.label, num_col=numerical_col, number_of_features=10)
        df = one_hot_encode_categorical(df, categorical_col)
        df = standardize_numerical(df, numerical_col)

        # Test train split
        _df_train, _df_test = train_test_split(df, test_size=0.20, random_state=77)
        print(len(_df_train.index))
        print(len(_df_test.index))
        return _df_train, _df_test


class SecondPreprocessor:
    def __init__(self, label, positive_value, negative_value, useless_fields, numeric_categorical_list):
        self.label = label
        self.positive_value = positive_value
        self.negative_value = negative_value
        self.useless_fields = useless_fields
        self.numeric_categorical_list = numeric_categorical_list
        self.train_length = 0
        self.test_length = 0

    def first_draft(self):
        df1 = pd.read_csv("adult.data.csv")
        df2 = pd.read_csv("adult.test.csv", names=list(df1.columns)).iloc[1:, :]

        df1 = df1.replace(r'\?', np.nan, regex=True)
        df2 = df2.replace(r'\?', np.nan, regex=True)

        df1 = df1.dropna()
        df2 = df2.dropna()

        self.train_length = len(df1.index)
        self.test_length = len(df2.index)

        print(len(df1.index))
        print(len(df2.index))

        df = pd.concat([df1, df2], axis=0)
        return df

    def label_encoding(self, _df):
        # Label encoding
        _df[self.label].replace({self.positive_value: "1", self.negative_value: "-1"}, inplace=True)
        _df[self.label].replace({self.positive_value + ".": "1", self.negative_value + ".": "-1"}, inplace=True)
        # Parsing strings to float
        _df = _df.astype({self.label: 'int64'})
        print(_df.info())
        print()
        return _df

    def get_column_type_categorized(self, _df):
        # Primarily separating numerical and categorical
        cate_col = _df.select_dtypes(include=['object']).columns.tolist()
        num_col = _df.select_dtypes(exclude=['object']).columns.tolist()
        # Removing Label and transferring Numeric Categorical
        for elem in [self.label] + self.numeric_categorical_list:
            num_col.remove(elem)
        # cate_col.extend(["SeniorCitizen"])
        cate_col.extend(self.numeric_categorical_list)
        # Checking
        print("Numerical", len(num_col))
        for column in num_col:
            print(len(_df[column].unique()), column)
        print()
        print("Categorical", len(cate_col))
        for column in cate_col:
            print(len(_df[column].unique()), column)
        print()
        print()
        return cate_col, num_col

    def preprocess(self):
        df = self.first_draft()
        df = self.label_encoding(df)
        categorical_col, numerical_col = self.get_column_type_categorized(df)

        df = one_hot_encode_categorical(df, categorical_col)
        df = standardize_numerical(df, numerical_col)

        _df_train = df.head(self.train_length).reset_index(drop=True)
        _df_test = df.iloc[-self.test_length:].reset_index(drop=True)

        return _df_train, _df_test


class ThirdPreprocessor:
    def __init__(self, filename, label, positive_value, negative_value, useless_fields, numeric_categorical_list):
        self.filename = filename
        self.label = label
        self.positive_value = positive_value
        self.negative_value = negative_value
        self.useless_fields = useless_fields
        self.numeric_categorical_list = numeric_categorical_list

    def first_draft(self):
        # Reading the data info dataframe
        _df = pd.read_csv(self.filename)

        _df = pd.concat([_df.loc[_df["Class"] == 1], _df.loc[_df["Class"] == 0].sample(10000)], axis=0)

        # Useless field drop
        _df = _df.drop(self.useless_fields, axis=1)
        # Replacing spaces and empty strings with NaNs
        _df = _df.replace(r'^\s+$', np.nan, regex=True)
        # Checking the number of NaNs is each column
        print(_df.isna().sum())
        print()
        return _df

    def label_encoding(self, _df):
        # Label encoding
        _df[self.label].replace({self.positive_value: "1", self.negative_value: "-1"}, inplace=True)
        # Parsing strings to float
        _df = _df.astype({self.label: 'int64'})
        print(_df.info())
        print()
        return _df

    def get_column_type_categorized(self, _df):
        # Primarily separating numerical and categorical
        cate_col = _df.select_dtypes(include=['object']).columns.tolist()
        num_col = _df.select_dtypes(exclude=['object']).columns.tolist()
        # Removing Label and transferring Numeric Categorical
        for elem in [self.label] + self.numeric_categorical_list:
            num_col.remove(elem)
        # cate_col.extend(["SeniorCitizen"])
        cate_col.extend(self.numeric_categorical_list)
        # Checking
        print("Numerical", len(num_col))
        for column in num_col:
            print(len(_df[column].unique()), column)
        print()
        print("Categorical", len(cate_col))
        for column in cate_col:
            print(len(_df[column].unique()), column)
        print()
        print()
        return cate_col, num_col

    def preprocess(self):
        df = self.first_draft()
        df = self.label_encoding(df)
        categorical_col, numerical_col = self.get_column_type_categorized(df)

        # Replacing NaNs with mean, mode
        df[categorical_col] = df[categorical_col].fillna(df.mode().iloc[0])
        df[numerical_col] = df[numerical_col].fillna(df.mean())
        print(df.isna().sum())
        print()

        df, categorical_col, numerical_col = \
            filter_df_with_info_gain(df, label=self.label, num_col=numerical_col, number_of_features=10)
        df = one_hot_encode_categorical(df, categorical_col)
        df = standardize_numerical(df, numerical_col)

        # Test train split
        _df_train, _df_test = train_test_split(df, test_size=0.20, random_state=77)
        print(len(_df_train.index))
        print(len(_df_test.index))
        return _df_train, _df_test


class LogisticRegressionLearner:
    def __init__(self, alpha,  train: pd.DataFrame, label, threshold=0):
        self.label = label
        self.X_train = train.drop([label], axis=1)
        # self.X_train["X0"] = np.ones(self.X_train.shape[0])
        self.Y_train = train[label]
        self.m = len(self.X_train.index)
        self.n = self.X_train.shape[1]
        self.W = np.zeros(self.n)
        self.alpha = alpha
        self.epoch = 1000
        self.threshold = threshold

    def HX(self):
        return np.tanh(np.dot(self.X_train, self.W))

    def train(self):
        for i in range(self.epoch):
            # Gradient descent
            # W = W - a * 2 * [[Y - H(X)] * [1- H^2(X)]]^T * X
            product = (self.Y_train - self.HX()) * (np.ones(self.m) - np.square(self.HX()))
            penalty = self.alpha * (2 / self.m) * np.dot(product.transpose(), self.X_train)
            self.W = self.W + penalty
            loss = np.mean(np.square(self.Y_train - self.HX()))
            if loss < self.threshold:
                break

    def predict(self, X: pd.DataFrame):
        hypothesises = np.tanh(np.dot(X, self.W))
        hypothesises = (hypothesises >= 0).astype(int)
        hypothesises[hypothesises == 0] = -1
        return hypothesises

    def get_test_metrics(self, test):
        y_predict = (self.predict(test.drop([self.label], axis=1)) >= 0).astype(int)
        y_true = (test[self.label] > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        accuracy = ((tp + tn)/(tn + fp + fn + tp))*100
        sensitivity = (tp/(tp+fn))*100
        specificity = (tn/(tn+fp))*100
        precision = (tp/(tp+fp))*100
        false_discovery = (fp/(tp+fp))*100
        f1 = (2*tp / (2*tp + fp + fn))*100
        return {"Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity,
                "Precision": precision, "False Discovery": false_discovery, "F1 score": f1}

    def get_training_metrics(self):
        y_predict = (self.HX() >= 0).astype(int)
        y_true = (self.Y_train > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        accuracy = ((tp + tn)/(tn + fp + fn + tp))*100
        sensitivity = (tp/(tp+fn))*100
        specificity = (tn/(tn+fp))*100
        precision = (tp/(tp+fp))*100
        false_discovery = (fp/(tp+fp))*100
        f1 = (2*tp / (2*tp + fp + fn))*100
        return {"Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity,
                "Precision": precision, "False Discovery": false_discovery, "F1 score": f1}


class Adaboost:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, K, label):
        self.K = K
        self.label = label
        self.N = len(train.index)
        self.train = train
        self.X_train = self.train.drop([label], axis=1)
        self.Y_train = self.train[self.label].to_numpy()
        self.X_test = test.drop([label], axis=1)
        self.Y_test = test[self.label].to_numpy()
        self.W = np.empty(self.N)
        self.W.fill(1/self.N)
        self.learners: List[LogisticRegressionLearner] = list()
        self.Z = np.empty(self.K)

    def boost(self):
        for k in range(self.K):
            resampled_train = self.train.sample(n=self.N, replace=True, weights=self.W, random_state=1111)
            learner_weak = LogisticRegressionLearner(0.1, resampled_train, self.label)
            learner_weak.train()
            self.learners.append(learner_weak)

            error = 0
            predictions = learner_weak.predict(self.X_train)
            for j in range(self.N):
                if predictions[j] != self.Y_train[j]:
                    error += self.W[j]
            print("Round:", k, " Error:", error)
            if error > 0.5:
                continue
            for j in range(self.N):
                if predictions[j] == self.Y_train[j]:
                    self.W[j] *= (error / (1-error))

            self.W = self.W / np.sum(self.W)
            self.Z[k] = np.log2((1 - error) / error)

    def get_training_accuracy(self):
        ensemble_h = np.empty(self.N)
        for k in range(self.K):
            ensemble_h += self.learners[k].predict(self.X_train) * self.Z[k]

        ensemble_h = (ensemble_h >= 0).astype(int)
        ensemble_h[ensemble_h == 0] = -1

        matching = (ensemble_h == self.Y_train)
        accuracy = np.mean(matching.astype(int))
        return accuracy*100

    def get_test_accuracy(self):
        ensemble_h = np.zeros(len(self.X_test.index))
        for k in range(self.K):
            ensemble_h += self.learners[k].predict(self.X_test) * self.Z[k]

        ensemble_h = (ensemble_h >= 0).astype(int)
        ensemble_h[ensemble_h == 0] = -1

        matching = (ensemble_h == self.Y_test)
        accuracy = np.mean(matching.astype(int))
        return accuracy*100


if __name__ == "__main__":
    # First Dataset
    Label = "Churn"
    df_train, df_test \
        = FirstPreprocessor("WA_Fn-UseC_-Telco-Customer-Churn.csv",
                            Label, "Yes", "No", ['customerID'], ['SeniorCitizen']).preprocess()

    # Second Dataset
    # Label = " <=50K"
    # df_train, df_test = SecondPreprocessor(Label, " >50K", " <=50K", [], []).preprocess()

    # Third Dataset
    # Label = "Class"
    # df_train, df_test = ThirdPreprocessor("creditcard.csv", Label, 1, 0, [], []).preprocess()

    print()
    print("Logistic Regression:")
    learner = LogisticRegressionLearner(.1, df_train, Label)
    learner.train()
    print("Training:", learner.get_training_metrics())
    print("Test", learner.get_test_metrics(df_test))
    print()
    print()

    print("Adaboost:")
    for _K in range(5, 25, 5):
        booster = Adaboost(df_train, df_test, _K, Label)
        booster.boost()
        print("K: ", _K)
        print("Training Accuracy:", booster.get_training_accuracy())
        print("Test Accuracy:", booster.get_test_accuracy())
        print()
