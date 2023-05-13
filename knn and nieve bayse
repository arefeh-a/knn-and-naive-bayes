import numpy as np
import pandas as pd


def accuracy_score(y_true, y_pred):
    # score = (y_true - y_pred) / len(y_true)
    return round(float(sum(y_pred == y_true)) / float(len(y_true)) * 100, 2)


def pre_processing(dataframe):
    # portioning data into features and target
    x = dataframe.drop([dataframe.columns[-1]], axis=1)
    y = dataframe[dataframe.columns[-1]]

    return x, y


def train_test_split(x, y, test_size=0.25, random_state=None):
    # portioning the data into train and test sets
    x_test = x.sample(frac=test_size, random_state=random_state)
    y_test = y[x_test.index]

    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)

    return x_train, x_test, y_train, y_test


class NaiveBayes:

    def __init__(self):

        self.features = list
        self.likelihoods = {}
        self.class_priors = {}
        self.pred_priors = {}

        self.x_train = np.array
        self.y_train = np.array
        self.train_size = int
        self.num_feats = int

    def fit(self, x, y):

        self.features = list(x.columns)
        self.x_train = x
        self.y_train = y
        self.train_size = x.shape[0]
        self.num_feats = x.shape[1]

        for feature in self.features:
            self.likelihoods[feature] = {}
            self.pred_priors[feature] = {}

            for feat_val in np.unique(self.x_train[feature]):
                self.pred_priors[feature].update({feat_val: 0})

                for outcome in np.unique(self.y_train):
                    self.likelihoods[feature].update({feat_val + '_' + outcome: 0})
                    self.class_priors.update({outcome: 0})

        self._calc_class_prior()
        self._calc_likelihoods()
        self._calc_predictor_prior()

        print(self.likelihoods)
        print(self.class_priors)
        print(self.pred_priors)

    def _calc_class_prior(self):
        # P(c) - Prior Class Probability
        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def _calc_likelihoods(self):
        # P(x|c) - Likelihood
        for feature in self.features:
            for outcome in np.unique(self.y_train):
                outcome_count = sum(self.y_train == outcome)
                feat_likelihood = self.x_train[feature][
                    self.y_train[self.y_train == outcome].index.values.tolist()].value_counts().to_dict()

                for feat_val, count in feat_likelihood.items():
                    self.likelihoods[feature][feat_val + '_' + outcome] = count / outcome_count

    def _calc_predictor_prior(self):
        # P(x) - Evidence
        for feature in self.features:
            feat_vals = self.x_train[feature].value_counts().to_dict()

            for feat_val, count in feat_vals.items():
                self.pred_priors[feature][feat_val] = count / self.train_size

    def predict(self, x):
        # Calculates Posterior probability P(c|x)
        results = []
        x = np.array(x)

        for new in x:
            probs_outcome = {}
            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1
                evidence = 1

                for feat, feat_val in zip(self.features, new):
                    likelihood *= self.likelihoods[feat][feat_val + '_' + outcome]
                    evidence *= self.pred_priors[feat][feat_val]

                posterior = (likelihood * prior) / evidence

                probs_outcome[outcome] = posterior

            result = max(probs_outcome, key=lambda x: probs_outcome[x])
            results.append(result)

        return np.array(results)
    
    
    
class knn:
    
    def __init__(self):

        self.features = list
        
        self.x_train = np.array
        self.y_train = np.array
        
        self.X_test = np.array
        self.y_test = np.array
        
        self.train_size = int
        self.num_feats = int
        
    def fit(self, X_train, y_train, X_test, k):
        """
        first calculating similarity_Matrix beatween X_train and X_test with manhattan metrics
        then calculating k-nearest-neighbors
        
        """
        self.k = k
        self.features = list(X.columns)
        self.X_train = X_train
        self.y_train = y_train
        # self.y_test = y_test
        self.train_size = X_train.shape[0]
        self.test_size = X_test.shape[0]
        
        # calculating similarity_Matrix beatween X_train and X_test with manhattan metrics
        similarity_Matrix = np.zeros(shape = (self.train_size, self.test_size))

        for i in range(self.test_size):
            for j in range(self.train_size):
                similarity_Matrix[j][i] = np.sum(np.absolute(np.subtract(self.X_train[j] , self.X_test[i])))
        
        # find the k-nearest-neighbors
        k_nearest_dist = np.zeros(shape = (k, self.test_size))    # a matrix of k * test size to keep distanses of k nearest neighbors
        k_nearest_lab = np.zeros(shape = (k, self.test_size))     # a matrix of k * test size to keep labels of k nearest neighbors
        trainIndex = np.zeros(shape = (k, self.test_size))        # a matrix of k * test size to keep index of k nearest neighbors
       
        for i in range(k):
            for j in range(self.test_size):
                k_nearest_dist[i, j] = np.amin(similarity_Matrix[: , j])
                result = np.where(similarity_Matrix == np.amin(similarity_Matrix[: , j]))
                a, _ = list(zip(result[0], result[1]))[0]
                trainIndex[i, j] = a
                k_nearest_lab[i, j] = y_train[trainIndex[i, j]]
                similarity_Matrix[a, j] = 100000.
                
        # making y_pred
        y_pred = np.zeros(shape= self.test_size)
        targets = np.unique(y_train)
        for m in range(self.test_size):
            sum_of_dists = np.zeros(shape=(len(targets)))
            for i in range(k):
                index_of_targets = 0
                for t in targets:
                    if k_nearest_lab[i, m] == t:
                        sum_of_dists[index_of_targets] += 1/k_nearest_dist[i, m]
                    t += 1
            y_pred[m] = targets[np.argmax(sum_of_dists)]

        return y_pred            
                

if __name__ == "__main__":
    # Weather Dataset
    # -print("\nWeather Dataset:")

    data = pd.read_csv("/home/arfeh/Downloads/success.csv")
    # print(df)

    # Split fearures and target
    X, y = pre_processing(data)

    # Split data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # X_train = np.array(list(X_train), ndmin=2)
    # X_test = np.array(list(X_test), ndmin=2)
    X_test = X_test.to_numpy()
    # X_test = X_test.to_numpy()
    X_train = X_train.to_numpy()
    # X_train = X_train.to_numpy()
    y_train = np.array(list(y_train))
    y_test = np.array(list(y_test))
    
    # print(X_test)
    knn_clsf = knn()
    print(knn_clsf.fit(X_train, y_train, X_test, 3))
"""
    naive_bayes_classifier = NaiveBayes()
    naive_bayes_classifier.fit(X_train, y_train)

    print("Train Accuracy = {}".format(accuracy_score(y_train, naive_bayes_classifier.predict(X_train))))
    print("Test Accuracy = {}".format(accuracy_score(y_test, naive_bayes_classifier.predict(X_test))))

    # new.1
    new = np.array([[24, 35]])
    print("new#1: {} ---> {}".format(new, naive_bayes_classifier.predict(new)))

    # new.2
    new = np.array([['Overcast', 'Cool', 'Normal', 't']])
    print("new#2: {} ---> {}".format(new, naive_bayes_classifier.predict(new)))

    # new.3
    new = np.array([['Sunny', 'Hot', 'High', 't']])
    print("new#3: {} ---> {}".format(new, naive_bayes_classifier.predict(new)))"""
