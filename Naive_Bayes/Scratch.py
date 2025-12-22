import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """
        Learn mean, variance, and priors for each class
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]

            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples

    def _gaussian_pdf(self, class_label, x):
        """
        Compute likelihood P(x | y) assuming Gaussian
        """
        mean = self.mean[class_label]
        var = self.var[class_label]

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def predict(self, X):
        predictions = []

        for x in X:
            posteriors = []

            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self._gaussian_pdf(c, x)))
                posterior = prior + likelihood
                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)

if __name__ == "__main__":
    # Simple dataset (2 features)
    X = np.array([
        [1.0, 2.1],
        [1.5, 1.8],
        [2.0, 2.2],
        [6.0, 5.5],
        [6.5, 6.0],
        [7.0, 6.8]
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = GaussianNaiveBayes()
    model.fit(X, y)

    X_test = np.array([
        [1.8, 2.0],
        [6.3, 5.9]
    ])

    preds = model.predict(X_test)
    print("Predictions:", preds)
