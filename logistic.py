import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def z_score_normalization(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.mean(axis=0)) / x.std(axis=0)


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(X @ self.coef_)

    def mse_cost(self, X, y):
        y_pred = self.sigmoid(X @ self.coef_)
        return np.mean((y_pred - y) ** 2)

    def log_loss_cost(self, X, y):
        y_pred = np.clip(self.sigmoid(X @ self.coef_), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit_mse(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.zeros(X.shape[1])
        N = len(y)
        self.errors_ = []

        for _ in range(self.n_epoch):
            for i in range(N):
                xi = X[i]
                yi = y[i]
                y_pred_i = self.sigmoid(np.dot(xi, self.coef_))
                grad = (y_pred_i - yi) * y_pred_i * (1 - y_pred_i)
                self.coef_ -= self.l_rate * grad * xi
                self.errors_.append(self.mse_cost(X, y))

    def fit_log_loss(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.zeros(X.shape[1])
        N = len(y)
        self.errors_ = []

        for _ in range(self.n_epoch):
            for i in range(N):
                xi = X[i]
                yi = y[i]
                y_pred_i = self.sigmoid(np.dot(xi, self.coef_))
                grad = (y_pred_i - yi) / N
                self.coef_ -= self.l_rate * grad * xi
                self.errors_.append(self.log_loss_cost(X, y))

    def predict(self, X, cut_off=0.5):
        return (self.predict_proba(X) >= cut_off).astype(int)


# --- Load dataset ---
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df[['worst concave points', 'worst perimeter', 'worst radius']]
y = df['target']

X = z_score_normalization(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=43
)

# --- Train custom MSE model ---
mse_model = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
mse_model.fit_mse(X_train.to_numpy(), y_train.to_numpy())
y_hat_mse = mse_model.predict(X_test.to_numpy())

# --- Train custom log-loss model ---
log_model = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
log_model.fit_log_loss(X_train.to_numpy(), y_train.to_numpy())
y_hat_log = log_model.predict(X_test.to_numpy())

# --- Train sklearn logistic regression ---
sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)
y_hat_sk = sk_model.predict(X_test)

# --- Compute accuracies ---
acc_mse = accuracy_score(y_test, y_hat_mse)
acc_log = accuracy_score(y_test, y_hat_log)
acc_sk = accuracy_score(y_test, y_hat_sk)

# --- Errors ---
mse_first, mse_last = mse_model.errors_[0], mse_model.errors_[-1]
log_first, log_last = log_model.errors_[0], log_model.errors_[-1]

# --- Errors ---
def to_native(obj):
    """Rekurencyjnie konwertuje numpy typy na natywne typy Pythona."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    else:
        return obj


result = {
    'mse_accuracy': acc_mse,
    'logloss_accuracy': acc_log,
    'sklearn_accuracy': acc_sk,
    'mse_error_first': mse_model.errors_[:len(X_train)],
    'mse_error_last': mse_model.errors_[-len(X_train):],
    'logloss_error_first': log_model.errors_[:len(X_train)],
    'logloss_error_last': log_model.errors_[-len(X_train):],
}

print(to_native(result))
