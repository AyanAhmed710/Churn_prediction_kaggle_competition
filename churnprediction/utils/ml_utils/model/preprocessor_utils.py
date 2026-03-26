from sklearn.base import BaseEstimator, TransformerMixin

class ManualEncoder(BaseEstimator, TransformerMixin):
    
    INTERNET_SERVICE_COLS = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    BINARY_YES_NO_COLS = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'
    ]

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        X = X.copy()

        # Step 1+2 — collapse & binary encode internet cols
        for col in self.INTERNET_SERVICE_COLS:
            X[col] = X[col].replace({'No internet service': 'No'})
            X[col] = X[col].map({'Yes': 1, 'No': 0})

        # Step 3 — binary encode Yes/No cols
        for col in self.BINARY_YES_NO_COLS:
            X[col] = X[col].map({'Yes': 1, 'No': 0})

        # Step 4 — MultipleLines
        X['MultipleLines'] = X['MultipleLines'].map(
            {'Yes': 1, 'No': 0, 'No phone service': 0}
        )

        return X