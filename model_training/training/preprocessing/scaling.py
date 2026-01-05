

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


class TimeSeriesScaler:
    """
    Scaler per dati time series 3D (N, T, F) coerente con Deep Learning.
    
    - Fit SOLO sui dati di training
    - Transform su training / validation / test
    - Usa StandardScaler di sklearn
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X):
        """
        Fit dello scaler sui dati di training.

        Parameters
        ----------
        X : np.ndarray
            Array di shape (N, T, F)
        """
        if X.ndim != 3:
            raise ValueError("X deve avere shape (N, T, F)")

        N, T, F = X.shape
        X_2d = X.reshape(-1, F)

        self.scaler.fit(X_2d)
        self.fitted = True

        return self

    def transform(self, X):
        """
        Applica lo scaling ai dati (senza rifittare).

        Parameters
        ----------
        X : np.ndarray
            Array di shape (N, T, F)

        Returns
        -------
        X_scaled : np.ndarray
            Array scalato di shape (N, T, F)
        """
        if not self.fitted:
            raise RuntimeError("Lo scaler deve essere fittato prima di chiamare transform().")

        if X.ndim != 3:
            raise ValueError("X deve avere shape (N, T, F)")

        N, T, F = X.shape
        X_2d = X.reshape(-1, F)
        X_scaled_2d = self.scaler.transform(X_2d)

        return X_scaled_2d.reshape(N, T, F)

    def fit_transform(self, X):
        """
        Fit + transform (da usare SOLO sul training).
        """
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        """
        Salva lo scaler su file (artefatto di training).
        """
        joblib.dump(self.scaler, path)

    def load(self, path):
        """
        Carica uno scaler già fittato.
        """
        self.scaler = joblib.load(path)
        self.fitted = True

    def get_params(self):
        """
        Restituisce informazioni utili per logging.
        """
        return {
            "type": "StandardScaler",
            "fitted": self.fitted
        }