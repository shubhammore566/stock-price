import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    def __init__(self, model_type: str = "Linear Regression"):
        self.model_type = model_type
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from OHLCV data."""
        feat = pd.DataFrame(index=df.index)
        feat['Close'] = df['Close']
        feat['Open'] = df['Open']
        feat['High'] = df['High']
        feat['Low'] = df['Low']
        feat['Volume'] = df['Volume'] if 'Volume' in df.columns else 0

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            feat[f'Close_lag_{lag}'] = df['Close'].shift(lag)

        # Rolling stats
        for w in [5, 10, 20]:
            feat[f'MA_{w}'] = df['Close'].rolling(w).mean()
            feat[f'STD_{w}'] = df['Close'].rolling(w).std()

        # Price momentum
        feat['Return_1'] = df['Close'].pct_change(1)
        feat['Return_5'] = df['Close'].pct_change(5)
        feat['HL_spread'] = df['High'] - df['Low']
        feat['OC_spread'] = df['Close'] - df['Open']

        feat.dropna(inplace=True)
        return feat

    def _train_linear(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def _train_rf(self, X_train, y_train):
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_split=5, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def _train_lstm(self, X_train, y_train):
        """Fallback to RF if TF not available."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping

            X_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            model = Sequential([
                LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_lstm, y_train, epochs=50, batch_size=16,
                     validation_split=0.1, callbacks=[es], verbose=0)
            self._is_lstm = True
            return model
        except ImportError:
            self._is_lstm = False
            return self._train_rf(X_train, y_train)

    def predict(self, df: pd.DataFrame, pred_days: int = 7):
        """Train model and predict future prices."""
        self._is_lstm = False

        if len(df) < 30:
            return None, 0, {}

        feat = self._build_features(df)
        if feat.empty or len(feat) < 20:
            return None, 0, {}

        close_idx = feat.columns.get_loc('Close')
        X = feat.values
        y = feat['Close'].values

        # Scale
        X_scaled = self.scaler.fit_transform(X)
        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        split = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]

        # Train chosen model
        mt = self.model_type
        if mt == "Linear Regression":
            model = self._train_linear(X_train, y_train)
        elif mt == "Random Forest":
            model = self._train_rf(X_train, y_train)
        elif mt == "LSTM Neural Network":
            model = self._train_lstm(X_train, y_train)
        else:  # Ensemble
            m1 = self._train_linear(X_train, y_train)
            m2 = self._train_rf(X_train, y_train)
            m3 = self._train_lstm(X_train, y_train)

            # Predict with each
            def _pred(m, X):
                if hasattr(self, '_is_lstm') and self._is_lstm and hasattr(m, 'predict') and 'keras' in type(m).__module__:
                    return m.predict(X.reshape(X.shape[0], 1, X.shape[1]), verbose=0).ravel()
                return m.predict(X)

            p1 = y_scaler.inverse_transform(_pred(m1, X_test).reshape(-1,1)).ravel()
            p2 = y_scaler.inverse_transform(_pred(m2, X_test).reshape(-1,1)).ravel()
            p3 = y_scaler.inverse_transform(_pred(m3, X_test).reshape(-1,1)).ravel()
            y_pred_test = (p1 + p2 + p3) / 3
            y_true_test = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()

            # Future predictions (use RF for ensemble future)
            last_features = X_scaled[-1:]
            predictions = []
            curr_feat = last_features.copy()
            for _ in range(pred_days):
                p_scaled = (m1.predict(curr_feat)[0] + m2.predict(curr_feat)[0]) / 2
                p = y_scaler.inverse_transform([[p_scaled]])[0][0]
                predictions.append(p)
                # Shift features
                new_row = curr_feat[0].copy()
                new_row[close_idx] = self.scaler.transform([[p] + [0]*(X.shape[1]-1)])[0][0]
                curr_feat = new_row.reshape(1, -1)

            metrics = self._calc_metrics(y_true_test, y_pred_test)
            confidence = max(0, min(100, 100 - metrics.get('mape', 20)))
            return predictions, confidence, metrics

        # Single model eval
        if self._is_lstm:
            y_pred_scaled = model.predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]), verbose=0).ravel()
        else:
            y_pred_scaled = model.predict(X_test)

        y_pred_test = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
        y_true_test = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()

        # Iterative future prediction
        predictions = []
        curr_feat = X_scaled[-1:].copy()
        for _ in range(pred_days):
            if self._is_lstm:
                p_scaled = model.predict(curr_feat.reshape(1, 1, curr_feat.shape[1]), verbose=0)[0][0]
            else:
                p_scaled = model.predict(curr_feat)[0]

            p = float(y_scaler.inverse_transform([[p_scaled]])[0][0])
            predictions.append(p)

            # Update rolling features with new prediction
            new_row = curr_feat[0].copy()
            scaled_p = self.scaler.transform(
                np.array([[p] + [0]*(X.shape[1]-1)])
            )[0][0]
            new_row[0] = scaled_p
            curr_feat = new_row.reshape(1, -1)

        metrics = self._calc_metrics(y_true_test, y_pred_test)
        confidence = max(0, min(100, 100 - metrics.get('mape', 20)))
        return predictions, confidence, metrics

    def _calc_metrics(self, y_true, y_pred) -> dict:
        """Calculate regression metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
