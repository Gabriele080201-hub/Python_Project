import pandas as pd 
import numpy as np

class SlidingWindowGenerator:
    def __init__(self, window_size, feature_cols, target_col, id_col = "id", stride = 1):
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.id_col = id_col
        self.stride = stride
    
    def transform(self, df):

        X_list, y_list = [], []

        for id, group in df.groupby(self.id_col):
            x = group[self.feature_cols].values
            y = group[self.target_col].values
            n_cycles = len(group)

            if n_cycles < self.window_size:
                continue

            indices = range(0, n_cycles - self.window_size + 1, self.stride)

            for start in indices:
                end = start + self.window_size
                X_list.append(x[start:end])
                y_list.append(y[end - 1])
            
        X_out = np.stack(X_list) if X_list else np.empty((0, self.window_size, len(self.feature_cols)))
        y_out = np.array(y_list) if y_list else np.empty((0,))

        return X_out, y_out 

    def get_params(self):
        return {"window_size" : self.window_size, "stride" : self.stride, "num_features" : len(self.feature_cols)}
    

def split_by_engine(df, val_ratio=0.2, seed=42):
    engine_ids = df["id"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(engine_ids)

    split = int(len(engine_ids) * (1 - val_ratio))
    train_ids = engine_ids[:split]
    val_ids = engine_ids[split:]

    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["id"].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df