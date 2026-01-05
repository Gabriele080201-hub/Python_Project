import pandas as pd 
import numpy as np 

column_names = (
    ["id", "cycle"] +
    ["setting1", "setting2", "setting3"] +
    [f"sensor{i}" for i in range(1, 22)]
) 

class DataSource:
    def __init__(self, df, feature_cols):
        self.feature_cols = feature_cols

        self.engines = {}
        for engine_id, g in df.groupby('id'):
            self.engines[engine_id] = g.sort_values('cycle').reset_index(drop=True)

        self.current_idx = {engine_id: 0 for engine_id in self.engines}

    def step(self):
        events = []

        for engine_id, df_engine in self.engines.items():
            idx = self.current_idx[engine_id]

            if idx >= len(df_engine):
                self.current_idx[engine_id] = 0
                idx = 0
            
            row = df_engine.iloc[idx]

            event = {
                "engine_id" : engine_id,
                "cycle" : int(row['cycle']),
                "features" : row[self.feature_cols].values.astype(np.float32)
            }

            events.append(event)

            self.current_idx[engine_id] += 1
        
        return events
    
    def run(self, n_steps=1, verbose=True):
        for t in range(n_steps):
            events = self.step()
            if verbose:
                for e in events:
                    print(e)