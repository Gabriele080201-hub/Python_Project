from collections import defaultdict
from typing import List, Dict, Any
import pandas as pd
import numpy as np

class FleetController:
    """
    Orchestratore della flotta. Gestisce il flusso di dati, l'invocazione del modello
    e mantiene lo storico completo (sensori e predizioni) per ogni motore.
    """
    def __init__(self, data_source, engine_manager):
        self.data_source = data_source
        self.engine_manager = engine_manager
        
        # La feature_cols ci serve per dare i nomi corretti alle colonne dei sensori
        self.feature_cols = data_source.feature_cols
        
        # history[engine_id] = [ {cycle, features, rul_prediction}, ... ]
        self.history = defaultdict(list)
        self.current_step = 0 

    def step(self) -> List[Dict[str, Any]]:
        """
        Esegue un passo della simulazione: acquisisce dati, genera predizioni
        e aggiorna lo storico interno.
        """
        events = self.data_source.step()
        predictions = self.engine_manager.process_events(events)

        # Creiamo una mappa rapida per associare la predizione all'id motore corretto
        # Nota: le predizioni sono presenti solo se il buffer (es. 30 cicli) è pieno
        pred_map = {p['engine_id']: p['rul_prediction'] for p in predictions}

        for event in events:
            eid = event['engine_id']
            
            record = {
                "cycle": event['cycle'],
                "features": event['features'], # Array numpy con i valori dei sensori
                "rul_prediction": pred_map.get(eid, None) # None se il modello non è ancora "ready"
            }
            self.history[eid].append(record)

        self.current_step += 1
        return predictions
    
    def get_fleet_table(self) -> List[Dict[str, Any]]:
        """
        Restituisce una lista di dizionari con l'ultimo stato noto di ogni motore.
        Utile per visualizzare la tabella riassuntiva in Streamlit.
        """
        table = []
        for eid, records in self.history.items():
            if not records: 
                continue
            
            last = records[-1]
            table.append({
                "engine_id": eid, 
                "cycle": last['cycle'], 
                "rul_prediction": last['rul_prediction']
            })
        return table 
    
    def get_engine_history_df(self, engine_id: int) -> pd.DataFrame:
        """
        Trasforma lo storico di un singolo motore in un DataFrame Pandas.
        Include il ciclo, la RUL predetta e tutte le colonne dei sensori.
        """
        if engine_id not in self.history or not self.history[engine_id]:
            return pd.DataFrame()
        
        records = self.history[engine_id]
        
        expanded_data = []
        for r in records:
            # Creiamo il dizionario base con Ciclo e RUL
            row = {
                "cycle": r["cycle"], 
                "rul_prediction": r["rul_prediction"]
            }
            
            # Esplodiamo l'array delle features nei nomi delle colonne corrispondenti
            # Es: {"sensor1": 591.2, "sensor2": 1.3, ...}
            sensor_values = dict(zip(self.feature_cols, r["features"]))
            row.update(sensor_values)
            
            expanded_data.append(row)
        
        return pd.DataFrame(expanded_data)

    def reset(self):
        """Ripristina lo stato della simulazione e dei motori."""
        self.history.clear()
        self.engine_manager.engine_states.clear()
        self.current_step = 0