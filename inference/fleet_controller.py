from collections import defaultdict

class FleetController:
    def __init__(self, data_source, engine_manager):
        self.data_source = data_source
        self.engine_manager = engine_manager

        self.history = defaultdict(list)

        self.current_step = 0 

    def step(self):
        events = self.data_source.step()
        predictions = self.engine_manager.process_events(events)

        for pred in predictions:
            self.history[pred['engine_id']].append({
                "cycle" : pred['cycle'],
                "rul_prediction" : pred['rul_prediction']
            })

        self.current_step += 1

        return predictions 
    
    def get_fleet_table(self):
        table = []

        for engine_id, records in self.history.items():
            if len(records) == 0:
                continue 

            last = records[-1]
            table.append({
                "engine_id" : engine_id, 
                "cycle" : last['cycle'], 
                "rul_prediction" : last['rul_prediction']
            })

        return table 
    
    def get_engine_history(self, engine_id):
        return self.history.get(engine_id, [])
    
    def reset(self):
        self.history.clear()
        self.engine_manager.engine_states.clear()
        self.current_step = 0