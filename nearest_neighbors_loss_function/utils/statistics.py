import pandas as pd
import logging
from statistics import mean
from nearest_neighbors_loss_function.utils.data_orchestrator import DataOrchestrator

class Statistics():

    def __init__(self, n_epochs: int, orchestrator: DataOrchestrator, experiment_hash, optimized_param_name: str, 
                 early_stop_window_size: int):
        self.n_epochs = n_epochs
        self.orchestrator = orchestrator
        self.experiment_hash = experiment_hash
        self.optimized_param = optimized_param_name
        self.early_stop_window_size = early_stop_window_size
        self.report_path = orchestrator.report_path

        '''
        as [{"seed": x, "state": y, "epoch": z, "loss": q, "experiment_hash": p, "model_hash": w,
        "accuracy": a, "precision": b, "recall": c, "f1": d, "ef01": e, "ef05": f,
        "roc_auc": j, "mcc": k}]
        '''
        self.agglomerated_statistics = self._load_statistics()
    
    def _load_statistics(self):
        if self.orchestrator.isfile(self.report_path):
            df = self.orchestrator.read_df(self.report_path)
            agglomerated_statistics = df.to_dict(orient="records")
        else:
            agglomerated_statistics = []
        
        return agglomerated_statistics
    
    def seed_processed(self, seed):
        for record in self.agglomerated_statistics:
            if seed == record["seed"]:
                return True
        
        return False

    def add(self, train_stats, test_stats):
        auxiliary_fields = {"best_model": 0, "model_path": None}
        full_stats = train_stats | test_stats | auxiliary_fields
        self.agglomerated_statistics.append(full_stats)

    def mark_best_model(self, seed, best_epoch, model_path):
        for stat_id in range(len(self.agglomerated_statistics)):
            if self.agglomerated_statistics[stat_id]["seed"] == seed and self.agglomerated_statistics[stat_id]["epoch"] == best_epoch:
                self.agglomerated_statistics[stat_id]["best_model"] = 1
                self.agglomerated_statistics[stat_id]["model_path"] = model_path

    def log_last_train_stats(self):
        logging.info(self.agglomerated_statistics[-1])

    def get_best_models_info(self):
        df = pd.DataFrame(self.agglomerated_statistics)
        return df[df["best_model"] == 1].to_dict(orient="records")

    def upload_test_stats(self, test_stats, seed, epoch):
        for stat_id in range(len(self.agglomerated_statistics)):
            if self.agglomerated_statistics[stat_id]["seed"] == seed and self.agglomerated_statistics[stat_id]["epoch"] == epoch:
                self.agglomerated_statistics[stat_id] = self.agglomerated_statistics[stat_id] | test_stats
                break

    def _get_best_model_stats(self):
        df = pd.DataFrame(self.agglomerated_statistics)
        best_model_stats = df.loc[df[self.optimized_param].idxmax()].to_dict()
        return best_model_stats

    def log_best_model_stats(self):
        best_model_stats = self._get_best_model_stats()
        logging.info(best_model_stats)

    def save(self):
        df = pd.DataFrame(self.agglomerated_statistics)
        df["experiment_hash"] = self.experiment_hash
        self.orchestrator.save_df(df, self.report_path)

    def early_stop_training(self, seed, alpha=0.0):
        seed_agglomerated_statistics = [stat for stat in self.agglomerated_statistics if stat["seed"] == seed]
        n_seed_stats = len(seed_agglomerated_statistics)

        if n_seed_stats < 2 * self.early_stop_window_size:
            return False

        current_window_optimized_params = [seed_agglomerated_statistics[i][f"valid_{self.optimized_param}"] 
                                           for i in range(n_seed_stats - self.early_stop_window_size, n_seed_stats)]
        self.current_window_mean = mean(current_window_optimized_params)

        last_window_optimized_params = [seed_agglomerated_statistics[i][f"valid_{self.optimized_param}"] 
                                           for i in range(n_seed_stats - 2 * self.early_stop_window_size, n_seed_stats - self.early_stop_window_size)]
        self.last_window_mean = mean(last_window_optimized_params)

        if (1 + alpha) * self.current_window_mean < self.last_window_mean:
            return True

        return False