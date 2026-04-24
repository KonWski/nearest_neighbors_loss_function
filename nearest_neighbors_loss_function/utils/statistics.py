import pandas as pd
import os
import logging
from statistics import mean

class Statistics():

    def __init__(self, n_epochs, experiment_dir_path, experiment_hash, optimized_param_name, early_stop_window_size):
        self.n_epochs = n_epochs
        self.experiment_dir_path = experiment_dir_path
        self.experiment_hash = experiment_hash
        self.optimized_param = optimized_param_name
        self.early_stop_window_size = early_stop_window_size
        self.report_path = os.path.join(self.experiment_dir_path, "train_report.xlsx")

        '''
        as [{"seed": x, "state": y, "epoch": z, "loss": q, "experiment_hash": p, "model_hash": w,
        "accuracy": a, "precision": b, "recall": c, "f1": d, "ef01": e, "ef05": f,
        "roc_auc": j, "mcc": k}]
        '''
        self.agglomerated_statistics = []
    
    def add(self, train_stats, test_stats):
        train_test_stats = train_stats | test_stats
        self.agglomerated_statistics.append(train_test_stats)

    def log_last_train_stats(self):
        logging.info(self.agglomerated_statistics[-1])

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
        df.to_excel(self.report_path, index=False)

    def early_stop_training(self, seed, alpha=0.01):

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