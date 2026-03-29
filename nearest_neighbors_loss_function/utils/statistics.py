import pandas as pd
import os
import logging

class Statistics():

    def __init__(self, n_epochs, experiment_dir_path, experiment_hash, optimized_param_name):
        self.n_epochs = n_epochs
        self.experiment_dir_path = experiment_dir_path
        self.experiment_hash = experiment_hash
        self.optimized_param = optimized_param_name
        self.report_path = os.path.join(self.experiment_dir_path, "train_report.xslx")
        print(f"self.report_path: {self.report_path}")

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

    def upload_test_stats(self, test_stats, epoch):
        self.agglomerated_statistics[epoch] = self.agglomerated_statistics[epoch] | test_stats

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