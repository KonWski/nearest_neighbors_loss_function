'''Auxliary methods for running reference model scripts'''

from uuid import uuid4
import os
import json
import datetime
import pickle

def update_statistics(statistics, epoch, phase, accuracy, precision, recall, f1, ef01, ef05, roc_auc, pr_auc, mcc):

    statistics["epoch"].append(epoch)
    statistics["phase"].append(phase)
    statistics["accuracy"].append(accuracy)
    statistics["precision"].append(precision)
    statistics["recall"].append(recall)
    statistics["f1"].append(f1)
    statistics["ef01"].append(ef01)
    statistics["ef05"].append(ef05)
    statistics["roc_auc"].append(roc_auc)
    statistics["pr_auc"].append(pr_auc)
    statistics["mcc"].append(mcc)

    return statistics


def create_experiment_structure(starting_path, epochs):

    experiment_hash = uuid4().hex
    experiment_path = os.path.join(starting_path, experiment_hash)
    os.makedirs(experiment_path, exist_ok=True)

    for i in range(epochs):
        os.makedirs(os.path.join(experiment_path, f"{i}"), exist_ok=True)

    return experiment_path, experiment_hash


def save_model(model, model_dir_path, epoch, experiment_hash):

    model_hash = uuid4().hex
    model_path = os.path.join(model_dir_path, f"model_{model_hash}.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump({
          "model": model,
          "experiment_hash": experiment_hash
        }, f)


def save_statistics(statistics, experiment_path):
    df_results_path = os.path.join(experiment_path, "results.csv")
    statistics.to_csv(df_results_path, index=False, sep="|")


def save_model_settings(experiment_settings, model_dir_path, random_state):
    model_settings = experiment_settings
    model_settings["random_state"] = random_state
    model_settings["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(model_dir_path, "model_settings.json"), "w") as f:
        json.dump(model_settings, f, indent=4)


def summarize_statistics(statistics, metrics):
    summary = statistics.groupby("phase")[metrics].agg(["mean", "std"])
    print(summary)