from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from skfp.metrics import enrichment_factor
from .generate_embeddings import generate_embeddings
from .checkpoints import load_model
from .auxiliary_functions import ignore_top_weight
import torch.nn.functional as F
import torch
import logging
from sklearn.ensemble import RandomForestClassifier
from nearest_neighbors_loss_function.utils.auxiliary_functions import set_seed

def evaluate_model(model, evaluation_mode, train_embeddings, train_labels, test_embeddings, 
                   test_labels, evaluate_model_params, stats_prefix, seed=None):

    model.eval()

    if torch.isnan(train_embeddings).any() or torch.isnan(test_embeddings).any():
        return None, True

    # reshape to 1d
    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()

    if evaluate_model_params.evaluation_model_name == "knn":
        accuracy, precision, recall, f1, ef01, ef05, roc_auc, pr_auc, mcc = knn_stats(evaluation_mode, train_embeddings, test_embeddings, train_labels, 
                                                                                    test_labels, evaluate_model_params)

    elif evaluate_model_params.evaluation_model_name == "rf":
        accuracy, precision, recall, f1, ef01, ef05, roc_auc, pr_auc, mcc = rf_stats(train_embeddings, test_embeddings, train_labels, 
                                                                                    test_labels, evaluate_model_params, seed)

    else:
        raise Exception("Evaluation model not implemented")

    # epoch_loss = round(running_loss / (data_id + 1), 5)
    test_stats = {f"{stats_prefix}_accuracy": accuracy, f"{stats_prefix}_precision": precision, f"{stats_prefix}_recall": recall, 
                  f"{stats_prefix}_f1": f1, f"{stats_prefix}_ef01": ef01, f"{stats_prefix}_ef05": ef05, 
                  f"{stats_prefix}_roc_auc": roc_auc, f"{stats_prefix}_pr_auc": pr_auc, f"{stats_prefix}_mcc": mcc}

    return test_stats, False


def knn_stats(evaluation_mode, train_embeddings, test_embeddings, y_train, y_test, evaluate_model_params):

    if evaluation_mode in ["valid", "test"]:

        # normalize the embeddings
        train_embeddings = F.normalize(train_embeddings, dim=1)
        test_embeddings = F.normalize(test_embeddings, dim=1)

        # calculate distances
        train_distances = 2 - 2 * (train_embeddings @ train_embeddings.T)
        test_train_distances = 2 - 2 * (test_embeddings @ train_embeddings.T)

        train_distances = train_distances.clamp(min=0)
        test_train_distances = test_train_distances.clamp(min=0)

        # convert torch -> numpy
        train_distances = train_distances.numpy()
        test_train_distances = test_train_distances.numpy()

        # fit model
        knn = KNeighborsClassifier(n_neighbors=evaluate_model_params.knn_n_neighbors, n_jobs=-1, metric="precomputed", weights='distance')
        knn.fit(train_distances, y_train)

    elif evaluation_mode == "train":

        # normalize the embeddings
        train_embeddings = F.normalize(train_embeddings, dim=1)

        # calculate distances
        train_distances = 2 - 2 * (train_embeddings @ train_embeddings.T)
        train_distances = train_distances.clamp(min=0)

        # convert torch -> numpy
        train_distances = train_distances.numpy()
        test_train_distances = train_distances

        # fit model
        knn = KNeighborsClassifier(n_neighbors=evaluate_model_params.knn_n_neighbors, n_jobs=-1, metric="precomputed", weights=ignore_top_weight)
        knn.fit(train_distances, y_train)

    # predictions
    y_train = y_train.numpy()
    y_test = y_test.numpy()

    y_pred = knn.predict(test_train_distances)
    y_pred_proba = knn.predict_proba(test_train_distances)[:,1]

    # scores
    accuracy, precision, recall, f1, ef01, ef05, roc_auc, pr_auc, mcc = calculate_stats(y_test, y_pred, y_pred_proba)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, pr_auc, mcc


def rf_stats(train_embeddings, test_embeddings, y_train, y_test, evaluate_model_params, seed):

        rf = RandomForestClassifier(
            n_jobs=-1,
            random_state=seed,
            n_estimators=evaluate_model_params.rf_n_estimators,
            min_samples_split=evaluate_model_params.rf_min_samples_split,
            min_samples_leaf=evaluate_model_params.rf_min_samples_leaf,
            criterion=evaluate_model_params.rf_criterion,
            max_depth=evaluate_model_params.rf_max_depth,
            class_weight=evaluate_model_params.rf_class_weight
            )

        train_embeddings = train_embeddings.numpy()
        test_embeddings = test_embeddings.numpy()

        rf.fit(train_embeddings, y_train)

        # predictions
        y_train = y_train.numpy()
        y_test = y_test.numpy()

        y_pred = rf.predict(test_embeddings)
        y_pred_proba = rf.predict_proba(test_embeddings)[:,1]

        # scores
        accuracy, precision, recall, f1, ef01, ef05, roc_auc, pr_auc, mcc = calculate_stats(y_test, y_pred, y_pred_proba)

        return accuracy, precision, recall, f1, ef01, ef05, roc_auc, pr_auc, mcc


def calculate_stats(y_test, y_pred, y_pred_proba):
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)    
    roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)
    pr_auc = round(average_precision_score(y_test, y_pred_proba), 4)
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)
    ef01 = round(enrichment_factor(y_test, y_pred, fraction=0.01), 4)
    ef05 = round(enrichment_factor(y_test, y_pred, fraction=0.05), 4)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, pr_auc, mcc


def test_best_seed_model(model_name, statistics, best_seed_models, in_channels, hidden_dim, n_blocks,
                embedding_size, train_loader, n_train_samples, test_loader, evaluate_model_params, phase, device):

    for seed, seed_data in best_seed_models.items():

        set_seed(seed)

        for epoch, model_path in zip(seed_data["epoch"], seed_data["model_path"]):

            model, _ = load_model(model_path, model_name, in_channels, hidden_dim, n_blocks, embedding_size)
            model.to(device)
            n_test_samples = len(test_loader.dataset)

            train_embeddings, train_labels = generate_embeddings(model, train_loader, n_train_samples, embedding_size, device)
            test_embeddings, test_labels = generate_embeddings(model, test_loader, n_test_samples, embedding_size, device)

            test_stats, _ = evaluate_model(model, phase, train_embeddings, train_labels, test_embeddings, 
                                           test_labels, evaluate_model_params, phase)
            
            logging.info(f"Seed: {seed}, {test_stats}")

            statistics.upload_test_stats(test_stats, seed, epoch)

    return statistics


def test_model(model_path, model_name, model_in_channels, model_hidden_channels, model_n_blocks, embedding_length, 
               train_loader, n_train_samples, valid_loader, n_valid_samples, test_loader, n_test_samples, 
               evaluate_model_params, statistics, seed, device):
    
    set_seed(seed)
    train_loader.batch_sampler.shuffle_data()
    model, checkpoint = load_model(model_path, model_name, model_in_channels, model_hidden_channels, model_n_blocks, embedding_length)
    model.to(device)

    train_embeddings, train_labels = generate_embeddings(model, train_loader, n_train_samples, embedding_length, device)
    valid_embeddings, valid_labels = generate_embeddings(model, valid_loader, n_valid_samples, embedding_length, device)        
    test_embeddings, test_labels = generate_embeddings(model, test_loader, n_test_samples, embedding_length, device)

    valid_stats, _ = evaluate_model(model, "valid", train_embeddings, train_labels, valid_embeddings, 
                                    valid_labels, evaluate_model_params, "valid", seed)
    test_stats, _ = evaluate_model(model, "test", train_embeddings, train_labels, test_embeddings, 
                                    test_labels, evaluate_model_params, "test", seed)
    basic_stats = {"running_type": model_path.split("/")[-4], "seed": seed, "epoch": checkpoint["epoch"]}
    logging.info(f"{basic_stats}, {valid_stats}, {test_stats}")

    statistics.add(basic_stats, valid_stats | test_stats)
    
    return statistics