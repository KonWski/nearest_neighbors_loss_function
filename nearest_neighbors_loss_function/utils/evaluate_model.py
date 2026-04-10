from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from skfp.metrics import enrichment_factor
from .generate_embeddings import generate_embeddings
from .checkpoints import load_model
import torch.nn.functional as F
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import logging


def evaluate_model(model, evaluation_model_name, train_loader, n_train_samples, test_loader, n_test_samples, embedding_length, 
         n_neighbors, stats_prefix, device):

    model.eval()
    train_embeddings, train_labels = generate_embeddings(model, train_loader, n_train_samples, embedding_length, device)
    test_embeddings, test_labels = generate_embeddings(model, test_loader, n_test_samples, embedding_length, device)

    if torch.isnan(train_embeddings).any() or torch.isnan(test_embeddings).any():
        return None, True

    # reshape to 1d
    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()

    if evaluation_model_name == "knn":
        accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = knn_stats(train_embeddings, test_embeddings, train_labels, test_labels, n_neighbors)
    elif evaluation_model_name == "svc":
        accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = svc_stats(train_embeddings, test_embeddings, train_labels, test_labels)
    elif evaluation_model_name == "linear_svc":
        accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = linear_svc_stats(train_embeddings, test_embeddings, train_labels, test_labels)
    elif evaluation_model_name == "reg_log":
        accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = reg_log_stats(train_embeddings, test_embeddings, train_labels, test_labels)
    else:
        raise Exception(f"Unimplemented evaluation method: {evaluation_model_name}")

    # epoch_loss = round(running_loss / (data_id + 1), 5)
    test_stats = {f"{stats_prefix}_accuracy": accuracy, f"{stats_prefix}_precision": precision, f"{stats_prefix}_recall": recall, 
                  f"{stats_prefix}_f1": f1, f"{stats_prefix}_ef01": ef01, f"{stats_prefix}_ef05": ef05, 
                  f"{stats_prefix}_roc_auc": roc_auc, f"{stats_prefix}_mcc": mcc}

    return test_stats, False



def knn_stats(train_embeddings, test_embeddings, y_train, y_test, n_neighbors):

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
    y_train = y_train.numpy()
    y_test = y_test.numpy()

    # fit model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1, metric="precomputed")
    knn.fit(train_distances, y_train)

    # predictions
    y_pred = knn.predict(test_train_distances)
    y_pred_proba = knn.predict_proba(test_train_distances)[:,1]

    # scores
    accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = calculate_stats(y_test, y_pred, y_pred_proba)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc


def svc_stats(train_embeddings, test_embeddings, y_train, y_test):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
    ])

    # fit model
    model.fit(train_embeddings, y_train)

    # predictions
    y_pred = model.predict(test_embeddings)
    y_pred_proba = model.predict_proba(test_embeddings)[:,1]

    # scores
    accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = calculate_stats(y_test, y_pred, y_pred_proba)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc


def linear_svc_stats(train_embeddings, test_embeddings, y_train, y_test):

    base_model = LinearSVC(C=1.0, max_iter=10000)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("calibrated", CalibratedClassifierCV(base_model, method="sigmoid"))
    ])

    # fit model
    model.fit(train_embeddings, y_train)

    # predictions
    y_pred = model.predict(test_embeddings)
    y_pred_proba = model.predict_proba(test_embeddings)[:,1]

    # scores
    accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = calculate_stats(y_test, y_pred, y_pred_proba)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc


def reg_log_stats(train_embeddings, test_embeddings, y_train, y_test):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.2,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000, 
            n_jobs=-1
        ))
    ])

    # fit model
    model.fit(train_embeddings, y_train)

    # predictions
    y_pred = model.predict(test_embeddings)
    y_pred_proba = model.predict_proba(test_embeddings)[:,1]

    # scores
    accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = calculate_stats(y_test, y_pred, y_pred_proba)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc


def calculate_stats(y_test, y_pred, y_pred_proba):
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)    
    roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)
    ef01 = round(enrichment_factor(y_test, y_pred, fraction=0.01), 4)
    ef05 = round(enrichment_factor(y_test, y_pred, fraction=0.05), 4)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc


def test_model(model_name, evaluation_model_name, statistics, best_seed_models, in_channels, hidden_dim, embedding_size, train_loader, 
               n_train_samples, test_loader, n_neighbors, stat_prefix, device):
    
    for seed, model_data in best_seed_models.items():
        
        model_path = model_data["model_path"]

        model, checkpoint = load_model(model_path, model_name, in_channels, hidden_dim, embedding_size)
        model.to(device)
        n_test_samples = len(test_loader.dataset)

        test_stats, _ = evaluate_model(model, evaluation_model_name, train_loader, n_train_samples, test_loader, n_test_samples, embedding_size, 
            n_neighbors, stat_prefix, device)
        
        logging.info(f"Seed: {seed}, {test_stats}")

        statistics.upload_test_stats(test_stats, seed, checkpoint["epoch"]) 

    return statistics