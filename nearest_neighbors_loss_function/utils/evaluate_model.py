from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from skfp.metrics import enrichment_factor
from .generate_embeddings import generate_embeddings
from .checkpoints import load_model
import torch.nn.functional as F
import torch

def evaluate_model(model, train_loader, n_train_samples, test_loader, n_test_samples, embedding_length, 
         n_neighbors, stats_prefix, device):

    model.eval()
    train_embeddings, train_labels = generate_embeddings(model, train_loader, n_train_samples, embedding_length, device)
    test_embeddings, test_labels = generate_embeddings(model, test_loader, n_test_samples, embedding_length, device)

    if torch.isnan(train_embeddings).any() or torch.isnan(test_embeddings).any():
        return None, True

    train_embeddings = F.normalize(train_embeddings, dim=1)
    test_embeddings = F.normalize(test_embeddings, dim=1)

    # reshape to 1d
    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()

    train_distances = 2 - 2 * (train_embeddings @ train_embeddings.T)
    test_distances = 2 - 2 * (test_embeddings @ train_embeddings.T)

    train_distances = train_distances.clamp(min=0)
    test_distances = test_distances.clamp(min=0)

    accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = knn_stats(train_distances, test_distances, train_labels, test_labels, n_neighbors)

    # epoch_loss = round(running_loss / (data_id + 1), 5)
    test_stats = {f"{stats_prefix}_accuracy": accuracy, f"{stats_prefix}_precision": precision, f"{stats_prefix}_recall": recall, 
                  f"{stats_prefix}_f1": f1, f"{stats_prefix}_ef01": ef01, f"{stats_prefix}_ef05": ef05, 
                  f"{stats_prefix}_roc_auc": roc_auc, f"{stats_prefix}_mcc": mcc}

    return test_stats, False


def knn_stats(train_distances, test_train_distances, y_train, y_test, n_neighbors):

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
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)    
    roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)
    ef01 = round(enrichment_factor(y_test, y_pred, fraction=0.01), 4)
    ef05 = round(enrichment_factor(y_test, y_pred, fraction=0.05), 4)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc


def test_model(model_name, statistics, best_seed_models, in_channels, hidden_dim, embedding_size, train_loader, 
               n_train_samples, test_loader, n_neighbors, stat_prefix, device):
    
    for seed, model_data in best_seed_models.items():
        
        model_path = model_data["model_path"]

        model, checkpoint = load_model(model_path, model_name, in_channels, hidden_dim, embedding_size)
        model.to(device)
        n_test_samples = len(test_loader.dataset)

        test_stats = evaluate_model(model, train_loader, n_train_samples, test_loader, n_test_samples, embedding_size, 
            n_neighbors, stat_prefix, device)

        statistics.upload_test_stats(test_stats, seed, checkpoint["epoch"]) 

    return statistics