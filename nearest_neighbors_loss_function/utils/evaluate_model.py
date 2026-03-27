from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from skfp.metrics import enrichment_factor
from .generate_embeddings import generate_embeddings
from .checkpoints import load_model
import torch.nn.functional as F

def evaluate_model(model, train_loader, n_train_samples, test_loader, n_test_samples, embedding_length, 
         n_neighbors, stats_prefix, device):

    model.eval()
    train_embeddings, train_labels = generate_embeddings(model, train_loader, n_train_samples, embedding_length, device)
    test_embeddings, test_labels = generate_embeddings(model, test_loader, n_test_samples, embedding_length, device)

    train_embeddings = F.normalize(train_embeddings, dim=1)
    test_embeddings = F.normalize(test_embeddings, dim=1)

    # reshape to 1d
    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()

    train_distances = 2 - 2 * (train_distances @ train_distances.T)
    test_distances = 2 - 2 * (test_embeddings @ train_distances.T)

    accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = knn_stats(train_distances, test_distances, train_labels, test_labels, n_neighbors)

    # epoch_loss = round(running_loss / (data_id + 1), 5)
    test_stats = {f"{stats_prefix}_accuracy": accuracy, f"{stats_prefix}_precision": precision, f"{stats_prefix}_recall": recall, 
                  f"{stats_prefix}_f1": f1, f"{stats_prefix}_ef01": ef01, f"{stats_prefix}_ef05": ef05, 
                  f"{stats_prefix}_roc_auc": roc_auc, f"{stats_prefix}_mcc": mcc}

    return test_stats


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


def test_model(statistics, model_path, in_channels, hidden_dim, embedding_size, model_name, train_loader, 
               n_train_samples, test_loader, embedding_length, n_neighbors, device):
    
    model = load_model(model_path, in_channels, hidden_dim, embedding_size, model_name)
    n_test_samples = len(test_loader.dataset)

    test_stats = evaluate_model(model, train_loader, n_train_samples, test_loader, n_test_samples, embedding_length, 
         n_neighbors, "test", device)

    statistics.upload_test_stats(test_stats) 
    statistics.log_best_model_stats()

    return statistics