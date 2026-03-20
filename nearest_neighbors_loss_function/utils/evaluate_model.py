from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from skfp.metrics import enrichment_factor
from nearest_neighbors_loss_function.utils.generate_embeddings import generate_embeddings
from nearest_neighbors_loss_function.utils.checkpoints import load_model

def evaluate_model(model, train_loader, n_train_samples, test_loader, n_test_samples, embedding_length, 
         n_neighbors, stats_prefix, device):

    model.eval()
    train_embeddings, train_labels = generate_embeddings(model, train_loader, n_train_samples, embedding_length, device)
    test_embeddings, test_labels = generate_embeddings(model, test_loader, n_test_samples, embedding_length, device)

    # reshape to 1d
    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()

    accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc = knn_stats(train_embeddings, test_embeddings, train_labels, test_labels, n_neighbors)

    # epoch_loss = round(running_loss / (data_id + 1), 5)
    test_stats = {f"{stats_prefix}_accuracy": accuracy, f"{stats_prefix}_precision": precision, f"{stats_prefix}_recall": recall, 
                  f"{stats_prefix}_f1": f1, f"{stats_prefix}_ef01": ef01, f"{stats_prefix}_ef05": ef05, 
                  f"{stats_prefix}_roc_auc": roc_auc, f"{stats_prefix}_mcc": mcc}

    return test_stats


def knn_stats(X_train, X_test, y_train, y_test, n_neighbors):

    # fit model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # predictions
    y_pred = knn.predict(X_test)

    # scores
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    roc_auc = round(roc_auc_score(y_test, y_pred), 4)
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)

    ef01 = round(enrichment_factor(y_test, y_pred, fraction=0.01), 4)
    ef05 = round(enrichment_factor(y_test, y_pred, fraction=0.05), 4)

    return accuracy, precision, recall, f1, ef01, ef05, roc_auc, mcc


def test_model(statistics, model_path, in_channels, hidden_dim, embedding_size, model_name, train_loader, 
               n_train_samples, test_loader, embedding_length, n_neighbors, device):
    
    model = load_model(model_path, in_channels, hidden_dim, embedding_size, model_name)
    n_test_samples = len(test_loader)

    test_stats = evaluate_model(model, train_loader, n_train_samples, test_loader, n_test_samples, embedding_length, 
         n_neighbors, "test", device)

    statistics.upload_test_stats(test_stats) 
    statistics.log_best_model_stats()
       
    return statistics