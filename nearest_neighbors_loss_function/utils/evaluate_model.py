from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from skfp.metrics import enrichment_factor


def evaluate_model(X_train, X_test, y_train, y_test, n_neighbors):

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
    ef10 = round(enrichment_factor(y_test, y_pred, fraction=0.1), 4)
    ef15 = round(enrichment_factor(y_test, y_pred, fraction=0.15), 4)
    ef20 = round(enrichment_factor(y_test, y_pred, fraction=0.2), 4)

    return accuracy, precision, recall, f1, ef01, ef05, ef10, ef15, ef20, roc_auc, mcc