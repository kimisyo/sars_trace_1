import argparse
import csv
import pandas as pd
import numpy as np
import gzip
import _pickle as cPickle

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.model_selection import permutation_test_score, StratifiedKFold


def calc_metrics_derived_from_confusion_matrix(metrics_name, y_true, y_predict):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_predict).ravel()
    #  PPV, precision
    #  TP / TP + FP
    if metrics_name in ["PPV", "precision"]:
        return tp / (tp + fp)

    #  NPV
    #  TN / TN + FN
    if metrics_name in ["NPV"]:
        return tn / (tn + fn)

    # sensitivity, recall, TPR
    #  TP / TP + FN
    if metrics_name in ["sensitivity", "recall", "TPR"]:
        return tp / (tp + fn)

    # specificity
    #  TN / TN + FP
    if metrics_name in ["specificity"]:
        return tn / (tn + fp)


def calc_metrics(metrics_name, y_true, y_predict):

    if metrics_name == "accuracy":
        return metrics.accuracy_score(y_true, y_predict)

    if metrics_name == "ba":
        return metrics.balanced_accuracy_score(y_true, y_predict)

    if metrics_name == "roc_auc":
        return metrics.roc_auc_score(y_true, y_predict)

    if metrics_name == "kappa":
        return metrics.cohen_kappa_score(y_true, y_predict)

    if metrics_name == "mcc":
        return metrics.matthews_corrcoef(y_true, y_predict)

    if metrics_name == "precision":
        return metrics.precision_score(y_true, y_predict)

    if metrics_name == "recall":
        return metrics.recall_score(y_true, y_predict)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str, required=True)
    parser.add_argument("-output_model", type=str, required=True)
    parser.add_argument("-output_report", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0)
    print(df.shape)

    y_train = df['outcome'].to_numpy()
    print(y_train)
    X_train = df.iloc[:, 1:]
    print(y_train.shape)
    print(X_train.shape)

    # Number of trees in random forest
    n_estimators = [100, 250, 500, 750, 1000]
    max_features = ['auto', 'sqrt']
    criterion = ['gini', 'entropy']
    class_weight = [None,'balanced',
                        {0:.9, 1:.1}, {0:.8, 1:.2}, {0:.7, 1:.3}, {0:.6, 1:.4},
                        {0:.4, 1:.6}, {0:.3, 1:.7}, {0:.2, 1:.8}, {0:.1, 1:.9}]
    random_state = [24]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'criterion': criterion,
                  'random_state': random_state,
                  'class_weight': class_weight}

    # setup model building
    rf = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1, cv=5, verbose=1)
    rf.fit(X_train, y_train)
    print()
    print('Best params: %s' % rf.best_params_)
    print('Score: %.2f' % rf.best_score_)

    rf_best = RandomForestClassifier(**rf.best_params_, n_jobs=-1)
    rf_best.fit(X_train, y_train)

    #一旦できたモデルを保存
    with gzip.GzipFile(args.output_model, 'w') as f:
        cPickle.dump(rf_best, f)

    with gzip.GzipFile(args.output_model, 'r') as f:
        rf_best = cPickle.load(f)

    # Params
    pred = []
    ad = []
    pred_proba = []
    index = []
    cross_val = StratifiedKFold(n_splits=5)

    # Do 5-fold loop
    for train_index, test_index in cross_val.split(X_train, y_train):
        fold_model = rf_best.fit(X_train.iloc[train_index], y_train[train_index])
        fold_pred = rf_best.predict(X_train.iloc[test_index])
        fold_ad = rf_best.predict_proba(X_train.iloc[test_index])

        pred.append(fold_pred)
        ad.append(fold_ad)
        pred_proba.append(fold_ad[:, 1])
        index.append(test_index)

    threshold_ad = 0.70

    # Prepare results to export
    fold_index = np.concatenate(index)
    fold_pred = np.concatenate(pred)
    fold_ad = np.concatenate(ad)
    fold_pred_proba = np.concatenate(pred_proba)

    fold_ad = (np.amax(fold_ad, axis=1) >= threshold_ad).astype(str)
    five_fold_morgan = pd.DataFrame({'Prediction': fold_pred, 'AD': fold_ad, 'Proba': fold_pred_proba}, index=list(fold_index))
    five_fold_morgan.AD[five_fold_morgan.AD == 'False'] = np.nan
    five_fold_morgan.AD[five_fold_morgan.AD == 'True'] = five_fold_morgan.Prediction
    five_fold_morgan.sort_index(inplace=True)
    five_fold_morgan['y_train'] = pd.DataFrame(y_train)

    # morgan stats
    all_datas = []
    datas = []
    datas.append(calc_metrics("accuracy", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics("ba", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics("roc_auc", five_fold_morgan['y_train'], five_fold_morgan['Proba']))
    datas.append(calc_metrics("kappa", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics("mcc", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics("precision", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics("recall", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics_derived_from_confusion_matrix("sensitivity", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics_derived_from_confusion_matrix("PPV", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics_derived_from_confusion_matrix("specificity", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))
    datas.append(calc_metrics_derived_from_confusion_matrix("NPV", five_fold_morgan['y_train'], five_fold_morgan['Prediction']))

    datas.append(1)
    all_datas.append(datas)

    # morgan AD stats
    morgan_ad = five_fold_morgan.dropna(subset=['AD'])
    morgan_ad['AD'] = morgan_ad['AD'].astype(int)
    coverage_morgan_ad = len(morgan_ad['AD']) / len(five_fold_morgan['y_train'])

    datas = []
    datas.append(calc_metrics("accuracy", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics("ba", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics("roc_auc", morgan_ad['y_train'], morgan_ad['Proba']))
    datas.append(calc_metrics("kappa", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics("mcc", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics("precision", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics("recall", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics_derived_from_confusion_matrix("sensitivity", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics_derived_from_confusion_matrix("PPV", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics_derived_from_confusion_matrix("specificity", morgan_ad['y_train'], morgan_ad['AD']))
    datas.append(calc_metrics_derived_from_confusion_matrix("NPV", morgan_ad['y_train'], morgan_ad['AD']))

    datas.append(coverage_morgan_ad)
    all_datas.append(datas)

    # print stats
    print('\033[1m' + '5-fold External Cross Validation Statistical Characteristcs of QSAR models developed morgan' + '\n' + '\033[0m')
    morgan_5f_stats = pd.DataFrame(all_datas, columns=["accuracy", "ba", "roc_auc", "kappa", "mcc", "precision", "recall",
                                                       "sensitivity", "PPV", "specificity", "NPV", "Coverage"])
    morgan_5f_stats.to_csv(args.output_report)
    print(morgan_5f_stats)

    # Y ramdomaization
    permutations = 20
    score, permutation_scores, pvalue = permutation_test_score(rf_best, X_train, y_train,
                                                           cv=5, scoring='balanced_accuracy',
                                                           n_permutations=permutations,
                                                           n_jobs=-1,
                                                           verbose=1,
                                                           random_state=24)
    print('True score = ', score.round(2),
          '\nY-randomization = ', np.mean(permutation_scores).round(2),
          '\np-value = ', pvalue.round(4))


if __name__ == "__main__":
    main()
