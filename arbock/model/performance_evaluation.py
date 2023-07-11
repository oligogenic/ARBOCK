import numpy as np
from numpy import sqrt, argmax
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, roc_auc_score
from sklearn.metrics._plot.roc_curve import RocCurveDisplay
import matplotlib.pyplot as plt
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


def compute_perf_metrics(y_true, y_pred_proba, positive_label=1, sample_weight=None):
    '''
    Compute performance metrics for a binary classification task
    :param y_true: list of true labels
    :param y_pred_proba: list of predicted probabilities
    :param positive_label: label of the positive class
    :param sample_weight: list of sample weights
    :return: dictionary of performance metrics
    '''
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    if sample_weight is not None:
        logger.info("Using sample_to_weight to evaluate performances")

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weight)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, sample_weight=sample_weight, pos_label=positive_label)
        pr_auc = auc(recall, precision)
    else:
        roc_auc = pr_auc = float("NaN")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def apply_model(model, y_test_samples, sample_to_weight, metapath_dict):
    '''
    Wrapper method applying the model to samples and returning all sort of results
    Refer to individual method documentation for more details
    :param model: model to apply
    :param y_test_samples: pandas Series of true labels for the test samples
    :param sample_to_weight: dictionary mapping samples to weights
    :param metapath_dict: dictionary mapping metapaths to their adjacency matrices
    :return: ordered_samples, sample_weights, predict_positive_probas, y_test, perf_metrics, explanations
    '''

    positive_samples = y_test_samples[y_test_samples == 1].index
    negative_samples = y_test_samples[y_test_samples == 0].index

    sample_to_class = {}
    for positive in positive_samples:
        sample_to_class[positive] = 1
    for negative in negative_samples:
        sample_to_class[negative] = 0

    sample_class_list = list(sample_to_class.items())
    samples = [s for s,c in sample_class_list]
    y_test = [c for s,c in sample_class_list]
    sample_weights = [sample_to_weight[s] for s,c in sample_class_list]

    ordered_samples, predict_probas, explanations = model.predict_and_explain(samples, metapath_dict)

    predict_positive_probas = predict_probas[:,1]
    perf_metrics = compute_perf_metrics(y_test, predict_positive_probas, positive_label=1, sample_weight=sample_weights)
    return ordered_samples, sample_weights, predict_positive_probas, y_test, perf_metrics, explanations


def xval_roc_curve(name, folds_y_true, folds_y_scores, folds_weights, ax, plot_color, display_extra_legends=True, display_variation=True, display_title=True, display_legend=True):
    '''
    Display the ROC curve for a cross-validation
    :param name: name of the model
    :param folds_y_true: list of true labels for each fold
    :param folds_y_scores: list of predicted probabilities for each fold
    :param folds_weights: list of sample weights for each fold
    :param ax: matplotlib axis to plot on
    :param plot_color: color to use for the plot
    :param display_extra_legends: whether to display extra legends
    :param display_variation: whether to display the variation of the curve
    :param display_title: whether to display the title
    :param display_legend: whether to display the legend
    '''
    mean_fpr = np.linspace(0, 1, 1000)
    aucs = []
    interp_tprs = []
    all_preds = []
    all_y = []
    all_tpr = []
    all_fpr = []
    all_thresholds = []

    i = 0

    # Getting and plotting curve for every fold
    for y_true, y_scores, weights in zip(folds_y_true, folds_y_scores, folds_weights):
        fprs, tprs, thresholds = roc_curve(y_true, y_scores, sample_weight=weights, drop_intermediate=True)
        all_tpr.append(tprs)
        all_fpr.append(fprs)
        all_thresholds.append(thresholds)
        all_preds.extend(y_scores)
        all_y.extend(y_true)
        roc_auc = auc(fprs, tprs)

        viz = RocCurveDisplay(
            fpr=fprs,
            tpr=tprs,
            roc_auc=roc_auc,
            estimator_name=f"fold_{i}",
            pos_label=1
        )

        if display_variation:
            viz.plot(ax=ax, alpha=0.1, lw=1, color=plot_color, label=None)

        interp_tpr = np.interp(mean_fpr, fprs, tprs)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        aucs.append(roc_auc)
        i += 1

    # Estimating and plotting the optimal ROC threshold
    fpr, tpr, thr = roc_curve(all_y, all_preds)
    best_thr_ix = argmax(sqrt(tpr * (1 - fpr)))
    best_thr = thr[best_thr_ix]
    best_tpr, best_fpr = mean_best_roc_rates(all_tpr, all_fpr, mean_fpr)
    ba = (best_tpr + (1 - best_fpr)) / 2
    x, y = best_fpr, best_tpr
    ax.scatter(x, y, marker='o', color=plot_color, label="Optimal threshold = %0.3f " % (best_thr,))
    print("Best threshold: %0.3f (TPRs %0.3f ; FPRs %0.3f, BA %0.3f)" % (best_thr, best_tpr, best_fpr, ba))

    arrowprops = {'arrowstyle': '-', 'ls': '--', 'color': "gray"}
    ax.annotate("%0.2f" % x, xy=(x, y), xytext=(x, -0.06),
                textcoords=ax.get_xaxis_transform(),
                arrowprops=arrowprops,
                va='top', ha='center', color="gray")
    ax.annotate("%0.2f" % y, xy=(x, y), xytext=(-0.06, y),
                textcoords=ax.get_yaxis_transform(),
                arrowprops=arrowprops,
                va='center', ha='right', color="gray")

    # Show mean ROC curve + std
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print("Mean ROC avg AUC = %0.3f  ; std. %0.2f)" % (mean_auc, std_auc))

    ax.plot(mean_fpr, mean_tpr, color=plot_color,
            label=r'%s (AUC = %0.3f $\pm$ %0.2f)' % (name, mean_auc, std_auc * 1.96),
            lw=2, alpha=1)

    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Display chance line
    chance_label = "Chance" if display_extra_legends else None
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.3, color='firebrick', label=chance_label, alpha=.8)

    # Display legend / titles / etc...
    if display_variation:
        std_label = r'$\pm$ 1 std. dev.' if display_extra_legends else None
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.15, label=std_label)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', size=14, labelpad=7)
    ax.set_ylabel('True Positive Rate', size=14, labelpad=14)
    if display_legend:
        ax.legend(loc="lower right")
    if display_title:
        ax.set_title("Receiver Operating Characteristic", size=14)

    return best_thr


def xval_pr_curve(name, folds_y_true, folds_y_scores, folds_weights, ax, plot_color, display_extra_legends=True, display_variation=True, display_title=True, display_legend=True):
    '''
    Plot the PR curve for a given model, using the given folds.
    :param name: name of the model
    :param folds_y_true: list of y_true arrays for each fold
    :param folds_y_scores: list of y_scores arrays for each fold
    :param folds_weights: list of weights arrays for each fold
    :param ax: matplotlib axis to plot on
    :param plot_color: color to use for the plot
    :param display_extra_legends: whether to display extra legends (e.g. std. dev.)
    :param display_variation: whether to display the variation of the ROC curves
    :param display_title: whether to display the title
    :param display_legend: whether to display the legend
    :return: the optimal threshold
    '''
    pr_aucs = []

    recall_array = np.linspace(0, 1, 1000)
    precision_arrays = []

    i = 0
    y_1_len = []
    y_len = []
    for y_true, y_scores, weights in zip(folds_y_true, folds_y_scores, folds_weights):
        y_1_len.append(len(y_true[y_true == 1]))
        y_len.append(len(y_true))
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores, sample_weight=weights, pos_label=1)

        precision_fold, recall_fold, thresh = precisions[::-1], recalls[::-1], thresholds[::-1]  # reverse order of results
        thresh = np.insert(thresh, 0, 1.0)
        precision_array = np.interp(recall_array, recall_fold, precision_fold)
        precision_arrays.append(precision_array)
        pr_auc = auc(recall_array, precision_array)
        pr_aucs.append(pr_auc)

        if display_variation:
            plt.plot(recall_fold, precision_fold, alpha=0.1, label=None, color=plot_color)

        i += 1

    mean_precision = np.mean(precision_arrays, axis=0)
    mean_auc = auc(recall_array, mean_precision)
    std_auc = np.std(pr_aucs)
    baseline = np.mean(y_1_len) / np.mean(y_len)

    print('Mean PR (avg AUC = %0.3f ; std %0.2f)' % (mean_auc, std_auc))


    ax.plot(recall_array, mean_precision, color=plot_color,
            label=r'%s (AUC = %0.3f $\pm$ %0.2f)' % (name, mean_auc, std_auc * 1.96),
            lw=2, alpha=1)

    std_tpr = np.std(precision_arrays, axis=0)
    tprs_upper = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower = np.maximum(mean_precision - std_tpr, 0)

    chance_label = "Chance" if display_extra_legends else None
    ax.plot([0, 1], [baseline, baseline], linestyle='--', lw=1.3, color='firebrick', label=chance_label, alpha=.8)

    if display_variation:
        std_label = r'$\pm$ 1 std. dev.' if display_extra_legends else None
        ax.fill_between(recall_array, tprs_lower, tprs_upper, color='grey', alpha=.15, label=std_label)
    if display_legend:
        ax.legend(loc="upper right")

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall', size=14, labelpad=7)
    ax.set_ylabel('Precision', size=14, labelpad=14)
    if display_title:
        ax.set_title("Precision-Recall Curve", size=14)


def mean_best_roc_rates(all_tpr, all_fpr, mean_fpr):
    '''
    Compute the mean best TPR and FPR for a given set of ROC curves, based on the geometric mean
    :param all_tpr: list of TPR arrays for each fold
    :param all_fpr: list of FPR arrays for each fold
    :param mean_fpr: mean FPR array
    :return: the mean best TPR and FPR
    '''
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
    best_ix = argmax(mean_tpr * (1 - mean_fpr))
    best_tpr = mean_tpr[best_ix]
    best_fpr = mean_fpr[best_ix]
    return best_tpr, best_fpr
