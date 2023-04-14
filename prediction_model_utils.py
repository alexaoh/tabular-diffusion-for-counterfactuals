# Utils for evaluating prediction models. 

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def make_confusion_matrix(y_test, predictions, text = "", show = False):
    """Make and plot confusion matrix for one prediction model."""
    labs = list(y_test.values)
    #predictions = np.where(predicted_probs > 0.5, 1, 0)

    cm = metrics.confusion_matrix(labs, list(predictions), labels = [0,1])
    conf_mat = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(1,1)
    conf_mat.plot(ax = ax, colorbar = False)
    ax.set_title(f"Confusion Matrix {text}")

    print(f"Classification metrics {text}:")
    print(metrics.classification_report(labs, predictions, labels = [0,1]))
    if show:
        plt.show()

def make_confusion_matrix_v2(y_test, predicted_probs_true_data, predicted_probs_synth, show = False):
    """Make and plot confusion matrix for models based on true and synthetic data."""
    labs = list(y_test.values)
    preds_true = predicted_probs_true_data.flatten() # These flattens are not necessary I believe.
    preds_synth = predicted_probs_synth.flatten()
    predicted_classes_true = np.where(preds_true > 0.5, 1, 0)
    predicted_classes_synth = np.where(preds_synth > 0.5, 1, 0)

    cm_true = metrics.confusion_matrix(labs, list(predicted_classes_true), labels = [0,1])
    conf_mat_true = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_true)

    cm_synth = metrics.confusion_matrix(labs, list(predicted_classes_synth), labels = [0,1])
    conf_mat_synth = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_synth)

    fig, ax = plt.subplots(1,2)
    conf_mat_true.plot(ax = ax[0], colorbar = False)
    ax[0].set_title("AD Real Data")
    conf_mat_synth.plot(ax = ax[1])
    ax[1].set_title("AD Synthetic Data")
    if show:
        plt.show()

    print("Some more classifaction statistics:")
    print(metrics.classification_report(labs, predicted_classes_true, labels = [0,1]))
    print(metrics.classification_report(labs, predicted_classes_synth, labels = [0,1]))

def calculate_auc_f1_acc(y_test, predicted_probs, show = False):
    """Calculate AUC and F1 for one prediction model."""
    labs = list(y_test.values)
    predictions = np.where(predicted_probs > 0.5, 1, 0)

    fpr, tpr, _ = metrics.roc_curve(labs, predicted_probs)
    auc = metrics.auc(fpr, tpr)

    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
                                    estimator_name='ROC')
    
    f1 = metrics.f1_score(labs, predictions, average = "macro") # Need to add the correct parameter here!

    acc = metrics.accuracy_score(labs, predictions, normalize=True)

    fig, ax = plt.subplots(1,1)
    display.plot(ax = ax)
    ax.set_title("Receiver Operating Characteristic (ROC)")
    if show:
        plt.show()

    return f1, auc, acc

def calculate_auc_f1_v2(y_test, predicted_probs_true_data, predicted_probs_synth, show = False):
    """Calculate metrics we want to use to compare ML efficacy with."""
    labs = list(y_test.values)
    preds_true = predicted_probs_true_data.flatten()
    preds_synth = predicted_probs_synth.flatten()
    predicted_classes_true = np.where(preds_true > 0.5, 1, 0)
    predicted_classes_synth = np.where(preds_synth > 0.5, 1, 0)

    fpr_true, tpr_true, _ = metrics.roc_curve(labs, preds_true)
    auc_true = metrics.auc(fpr_true, tpr_true)

    display_true = metrics.RocCurveDisplay(fpr=fpr_true, tpr=tpr_true, roc_auc=auc_true,
                                    estimator_name='AD Real Data')
    
    f1_true = metrics.f1_score(labs, predicted_classes_true)

    fpr_synth, tpr_synth, _ = metrics.roc_curve(labs, preds_synth)
    auc_synth = metrics.auc(fpr_synth, tpr_synth)

    display_synth = metrics.RocCurveDisplay(fpr=fpr_synth, tpr=tpr_synth, roc_auc=auc_synth,
                                    estimator_name='AD Synthetic Data')
    
    f1_synth = metrics.f1_score(labs, predicted_classes_synth, average = "macro") # Need to add the correct parameter here!

    fig, ax = plt.subplots(1,2)
    display_true.plot(ax = ax[0])
    display_synth.plot(ax = ax[1])
    ax[0].set_title("AD Real Data")
    ax[1].set_title("AD Synthetic Data")
    if show: 
        plt.show()

    return f1_true, auc_true, f1_synth, auc_synth
