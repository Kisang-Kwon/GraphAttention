#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

from sklearn.metrics import roc_curve, auc, confusion_matrix


def Stats(class_scores, labels, checkpoint, prefix):
    class_prob = tf.nn.softmax(class_scores)

    cm = confusion_matrix(labels[:, 1], class_prob[:, 1] > 0.5)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    filename = os.path.join(checkpoint, f'{prefix}_Confusion_matrix.png')
    plt.savefig(filename, bbox_inches='tight')
    
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    acc = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN )
    f1 = (2 * precision * recall) / (precision + recall)

    return TP, TN, FP, FN, acc, precision, recall, f1

def get_auc(class_probs, labels):
    fpr, tpr, thresholds = roc_curve(labels[:, 1], class_probs[:, 1])
    AUC = auc(fpr, tpr)

    return AUC

def ROC_curve(class_probs, labels, checkpoint, prefix):
    fpr, tpr, thresholds = roc_curve(labels[:, 1], class_probs[:, 1])
    AUC = auc(fpr, tpr)
    
    fig1 = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (auc = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    
    filename = os.path.join(checkpoint, f'{prefix}_ROC_curve.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig1)

    return AUC

def loss_plot(loss_dict, checkpoint, prefix, epochs):
    fig1 = plt.figure()
    plt.plot(range(1, len(loss_dict['tr'])+1), loss_dict['tr'])
    plt.plot(range(1, len(loss_dict['va'])+1), loss_dict['va'])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.xlim(0, epochs)
    plt.ylim(0.0, 5.0)

    filename = os.path.join(checkpoint, f'{prefix}_training_loss.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig1)

def accuracy(class_scores, labels):
    class_scores = tf.convert_to_tensor(class_scores)
    labels = tf.convert_to_tensor(labels)

    class_prob = tf.nn.softmax(class_scores)
    pred = class_prob[:, 1] > 0.5

    T = 0
    F = 0
    for i, label in enumerate(labels[:, 1].numpy()):
        if label == pred[i]:
            T += 1
        else:
            F += 1
    
    accuracy = T / (T + F)
    print(T, F, round(accuracy, 4))
    return accuracy


def accuracy_plot(acc_dict, checkpoint, prefix, epochs):
    fig1 = plt.figure()
    plt.plot(range(1, len(acc_dict['tr'])+1), acc_dict['tr'])
    plt.plot(range(1, len(acc_dict['va'])+1), acc_dict['va'])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy plot')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.xlim(0, epochs)
    plt.ylim(0.0, 1.0)

    filename = os.path.join(checkpoint, f'{prefix}_training_accuracy.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig1)


def prediction(filepath, PIDs, CIDs, classification, class_prob, smi_list, labels=None):
    out_predict = open(filepath, 'w')
    
    if labels is None:
        T = []
        F = []
        for i, prediction in enumerate(classification):
            if prediction == 1:
                T.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Active,{smi_list[i]}')
            elif prediction == 0:
                F.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Inactive,{smi_list[i]}')
        
        out_predict.write('PID,CID,Class_prob,Prediction,SMILES\n')
        out_predict.write('\n'.join(T) + '\n')
        out_predict.write('\n'.join(F))
    
    else:
        TP = []
        TN = []
        FP = []
        FN = []
        for i, prediction in enumerate(classification):
            if prediction == 1:
                if tf.argmax(labels[i]) == 1:
                    TP.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Active,TP,{smi_list[i]}')
                elif tf.argmax(labels[i]) == 0:
                    FN.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Active,FP,{smi_list[i]}')
            elif prediction == 0:
                if tf.argmax(labels[i]) == 1:
                    FN.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Inactive,FN,{smi_list[i]}')
                elif tf.argmax(labels[i]) == 0:
                    TN.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Inactive,TN,{smi_list[i]}')

        output = []
        output.extend(TP)
        output.extend(TN)
        output.extend(FP)
        output.extend(FN)

        out_predict.write('PID,CID,Class_prob,Prediction,Result,SMILES\n')
        out_predict.write('\n'.join(output))

    out_predict.close()