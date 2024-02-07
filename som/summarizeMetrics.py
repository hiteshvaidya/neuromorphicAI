"""
testClass.py

description: Use CosineDistanceLayer for inference
version: 1.0
author: Manali Dangarikar

"""

# import libraries
# import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
# import util

def getBWT(accuracies):
    bwt = 0
    for i in range(accuracies.shape[0]-1):
        bwt += accuracies[-1, i] - accuracies[i, i]
    bwt /= (accuracies.shape[0]-1)
    return bwt

def getFWT(accuracies, b):
    fwt = 0
    for i in range(1, accuracies.shape[0]):
        fwt += accuracies[i-1, i] - b[i]
    fwt /= (accuracies.shape[0] - 1)

    return fwt

def getAverageAccuracy(accuracies):
    avgAccuracy = 0
    avgAccuracy = float(np.sum(accuracies[-1, :]) / accuracies.shape[0])
    return avgAccuracy 

def getLearningAccuracy(accuracies):
    value = float(np.trace(accuracies) / accuracies.shape[0])
    return value

def getForgettingMeasure(accuracies):
    values = accuracies[-1, :] - np.max(accuracies, axis=0)
    forgettingMeasure = float(np.sum(np.abs(values)) / accuracies.shape[0])
    return forgettingMeasure


if __name__ == "__main__":
    parent_path = "./logs/domain_incremental/"
    version = ['mnist', 'kmnist', 'fashion', 'cifar-10']
    folder = "contSOM-trial-" # "cifar-green-trial-"  

    task_matrix = []
    metrics = {"aa":[], "fm":[], "la":[], "bwt":[]}
    for t in range(10):
        path = os.path.join(parent_path, version[1], folder + str(t))
        df = pd.read_csv(os.path.join(path, "transfer_metrics.csv"), header=None).to_numpy()
        task_matrix.append(df.copy())
        df *= 100
        metrics["aa"].append(getAverageAccuracy(df))
        metrics["la"].append(getLearningAccuracy(df))
        metrics["fm"].append(getForgettingMeasure(df))
        metrics["bwt"].append(getBWT(df))

    task_matrix = np.asarray(task_matrix)
    print("task_matrix: ", task_matrix.shape)
    mean_matrix = np.mean(task_matrix, axis=0)
    mean_matrix = np.around(mean_matrix * 100, decimals=2)
    std_matrix = np.std(task_matrix, axis=0)
    std_matrix = np.around(std_matrix * 100, decimals=2)

    for r in range(mean_matrix.shape[0]):
        for c in range(std_matrix.shape[1]-1):
            print(f"{mean_matrix[r,c]} $\pm$ {std_matrix[r,c]}", end=' & ')
        print(f"{mean_matrix[r,-1]} $\pm$ {std_matrix[r,-1]}", end=' \\\\')
        print()

    mean_bwt = getBWT(mean_matrix)
    mean_avgAccuracy = getAverageAccuracy(mean_matrix)
    mean_la = getLearningAccuracy(mean_matrix)
    mean_fm = getForgettingMeasure(mean_matrix)

    std_bwt = getBWT(std_matrix)
    std_avgAccuracy = getAverageAccuracy(std_matrix)
    std_la = getLearningAccuracy(std_matrix)
    std_fm = getForgettingMeasure(std_matrix)

    print("-------------- mean of all metrics -------------")
    aa_mean = np.mean(metrics['aa'])
    aa_std = np.std(metrics['aa'])
    print(f"Average accuracy: mean = {aa_mean}, std = {aa_std}")
    la_mean = np.mean(metrics["la"])
    la_std = np.std(metrics["la"])
    print(f"Learning Accuracy: mean = {la_mean}, std = {la_std}")
    fm_mean = np.mean(metrics["fm"])
    fm_std = np.std(metrics["fm"])
    print(f"Forgetting Measure: mean = {fm_mean}, std = {fm_std}")
    bwt_mean = np.mean(metrics["bwt"])
    bwt_std = np.std(metrics["bwt"])
    print(f"BWT: mean = {bwt_mean}, std = {bwt_std}")

    print("--------------- metrics of all means ----------------")
    print(f"Average accuracy: mean = {mean_avgAccuracy}, std = {std_avgAccuracy}")
    print(f"Learning Accuracy: mean = {mean_la}, std = {std_la}")
    print(f"Forgetting Measure: mean = {mean_fm}, std = {std_fm}")
    print(f"BWT: mean = {mean_bwt}, std = {std_bwt}")

    # print("mean:\n")
    # print(tabulate(mean_matrix, tablefmt="latex"))

    # print("\nstd:\n")
    # print(tabulate(std_matrix, tablefmt="latex"))