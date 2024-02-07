"""
Generate Sample() objects of grayscaled datasets and store them in pkl files in specified folder
"""
import dataloader
import numpy as np

def generateTrainSamples(dataset):
    samples = dataloader.loadNistTrainData("../data/" + dataset)

    images = []
    labels = []
    for index, sample in enumerate(samples):
        images.append(sample.getImage())
        labels.append(sample.getLabel())
        

    images = np.asarray(images)
    labels = np.asarray(labels)

    print("train images: ", images.shape)
    print("train labels: ", labels.shape)

    dataloader.dumpSplitData(images, labels, 10, "../data/" + dataset + "/train")

def generateTestSamples(dataset):
    samples = dataloader.loadNistTestData("../data/" + dataset)

    images = []
    labels = []
    for index, sample in enumerate(samples):
        images.append(sample.getImage())
        labels.append(sample.getLabel())
        

    images = np.asarray(images)
    labels = np.asarray(labels)

    print("test images: ", images.shape)
    print("test labels: ", labels.shape)

    dataloader.dumpSplitData(images, labels, 10, "../data/" + dataset + "/test")


if __name__ == "__main__":
    # generateTrainSamples("mnist")
    # generateTestSamples("mnist")

    generateTrainSamples("fashion")
    generateTestSamples("fashion")

    # generateTrainSamples("kmnist")
    # generateTestSamples("kmnist")