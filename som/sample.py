"""
This file contains code for loading one input sample
"""
import numpy as np
import tensorflow as tf

class Sample:   
    def __init__(self, label, sizeX, sizeY, values):
        self.label = label
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.values = tf.convert_to_tensor(values, dtype=tf.float32)

    def printSample(self):
        print("label:", self.label)
        print("sizeX:", self.sizeX)
        print("sizeY:", self.sizeY)
        print("values:")
        for row in self.values.numpy():
            print(row)

    def getStats(self):
        mean = tf.reduce_mean(self.values)
        std = tf.math.reduce_std(self.values)
        return (mean, std)

    def getSum(self):
        total = tf.reduce_sum(self.values)
        return (total)

    def getValues(self):
        return (self.values)

    def getLabel(self):
        return (self.label)

    def setLabel(self, label):
        self.label = label

    def write(self, file):
        file.write(str(self.label) + "\n")
        file.write(str(self.sizeX) + "\n")
        file.write(str(self.sizeY) + "\n")
        for row in self.values.numpy():
            file.write(" ".join([str(val) for val in row]) + "\n")

    def read(self, file):
        self.label = int(file.readline())
        self.sizeX = int(file.readline())
        self.sizeY = int(file.readline())
        self.values = []
        for _ in range(self.sizeX):
            row = [float(val) for val in file.readline().split()]
            self.values.append(row)
        self.values = tf.convert_to_tensor(self.values, dtype=tf.float32)

    def RMS_value(self, s):
        output = 0.0
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                output += tf.math.square(self.values[i][j] - s.values[i][j])

        output /= self.sizeX*self.sizeY
        output = tf.math.sqrt(output)
        return output.numpy()
