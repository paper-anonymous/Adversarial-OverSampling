# SMOTE Method
import random
import numpy as np


class SMOTE(object):
    def __init__(self, k=2, m=50, r=2):
        self.k = k  # Number of neighbors: [1,20]
        # Number of synthetic examples to create. Expressed as percent of final training data
        self.m = m  # [50, 100, 200, 400]
        self.r = r  # Power parameter for the Minkowski distance metric: [0.1, 5]

    def fit_sample(self, data, label):
        data_t, data_f, label_t, label_f = [], [], [], []

        N = label.shape[0]
        for i in range(N):
            if label[i] == 1:
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == 0:
                data_f.append(data[i])
                label_f.append(label[i])
        T = int(self.m /100 * N)
        num_minority = len(data_t)

        if self.k >= num_minority:
            self.k = num_minority - 1

        while len(label_f) > T and len(label_f) != 0:
            remove_index = random.randrange(0, len(label_f))
            data_f.pop(remove_index)
            label_f.pop(remove_index)

        new_sample_list, new_sample_num = [], 0
        while new_sample_num < T - len(label_t):
            current_index = random.randrange(0, len(label_t))
            new_sample_list.extend(self.something_like(data_t, data_t[current_index]))
            new_sample_num += self.k

        data_t.extend(new_sample_list)
        label_t = np.ones(len(data_t))
        label_f = np.zeros(len(data_f))
        data_t = np.array(data_t)
        data_f = np.array(data_f)

        if len(data_t.shape) == 1 or len(data_f.shape) == 1:
            print(data_t.shape)
            print(data_f.shape)

        data_new = np.row_stack((data_t, data_f))
        # data_new = np.vstack((data_t, data_f))
        label_new = np.append(label_t, label_f, axis=0)
        # np.append(data_t, data_f, axis=0), np.append(label_t, label_f, axis=0)
        return data_new, label_new

    def something_like(self, data, x0):
        relevant = []
        k1 = 0
        neighbors = self.found(data, x0)
        for neighbor in neighbors:
            bar_ab = neighbor - x0
            bar_ac = random.random() * bar_ab
            c = x0 + bar_ac
            relevant.append(c)
        return relevant

    def minkowski_distance(self, a, b):
        distance_r = 0
        for i in range(len(a)):
            distance_r += pow(abs(a[i]-b[i]), self.r)
        return pow(distance_r, 1/self.r)

    def found(self, data, x0):
        distances = []
        for i in range(len(data)):
            d = self.minkowski_distance(x0, data[i])
            d_tuple = (i, d)
            distances.append(d_tuple)
        distances.sort(key=lambda x: x[1])
        return [data[distances[i][0]] for i in range(1, int(self.k+1))]

