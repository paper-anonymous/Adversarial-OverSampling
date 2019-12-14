import numpy as np


class MAHAKIL(object):
    def __init__(self, pfp=0.5):
        self.data_t = None
        self.pfp = pfp
        self.new = []

    # input  : data
    # return : data_new
    def fit_sample(self, data, label):
        data_t, data_f, label_t, label_f = [], [], [], []

        for i in range(label.shape[0]):
            if label[i] == 1:
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == 0:
                data_f.append(data[i])
                label_f.append(label[i])

        T = len(data_f) / (1 - self.pfp) - len(data_f)
        self.data_t = np.array(data_t)
        d = self.mahalanobis_distance(self.data_t)
        d.sort(key=lambda x: x[1], reverse=True)

        k = len(d)
        d_index = [d[i][0] for i in range(k)]
        data_t_sorted = [list(data_t[i]) for i in d_index]
        mid = k // 2
        bin1 = [data_t_sorted[i] for i in range(0, mid)]
        bin2 = [data_t_sorted[i] for i in range(mid, k)]
        parents = zip(bin1, bin2)

        if T > 0:
            count = 0
            while count <= T:
                parents = self.update_parents(parents)
                count = len(parents)

            temp = []
            for i in range(0, len(parents), 2):
                temp.append(parents[i][0])
                temp.append(parents[i][1])
                temp.append(parents[i+1][1])

            self.new = np.array(temp)
            train_new = np.append(data_f, self.new, axis=0)
            label_new = np.append(np.zeros(len(data_f)), np.ones(len(self.new)), axis=0)

        else:
            train_new = np.append(data_f, data_t, axis=0)
            label_new = np.append(np.zeros(len(data_f)), np.ones(len(data_t)), axis=0)

        return train_new, label_new

    def mahalanobis_distance(self, x):
        # x : [ndarray]
        mu = np.mean(x, axis=0)  # 均值
        d = []
        for i in range(x.shape[0]):
            x_mu = np.atleast_2d(x[i] - mu)
            s = self.cov(x)
            m = 10 ** -6
            d_squre = np.dot(np.dot(x_mu, np.linalg.inv(s + np.eye(s.shape[1]) * m)), np.transpose(x_mu))[0][0]
            d_tuple = (i, d_squre)
            d.append(d_tuple)
        return d

    @staticmethod
    def cov(x):
        s = np.zeros((x.shape[1], x.shape[1]))
        mu = np.mean(x, axis=0)
        for i in range(x.shape[0]):
            x_xbr = np.atleast_2d(x - mu)
            s_i = np.dot(np.transpose(x_xbr), x_xbr)
            s = s + s_i
        return np.divide(s, x.shape[0])

    def update_parents(self, parents):
        temp = []
        for i in parents:
            instance = [sum(e)/2.0 for e in zip(*i)]
            temp.append([i[0], instance])
            temp.append([instance, i[1]])
        return temp
