import sys
sys.dont_write_bytecode = True


class Counter(object):
    def __init__(self, actual, predict):
        self.actual = actual
        self.predict = predict
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0

        for a, b in zip(self.actual, self.predict):
            if a == 1 and b == 1:
                self.TP += 1

            elif a == 0 and b == 0:
                self.TN += 1

            elif a == 0 and b == 1:
                self.FP += 1

            elif a == 1 and b == 0:
                self.FN += 1

    def stats(self):
        Rec = self.TP / (self.TP + self.FN + 1e-10)  # Sensitivity, Recall
        Spec = self.TN / (self.FP + self.TN + 1e-10)
        Prec = self.TP / (self.TP + self.FP + 1e-10)
        acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
        pf = self.FP / (self.FP + self.TN + 1e-10)  # False Positive Rate
        F1_measure = 2 * (Prec * Rec) / (Prec + Rec + 1e-10)
        F2_measure = (4 + 1) * (Prec * Rec) / (4 * Prec + Rec + 1e-10)
        G_measure = 2 * Rec * Spec / (Rec + Spec + 1e-10)
        Bal_measure = 1 - (((0 - pf) ** 2 + (1 - Rec) ** 2)/2) ** 0.5
        return Rec, pf, Spec, Prec, acc, F1_measure, F2_measure, G_measure, Bal_measure


class Metrics(object):
    """
    Statistics Stuff, confusion matrix, all that jazz...
    """

    def __init__(self, actual, prediction):
        self.actual = actual
        self.prediction = prediction

    def __call__(self):
        yield Counter(self.actual, self.prediction)











