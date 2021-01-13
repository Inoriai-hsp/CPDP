import numpy as np
from sklearn.model_selection import train_test_split
from Algorithms.DTB import *
from Algorithms.domainAdaptation import *
from sklearn.metrics import roc_auc_score
from Algorithms.Classifier import *
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from Utils.helper import *


class TCSBoost(object):

    def __init__(self, Xs, Ys, Xt, Yt, TCSBoost_lamb=0.6, TCSBoost_c=0.5, iter=30, clf='RF',
                 n_estimators=10, criterion='gini', max_features='auto', RFmin_samples_split=2,     # RF
                 Boostnestimator=50, BoostLearnrate=1,                                              # Boost
                 CARTsplitter='best',                                                               # CART
                 Ridgealpha=1, Ridgenormalize=False,                                                # Ridge
                 NBtype='gaussian',
                 SVCkernel='poly', C=1, degree=3, coef0=0.0, SVCgamma=1
                 ):
        self.Xsource = np.asarray(Xs)
        self.Ysource = np.asarray(Ys)
        self.Xtarget = np.asarray(Xt)
        self.Ytarget = np.asarray(Yt)
        self.lamb = TCSBoost_lamb
        self.c = TCSBoost_c
        self.N = iter
        self.clfType = clf

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.RFmin_samples = RFmin_samples_split
        self.Boostne = Boostnestimator
        self.BoostLearnrate = BoostLearnrate
        self.NBType = NBtype
        self.CARTsplitter = CARTsplitter
        self.Ridgealpha = Ridgealpha
        self.Ridgenormalize = Ridgenormalize
        self.SVCkernel = SVCkernel
        self.coef0 = coef0
        self.gamma = SVCgamma
        self.degree = degree
        self.C = C


    def _max_min(self, x):
        shape = np.asarray(x).shape
        Max = np.zeros(shape[1])
        Min = np.zeros(shape[1])
        for i in range(0, shape[1]):
            a = x[:, i]
            Max[i] = np.max(a)
            Min[i] = np.min(a)

        return Max, Min


    def similarity_weight(self, x):
        k = self.Xtarget.shape[1]
        Max, Min = self._max_min(self.Xtarget)
        S = np.zeros(x.shape[0])
        for i in range(0, x.shape[0]):
            s = 0
            for j in range(0, k):
                if x[i][j] >= Min[j] and x[i][j] <= Max[j]:
                    s = s + 1
            w = s / k
            S[i] = w
        return S

    def resampling_smote(self, S):
        dissimilar = np.where(S != 1)[0]
        minority = np.where(self.Ysource == 1)[0]
        s = []
        # 移除少数类中similarity不为1的实例
        for i in range(0, len(minority)):
            if minority[i] in dissimilar:
                s.append(minority[i])
        if len(s) == len(minority):
            return
        Xs = np.delete(self.Xsource, s, 0)
        Ys = np.delete(self.Ysource, s, 0)
        smote = SMOTE()
        Xs, Ys = smote.fit_resample(Xs, Ys)
        self.Xsource = np.concatenate((Xs, self.Xsource[s, :]), axis=0)
        self.Ysource = np.concatenate((Ys, self.Ysource[s]), axis=0)


    def resampling_tomek(self, S):
        dissimilar = np.where(S != 1)[0]
        minority = np.where(self.Ysource == 1)[0]
        s = []
        # 移除少数类中similarity不为1的实例
        for i in range(0, len(minority)):
            if minority[i] in dissimilar:
                s.append(minority[i])
        if len(s) == len(minority):
            return
        Xs = np.delete(self.Xsource, s, 0)
        Ys = np.delete(self.Ysource, s, 0)
        tomeklinks = TomekLinks()
        Xs, Ys = tomeklinks.fit_resample(Xs, Ys)
        self.Xsource = np.concatenate((Xs, self.Xsource[s, :]), axis=0)
        self.Ysource = np.concatenate((Ys, self.Ysource[s]), axis=0)
        

    def fit(self):
        self.Ysource[np.where(self.Ysource == 0)] = -1
        self.Ytarget[np.where(self.Ytarget == 0)] = -1

        S = self.similarity_weight(self.Xsource)
        self.resampling_smote(S)
        for i in range(0, 5):
            S = self.similarity_weight(self.Xsource)
            self.resampling_tomek(S)


        if self.clfType == 'RF':
            self.m = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                            max_features=self.max_features, min_samples_split=self.RFmin_samples)
        if self.clfType == 'SVM':
            self.m = SVC(kernel=self.SVCkernel, C=self.C, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        if self.clfType == 'Boost':
            self.m = AdaBoostClassifier(n_estimators=self.Boostne, learning_rate=self.BoostLearnrate)
        if self.clfType == 'NB':
            if self.NBType == 'gaussian':
                self.m = GaussianNB()
            elif self.NBType == 'multinomial':
                self.m = MultinomialNB()
            elif self.NBType == 'bernoulli':
                self.m = BernoulliNB()
        if self.clfType == 'CART':
            self.m = DecisionTreeClassifier(criterion=self.criterion, splitter=self.CARTsplitter, max_features=self.max_features, min_samples_split=self.RFmin_samples)
        if self.clfType == 'Ridge':
            self.m = RidgeClassifier(alpha=self.Ridgealpha, normalize=self.Ridgenormalize)

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.Xtarget, self.Ytarget, test_size=0.3)
        while len(np.unique(self.testY)) <= 1:
            self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.Xtarget, self.Ytarget, test_size=0.3)

        trans_data = np.concatenate((self.Xsource, self.trainX), axis=0)
        trans_label = np.concatenate((self.Ysource, self.trainY), axis=0)

        row_trans = trans_data.shape[0]
        row_T = self.testX.shape[0]
        lamb = self.lamb
        c = self.c
        N = self.N
        SW = self.similarity_weight(trans_data)

        test_data = np.concatenate((trans_data, self.testX), axis=0)

        # 初始化权重
        weights = np.ones([row_trans, 1]) / row_trans

        # 防止除数为零
        # if N == 0 or (1 + np.sqrt(2 * np.log(row_A / N))) == 0:
        #     self.error = 1
        #     return
        # bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

        # 存储每次迭代的标签和bata值？
        alpha = np.zeros(N)
        result_label = np.ones([row_trans + row_T, N])

        predict = np.zeros([row_T])

        # print('params initial finished.')
        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')
        test_data = np.asarray(test_data, order='C')

        for i in range(N):

            result_label[:, i] = self.train_classify(trans_data, trans_label,
                                                     test_data, weights)
            # print('result,', result_label[:, i], row_A, row_S, i, result_label.shape)

            error_rate = self.calculate_error_rate(trans_label, result_label[0:row_trans, i],
                                                   weights[0:row_trans, :])
            # print('Error rate:', error_rate)
            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                N = i
                break  # 防止过拟合
                # error_rate = 0.001

            alpha[i] = lamb * np.log((1 - error_rate) / error_rate)

            # 调整样本权重
            for j in range(row_trans):
                if SW[j] == 1:
                    if trans_label[j] == result_label[j][i]:
                        beta = -0.25 * c + 0.25
                    else:
                        beta = 0.25 * c
                else:
                    if trans_label[j] == result_label[j][i]:
                        beta = -0.25 * c + 0.25
                    else:
                        beta = 0.25 * c - 0.5
                weights[j] = weights[j] * np.exp(-1 * alpha[i] * trans_label[j] * result_label[j][i] * beta)

        # print bata_T
        for i in range(row_T):
            # 跳过训练数据的标签
            tmp = np.sum(np.multiply(alpha, result_label[row_trans + i, :]))

            if tmp > 0:
                predict[i] = 1
            elif tmp < 0:
                predict[i] = -1
            else:
                predict[i] = 0

        self.label_p = predict

    def predict(self):
        self.AUC = roc_auc_score(self.testY, self.label_p)

    def train_classify(self, trans_data, trans_label, test_data, P):
        trans_data[trans_data != trans_data] = 0
        trans_label[trans_label != trans_label] = 0
        test_data[test_data != test_data] = 0
        P[P != P] = 0

        self.m.fit(trans_data, trans_label, sample_weight=P[:, 0])
        return self.m.predict(test_data)

    def calculate_error_rate(self, label_R, label_H, weight):
        total = np.sum(weight)

        # print(weight[:, 0] / total)
        # print(np.abs(label_R - label_H))
        return np.sum(weight[:, 0] / total * np.abs(label_R - label_H) / 2)

# if __name__ == '__main__':
#     flist = []
#     group = sorted(['AEEEM', 'ReLink', 'JURECZKO'])
#     for i in range(len(group)):
#         tmp = []
#         fnameList('../data/' + group[i], tmp)
#         tmp = sorted(tmp)
#         flist.append(tmp)
#
#     tmp = flist[0].copy()
#     target = tmp.pop(0)
#     targetName = target.split('/')[-1].split('.')[0]
#     Xsource, Ysource, Xtarget, Ytarget, loc = MfindCommonMetric(tmp, target, split=True)
#     # RunExperiment(Xsource, Ysource, Xtarget, Ytarget, loc, targetName, 'Bruakfilter', 'Boost', 'adpt')
#
#     DA = Bruakfilter(n_neighbors=5)
#     Xsource, Ysource, Xtarget, Ytarget = DA.run(Xsource, Ysource, Xtarget, Ytarget)
#     m = AdaBoostClassifier(n_estimators=50, learning_rate=1)
#     m.fit(Xsource, Ysource)
#     predict = m.predict(Xtarget)
#     auc = roc_auc_score(Ytarget, predict)
#     print(auc)