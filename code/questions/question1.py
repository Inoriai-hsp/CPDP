import os
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean, stdev
from math import sqrt

def getData():
    list_path = []
    Cohens = []
    path = '../../results (rawAUC)'
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        list_path.append(file_path)

    for file_path in list_path:
        contents = []
        cohens = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                contents.append(line.split(','))
        contents = np.array(contents).T
        for index in range(1, contents.shape[0], 5):
            s = contents[int(index - index % 5)][0]
            if 'Bruakfilter-NB' in s or 'Peterfilter-RF' in s or 'Peterfilter-NB' in s or 'Peterfilter-KNN' in s or 'DS-RF' in s or 'DS-NB' in s or 'Universal-NB' in s or 'DSBF-RF' in s or 'DSBF-NB' in s or 'DTB-NB' in s or 'DBSCANfilter-RF' in s or 'DBSCANfilter-NB' in s or 'DBSCANfilter-KNN' in s:
                c0 = contents[index][1:]
                c1 = contents[int(index - index % 5)][1:]
                c0 = list(map(float, c0))
                c1 = list(map(float, c1))
                cohen = cohens_d(c0, c1)
                cohens.append(cohen)
        Cohens.append(cohens)
    # print(os.listdir(path))
    # Cohens_best = []
    # for i in range(0, len(Cohens)):
    #     cohens_best = []
    #     for j in range(4, len(Cohens[0]), 5):
    #         cohen_best = max(Cohens[i][j-4:j])
    #         cohens_best.append(cohen_best)
    #     Cohens_best.append(cohens_best)
    # print(Cohens_best)
    return Cohens


def cohens_d(c0, c1):
    n1 = len(c0)
    n2 = len(c1)
    if ((n1 - 1) * stdev(c0) ** 2 + (n2 - 1) * stdev(c1) ** 2) == 0:#两组样本方差都为0，怎么处理？
        return mean(c0) - mean(c1)
    cohens_d = (mean(c0) - mean(c1)) / (sqrt((n1 * stdev(c0) ** 2 + n2 * stdev(c1) ** 2) / (n1 + n2)))
    return cohens_d


def cliffsDelta(lst1, lst2, **dull):

    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474} # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    return d


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two


if __name__ == '__main__':
    # all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]
    data = getData()
    data = np.array(data).T
    #
    # tmpData = []
    # for i in range(0, len(data)):
    #     for j in range(0, len(data[i])):
    #         tmpData.append([j+1, data[i][j]])
    # df = pd.DataFrame(tmpData, columns=['technique', 'effect_size'])
    # sns.violinplot(x=df['technique'], y=df['effect_size'], data=df)
    # plt.show()

    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(120, 50))
    # axes.violinplot(data, showmeans=True, showmedians=True)
    # axes.set_title('violin plot')
    # xlabels = []
    # for i in range(0, len(data)):
    #     xlabels.append('technique' + str(i + 1))
    # # 设置格子
    # axes.yaxis.grid(True)
    # axes.set_xticks([y + 1 for y in range(len(data))])
    # plt.setp(axes, xticks=[y + 1 for y in range(len(data))], xticklabels=xlabels)
    # plt.show()

    data = list(data[0:8])
    data.pop(7)
    data.pop(3)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    xlabels = []
    for i in range(0, len(data)):
        xlabels.append('technique' + str(i + 1))
    xticks = range(0, len(data), 1)
    ax.violinplot(data)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_xlabel("technique")
    ax.set_ylabel("effect size")
    ax.set_title("violin plot")
    plt.show()
