import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
"""
Author:
        ITryagain
Modify:
        2019-04-09
"""

def loadDataSet(fileName):
    """
    读取数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
        labelMat - 数据标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    return np.array(dataMat), np.array(labelMat)


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """
        :param dataMatIn: 输入数据 [X1, X2, ... , XN]
        :param classLabels: 分类标签 [y]
        :param C: 松弛变量
        :param toler: 容错率
        """
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))   # 误差缓存
        self.k = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    """
    :param oS:
    :param k: 第k行
    :return: 预测值与实际值的差
    """
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.k[:, k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def clipAlpha(aj, H, L):
    """
    调整aj的值，使aj处于 L<=aj<=H
    :param aj: 目标值
    :param H: 最大值
    :param L: 最小值
    :return:
        aj 目标值
    """
    if aj > H:
        aj = H
    elif L > aj:
        aj = L
    return aj

def selectJrand(i, m):
    """
      随机选择一个整数
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数目
    :return:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):
    """
    内循环的启发式方法，选择第二个alpha的值
    :param i: 第一个alpha的下标
    :param oS:
    :param Ei: 预测结果与真实结果比对，计算误差Ei
    :return:
        j  随机选出的第j一行
        Ej 预测结果与真实结果比对，计算误差Ej
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0] # 非零E值的行的list列表，所对应的alpha值
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:   # 在所有的值上进行循环，并选择其中使得改变最大的那个值
            if k ==1:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxk = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:   # 如果是第一次循环，则随机选择一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """
    计算误差值并存入缓存中
    :param oS:
    :param k: 某一列的行号
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def kernelTrans(X, A, kTup):
    """
    核转换函数
    :param X: dataMatIn数据集
    :param A: dataMatIn数据集的第i行的数据
    :param kTup:核函数的信息
    :return:
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        # linear kernel:   m*n * n*1 = m*1
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        # print(len(K))
        # K = math.exp(K / (-1 * kTup[1] ** 2))
        for i in range(m):
            K[i] = math.exp(K[i] / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def innerL(i, oS):
    """
    内循环代码
    :param i:具体的某一行
    :param oS:
    :returns:
         0   找不到最优的值
        1   找到了最优的值，并且oS.Cache到缓存中
    """
    Ei = calcEk(oS, i)

    # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
    # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
    # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
    '''
    # 检验训练样本(xi, yi)是否满足KKT条件
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化。效果更明显
        j ,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L ==H:
            print("L==H")
            return 0
        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = 2.0 * oS.k[i,j] - oS.k[i, i] - oS.k[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算出一个新的alphas[j]值
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # w= Σ[1~n] ai*yi*xi => b = yj Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.k[i, i] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.k[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.k[i, j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.k[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    """
    SMO算法外循环
    :param dataMatIn: 输入数据集
    :param classLabels: 标签
    :param C: 松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :param kTup: 核函数
    :return:
    """
    # 创建一个 optStruct 对象
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    # 循环迭代结束 或者 循环遍历所有alpha后，alphaPairs还是没变化
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non_bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS


def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    :param alphas: 拉格朗日乘子
    :param dataArr: 数据集
    :param classLabels:
    :return:
        wc 回归系数
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def plotfig_SVM(xArr, yArr, ws, b, alphas):
    """
    参考地址：
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = np.array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = np.arange(-1.0, 10.0, 0.1)

    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b-ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)

    for i in range(np.shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(70):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()


def predict(data, oS):
    r = oS.b
    for i in range(oS.m):
        r += oS.alphas[i] * oS.labelMat[i] * data * oS.X[i, :].T
    return 1 if r > 0 else -1


def score(X_test, y_test, oS):
    right_count = 0
    for i in range(len(X_test)):
        result = predict(X_test[i], oS)
        if result == y_test[i]:
            right_count += 1
    return right_count / len(X_test)


if __name__ == "__main__":
    # 获取特征和目标变量
    dataArr, labelArr = loadDataSet('testSet.txt')
    X_train, X_test, y_train, y_test = train_test_split(dataArr, labelArr, test_size=0.3, random_state=4)
    # print labelArr

    # b是常量值， alphas是拉格朗日乘子
    # 0.6 0.001 40 0.5666666666666667
    oS = smoP(X_train, y_train, 0.6, 0.0001, 200, kTup=('rbf', 10))
    b = oS.b
    alphas = oS.alphas
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = np.mat(X_train)[svInd]
    labelSV = np.mat(y_train).transpose()[svInd]
    print('/n/n/n')
    print('b=', b)
    print('alphas[alphas>0]=', alphas[alphas > 0])
    print('shape(alphas[alphas > 0])=', np.shape(alphas[alphas > 0]))
    for i in range(70):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])
    # 画图
    ws = calcWs(alphas, X_train, y_train)
    plotfig_SVM(X_train, y_train, ws, b, alphas)
    # print(score(X_test, y_test, oS))
    datMat = np.mat(X_test)
    lableMat = np.mat(y_test).transpose()
    m, n = np.shape(datMat)
    right_count = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', 10))
        pre = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        sign = 1 if pre > 0 else -1
        if sign == labelArr[i]:
            right_count += 1
    print(right_count / len(X_test))
