import pandas as pd
from math import log

def load(filename):
	'''
		input: 文件
		output: DataFrame数据
	'''
	df = pd.read_csv(filename)
	return df

def calEnt(df):
	'''
		input: 数据集
		output: 熵
		descripion: 计算给定数据集的熵
	'''
	# 返回数据行数
	numEntries = df.shape[0]
	# 字典，用于保存每个标签（label）出现次数
	labelCounts = {}
	cols = df.columns.tolist()
	# 获取标签
	classlabel = df[cols[-1]].tolist()
	# 对每组特征向量进行统计
	for currentlabel in classlabel:
		if currentlabel not in labelCounts.keys():
			labelCounts[currentlabel] = 1
		else:
			labelCounts[currentlabel] += 1

	Ent = 0.0
	# 计算熵
	for key in labelCounts:
		prob = labelCounts[key]/numEntries
		Ent -= prob*log(prob, 2)

	return Ent

def splitDatsSet(df, axis, value):
	'''
		input: 数据集，所占列，选择值
		output: 划分数据集
		description: 按照给定特征划分数据集；选择所占列中等于选择值的项
	'''
	cols = df.columns.tolist()
	axisFeat = df[axis].tolist()
	# L = []
	# L.append(axis)

	# retDataset = df[list(set(cols)-set(L))]
	retDataset = pd.concat([df[feaVec] for feaVec in cols if feaVec != axis], axis=1)

	i = 0
	# 需要丢弃的行
	dropIndex = []
	for feaVec in axisFeat:
		if feaVec != value:
			dropIndex.append(i)
			i += 1
		else:
			i += 1
	# 划分数据集
	newDataset = retDataset.drop(dropIndex)
	return newDataset.reset_index(drop=True)


def chooseBestFeatureToSplit(df):
	'''
		input: 数据集
		output: 最优特征和信息增益
	'''
	# 获取特征数量
	numFeatures = len(df.columns) - 1
	# 计算熵
	Ent = calEnt(df)
	# 信息增益
	bestInfoGain = 0.0
	splitInfo = 0.0
	# 最优特征索引值
	bestFeature = -1
	cols = df.columns.tolist()
	# 遍历所有特征
	for i in range(numFeatures):
		# 获取第i个特征的所有不同的取值
		equalVals = set(df[cols[i]].tolist())
		# 经验条件熵
		newEnt = 0.0
		# 计算信息增益
		for value in equalVals:
			# 获取划分后的子集
			subDataset = splitDatsSet(df, cols[i], value)
			# 计算子集概率
			prob = subDataset.shape[0] / df.shape[0]
			# 根据公式计算条件熵
			newEnt += prob * calEnt(subDataset)
			# 计算特征熵
			splitInfo += -prob * log(prob, 2)
		infoGain = Ent - newEnt
		print(cols[i], infoGain)
		# 计算信息增益比
		infoGainRatio = infoGain / splitInfo
		# 更新信息增益，找到最大的信息增益
		if infoGainRatio > bestInfoGain:
			bestInfoGain = infoGainRatio
			bestFeature = cols[i]
	return bestFeature, infoGainRatio



def majorityCnt(classList):
	'''
		input: 类别列表
		output: 子节点的分类
		description: 数据集已经处理了所有属性，但是类标签依然不是唯一的，
		  采用多数判决的方法决定该子节点的分类
	'''
	classCount = {}
	# 统计classList中每个元素出现的次数
	for clas in classList:
		if clas not in classCount.keys():
			classCount[clas] = 0
		classCount[clas] += 1
	# 根据字典值降序排列
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)
	return sortedClassCount[0][0]

def createTree(df, dropcol):
	'''
		input: 数据集和需删除特征
		output: 决策树
		description: 递归实现决策树构建
	'''
	# cols = df.columns.tolist()[:-1]
	# 取分类标签
	classList = df[df.columns.tolist()[-1]].tolist()

	# 若数据集中所有实例属于同一类，将Ck作为该节点的类标记
	if classList.count(classList[0]) == len(classList):
		return classList[0]

	# 若特征集为空集，将数据集中实例数最大的类Ck作为该节点的类标记
	if len(df[0:1]) == 0:
		return majorityCnt(classList)

	# 获取最优特征与信息增益
	bestFeature, bestInfoGain = chooseBestFeatureToSplit(df)

	print('特征集和类别:',df.columns.tolist())
	print('bestFeature:',bestFeature)
	# 根据最优特征的标签生成树
	myTree = {bestFeature:{}}
	# 得到最优特征的属性值
	featValues = df[bestFeature]
	# 去掉重复属性值
	uniqueVals = set(featValues)
	# 遍历创建决策树
	for value in uniqueVals:
		myTree[bestFeature][value] = createTree(splitDatsSet(df, bestFeature, value), bestFeature)
	return myTree

def main():
	filename = "data.csv"
	dataset = load(filename)
	dropCol = []
	myTree = createTree(dataset, dropCol)
	print(myTree)

if __name__ == '__main__':
	main()