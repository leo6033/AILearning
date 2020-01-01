# 前言

前面的文章中介绍了决策树以及其它一些算法，但是，会发现，有时候使用使用这些算法并不能达到特别好的效果。于是乎就有了`集成学习`（Ensemble Learning），通过构建多个学习器一起结合来完成具体的学习任务。这篇文章将介绍集成学习，以及其中的一种算法 AdaBoost。

# 集成学习

首先先来介绍下什么是集成学习：

+ 构建多个学习器一起结合来完成具体的学习任务，常可获得比单一学习器显著优越的泛化性能，对“弱学习器” 尤为明显（三个臭皮匠，顶个诸葛亮）
+ 也称为`Multi-Classifier System`, `Committee-Based Learning `
+ 学习器可以是同类型的，也可以是不同类型 

这么一看，就感觉集成学习与常说的模型融合很像，甚至可以理解为就是模型融合。

那么，常用的集成学习方法有哪些呢？

1. **Boosting**，将各种弱分类器串联起来的集成学习方式，每一个分类器的训练都依赖于前一个分类器的结果，代表：AdaBoost，Gradient Boosting Machine
2. **Bagging**，Bootstrap Aggregating 的缩写。这种方法采用的是随机有放回的选择训练数据然后构造分类器，最后进行组合，代表：Random Forest
3. **Voting/Averaging**，在不改变模型的情况下，直接对各个不同的模型预测的结果进行投票或者平均
4. **Binning**，最近看到的一种方法，还没细看，参考[论文](<http://cseweb.ucsd.edu/~elkan/254spring01/jdrishrep.pdf>)
5. **Stacking**
6. **Blending**

*后面几种方法这里暂时不做介绍，后面会单独写博客来介绍这些方法*

# AdaBoost

## 算法思想

这里将介绍一个基于 Boosting 方法的一个学习算法 AdaBoost，于1995年由 Freund 和 Schapire 提出。其主要思想为：

1. 先训练出一个基学习器
2. 根据该学习器的表现对训练样本权重进行调整，使得现有基学习器做错的样本在后续学习器的训练中受到更多的关注
3. 基于调整后的权重来训练下一个基学习器
4. 重复 2、3 直至学习器数目达到事先指定的值 T
5. 最终将这 T 个学习器进行加权结合

$$
H(x)=\operatorname{sign}\left(\sum_{t=1}^{T} \alpha_{t} h_{t}(x)\right)
$$

## 具体算法

设训练数据集 

$$
\{x^{(i)}, y^{(i)}\}_{i=1}^{m},x^{(i)} \in \mathbb{R}^n, y \in \{-1, +1\}
$$
初始化训练数据的权值分布
$$
\mathcal{D}_{1}\left(x^{(i)}\right)=\frac{1}{m}
$$
for t in range(T):

​	假设训练得到分类器 $h_t(x)$ ，则可计算 $h_t(x)$ 在当前训练集上的分类误差：
$$
\epsilon_{t}=P_{x \sim \mathcal{D}_{t}}\left[h_{t}(x) \neq y\right]=\sum_{y^{(i)} \neq h_{t}\left(x^{(i)}\right)} \mathcal{D}_{t}\left(x^{(i)}\right)
$$
​	若 $\epsilon_{t} > 0.5$, break; 否则计算**分类器**权重
$$
\alpha_{t}=\frac{1}{2} \log \frac{1-\epsilon_{t}}{\epsilon_{t}}
$$
​	然后更新样本权重
$$
\mathcal{D}_{t+1}\left(x^{(i)}\right)=\frac{1}{Z_{t}} \mathcal{D}_{t}\left(x^{(i)}\right) \exp \left[-\alpha_{t} y^{(i)} h_{t}\left(x^{(i)}\right)\right]
$$
​	其中 $Z_t$ 为归一化因子
$$
Z_{t}=\sum_{i} \mathcal{D}_{t}\left(x^{(i)}\right) \exp \left[-\alpha_{t} y^{(i)} h_{t}\left(x^{(i)}\right)\right]
$$
构建基本分类器的线性组合
$$
f(x)=\sum_{t=1}^{T} \alpha_{t} h_{t}(x)
$$
得到最终分类器
$$
H(x)=\operatorname{sign}\left(\sum_{t=1}^{T} \alpha_{t} h_{t}(x)\right)
$$


这里我们可以看到 $\alpha_t$ 是大于 $\frac{1}{2} $ 的，如果误分类了，那么 $-\alpha_{t} y^{(i)} h_{t}\left(x^{(i)}\right)$ 为大于 0 的数，那么样本的权重就会被放大，反之，则会被缩小。并且， $\epsilon_t$ 越大，$\alpha_t$ 就越小，即在最终构建强分类器的时候，误差率越小的弱分类器预测结果所占比重越高。

## 算法推导

思考两个个问题， $\alpha_t$ 的公式是怎么来的？以及权重更新公式是怎么来的？下面通过公式推导来讲解

假设已经经过 $t-1$ 轮迭代，得到$f_{t-1}(x)$，根据前向分布加法算法
$$
f_t(x) = f_{t-1}(x) + \alpha_{t}h_t(x)
$$
目标是损失函数最小，即
$$
\min{Loss} = \min\sum_{i=1}^{N}exp[-y_i(f_{t-1}(x_i)+\alpha_th_t)]
$$
所以，有
$$
\begin{eqnarray}(\alpha_t,h_t(x)) & = & \arg {\min_{\alpha,h}\sum_{i=1}^{N}exp[-y_i(f_{t-1}(x_i)+\alpha_th_t()x_i)]} \\ & = & \arg {\min_{\alpha,h}\sum_{i=1}^{N}w_{t,i}exp[-y_i(\alpha_th_t(x_i))]} \end{eqnarray}
$$

$$
w_{t,i} = \exp[-y_if_{t-1}(x_i)]
$$

我们先来化简损失函数
$$
\begin{eqnarray}Loss & = &\sum_{y_i=h_t(x_i)}w_{t,i}exp(-\alpha_t)+\sum_{y_i \ne h_t(x_i)}w_{t,i}exp(\alpha_t)
\\ & = & \sum_{i=1}^{N}w_{t,i}(\frac{\sum_{y_i=h_t(x_i)}w_{t,i}}{\sum_{i=1}^{N}w_{t,i}}exp(-\alpha_t)+\frac{\sum_{y_i \ne h_t(x_i)}w_{t,i}}{\sum_{i=1}^{N}w_{t,i}}exp(-\alpha_t))
\end{eqnarray}
$$
仔细以看，后面那项 $\frac{\sum_{y_i \ne h_t(x_i)}w_{t,i}}{\sum_{i=1}^{N}w_{t,i}}$ 就是分类误差率 $\epsilon_{t}$，所以
$$
Loss = \sum_{i=1}^{N}w_{t,i}[(1-\epsilon_t)exp(-\alpha_t)+\epsilon_texp(\alpha_t)]
$$
对 $\alpha_t$ 求偏导
$$
\begin{eqnarray}
\frac{\partial Loss}{\partial \alpha_t} & = & \sum_{i=1}^{N}w_{t,i}[-(1-\epsilon_t)exp(-\alpha_t)+\epsilon_texp(\alpha_t)]
\end{eqnarray}
$$
令 $\frac{\partial Loss}{\partial \alpha_t} = 0$ ，则
$$
-(1-\epsilon_t)exp(-\alpha_t)+\epsilon_texp(\alpha_t) = 0
$$
推得
$$
\alpha_{t}=\frac{1}{2} \log \frac{1-\epsilon_{t}}{\epsilon_{t}}
$$
另，由前向分布加法算法
$$
\begin{eqnarray}
w_{t,i} & = & \exp[-y_if_{t-1}(x_i)] \\
& = & \exp[-y_i(f_{t-2}(x_i)+\alpha_{t-1}h_{t-1}(x_i))] \\
& = & w_{t-1,i}\exp[\alpha_{t-1}h_{t-1}(x_i)]
\end{eqnarray}
$$
再加上规范化因子即为算法中的更新公式。（公式敲的要累死了\~~~）

## 代码实现

这里为了方便起见，我使用了 sklearn 里面的决策树，之前使用的时候一直没发现 sklearn 里的决策树可以带权重训练 orz。。。决策树带权训练的代码我后面再研究研究

```python
from sklearn.tree import DecisionTreeClassifier
def adaboost(X, y, M, max_depth=None):
    """
    adaboost函数，使用Decision Tree作为弱分类器
    参数:
        X: 训练样本
        y: 样本标签, y = {-1, +1}
        M: 使用 M 个弱分类器
        max_depth: 基学习器决策树的最大深度
    返回:
        F: 生成的模型
    """
    num_X, num_feature = X.shape
    
    # 初始化训练数据的权值分布
    D = np.ones(num_X) / num_X
    
    G = []
    alpha = []
    
    for m in range(M):
        # 使用具有权值分布 D 的训练数据集学习，得到基本分类器
        # 使用 DecisionTreeClassifier，设置树深度为 max_depth
        G_m = DecisionTreeClassifier(max_depth=max_depth)
        # 开始训练
        G_m.fit(X, y, D)
        # 计算G_m在训练数据集上的分类误差率
        y_pred = G_m.predict(X)
        e_m = np.sum(D[y != y_pred])
        
        if e_m == 0:
            break
        
        if e_m == 1:
            raise ValueError("e_m = {}".format(e_m))
            
        # 计算 G_m 的系数
        alpha_m = np.log((1 - e_m) / e_m) / 2
#         print(alpha_m)
        # 更新训练数据集的权值分布
        D = D * np.exp(-alpha_m * y * y_pred)
        D = D / np.sum(D)
        # 保存 G_m 和其系数
        G.append(G_m)
        alpha.append(alpha_m)
    
    # 构建基本分类器的线性组合
    def F(X):
        num_G = len(G)
        score = 0
        for i in range(num_G):
            score += alpha[i] * G[i].predict(X)
        return np.sign(score)
        
    return F
```

# 小节

上面介绍了集成学习的一些知识点以及 AdaBoost 的基本原理及实现，下一篇将介绍集成学习中基于 Bagging 的随机森林(Random Forest)。