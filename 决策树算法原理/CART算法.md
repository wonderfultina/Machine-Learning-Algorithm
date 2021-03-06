1. CART分类树算法的最优特征选择方法

在ID3算法中我们使用了信息增益来选择特征，信息增益大的优先选择。在C4.5算法中，采用了信息增益比来选择特征，以减少信息增益容易选择特征值多的特征的问题。但是无论是ID3还是C4.5,都是基于信息论的熵模型的，这里面会涉及大量的对数运算。能不能简化模型同时也不至于完全丢失熵模型的优点呢？
CART分类树算法使用基尼系数来代替信息增益比，基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好。这和信息增益(比)是相反的。
假设有K个类，样本点属于第k类的概率为pk，则概率分布的基尼指数定义为：

![1](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/1.png)

如果是二类分类问题，计算就更加简单了，如果属于第一个样本输出的概率是p，则基尼系数的表达式为：

Gini(p)=2p(1−p)

对于个给定的样本D,假设有K个类别, 第k个类别的数量为Ck,则样本D的基尼系数表达式为：

![2](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/2.png)

特别的，对于样本D,如果根据特征A的某个值a,把D分成D1和D2两部分，则在特征A的条件下，D的基尼系数表达式为：

![3](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/3.png)

对于二类分类，基尼系数和熵之半的曲线如下：

![4](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/4.png)

从上图可以看出，基尼系数和熵之半的曲线非常接近，仅仅在45度角附近误差稍大。因此，基尼系数可以做为熵模型的一个近似替代。
而CART分类树算法就是使用的基尼系数来选择决策树的特征。同时，为了进一步简化，CART分类树算法每次仅仅对某个特征的值进行二分，而不是多分，这样CART分类树算法建立起来的是二叉树，而不是多叉树。这样一可以进一步简化基尼系数的计算，二可以建立一个更加优雅的二叉树模型。


2. CART分类树算法对于连续特征和离散特征处理的改进

对于CART分类树连续值的处理问题，是将连续的特征离散化。在选择划分点时的度量方式，CART分类树使用的是基尼系数。
具体的思路如下，比如m个样本的连续特征A有m个，从小到大排列为a1,a2,...,am,则CART算法取相邻两样本值的平均数，一共取得m-1个划分点，
其中第i个划分点Ti表示为：Ti=(ai+ai+1)/2。对于这m-1个点，分别计算以该点作为二元分类点时的基尼系数。
选择基尼系数最小的点作为该连续特征的二元离散分类点。比如取到的基尼系数最小的点为at,则小于at的值为类别1，大于at的值为类别2，这样我们就做到了连续特征的离散化。
要注意的是，与ID3或者C4.5处理离散属性不同的是，如果当前节点为连续属性，则该属性后面还可以参与子节点的产生选择过程。

对于CART分类树离散值的处理问题，采用的思路是不停的二分离散特征。

回忆下ID3或者C4.5，如果某个特征A被选取建立决策树节点，如果它有A1,A2,A3三种类别，我们会在决策树上一下建立一个三叉的节点。这样导致决策树是多叉树。但是CART分类树使用的方法不同，他采用的是不停的二分，还是这个例子，CART分类树会考虑把A分成{A1}和{A2,A3}, {A2}和{A1,A3}, {A3}和{A1,A2}三种情况，找到基尼系数最小的组合，比如{A2}和{A1,A3},然后建立二叉树节点，一个节点是A2对应的样本，另一个节点是{A1,A3}对应的节点。
同时，由于这次没有把特征A的取值完全分开，后面我们还有机会在子节点继续选择到特征A来划分A1和A3。这和ID3或者C4.5不同，在ID3或者C4.5的一棵子树中，离散特征只会参与一次节点的建立。



3. CART分类树建立算法的具体流程

算法输入是训练集D，基尼系数的阈值，样本个数阈值。
输出是决策树T。
我们的算法从根节点开始，用训练集递归的建立CART树。
(1) 对于当前节点的数据集为D，如果样本个数小于阈值或者没有特征，则返回决策子树，当前节点停止递归。
(2) 计算样本集D的基尼系数，如果基尼系数小于阈值，则返回决策树子树，当前节点停止递归。
(3) 计算当前节点现有的各个特征的各个特征值对数据集D的基尼系数，对于离散值和连续值的处理方法和基尼系数的计算见第二节。缺失值的处理方法和上篇的C4.5算法里描述的相同。
(4) 在计算出来的各个特征的各个特征值对数据集D的基尼系数中，选择基尼系数最小的特征A和对应的特征值a。根据这个最优特征和最优特征值，把数据集划分成两部分D1和D2，同时建立当前节点的左右节点，做节点的数据集D为D1，右节点的数据集D为D2.
(5) 对左右的子节点递归的调用1-4步，生成决策树。

4. CART回归树建立算法
CART回归树和CART分类树的建立算法大部分是类似的，所以这里我们只讨论CART回归树和CART分类树的建立算法不同的地方。
首先，我们要明白，什么是回归树，什么是分类树。两者的区别在于样本输出，如果样本输出是离散值，那么这是一颗分类树。如果果样本输出是连续值，那么那么这是一颗回归树。
除了概念的不同，CART回归树和CART分类树的建立和预测的区别主要有下面两点：
(1)连续值的处理方法不同
(2)决策树建立后做预测的方式不同。
对于连续值的处理，我们知道CART分类树采用的是用基尼系数的大小来度量特征的各个划分点的优劣情况。这比较适合分类模型，但是对于回归模型，我们使用了常见的和方差的度量方式，CART回归树的度量目标是，对于任意划分特征A，对应的任意划分点s两边划分成的数据集D1和D2，求出使D1和D2各自集合的均方差最小，同时D1和D2的均方差之和最小所对应的特征和特征值划分点。表达式为：


 
对于决策树建立后做预测的方式，上面讲到了CART分类树采用叶子节点里概率最大的类别作为当前节点的预测类别。而回归树输出不是类别，它采用的是用最终叶子的均值或者中位数来预测输出结果。
除了上面提到了以外，CART回归树和CART分类树的建立算法和预测没有什么区别。
对于生成的决策树做预测的时候，假如测试集里的样本A落到了某个叶子节点，而节点里有多个训练样本。则对于A的类别预测采用的是这个叶子节点里概率最大的类别。

![5](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/5.png)

其中，c1为D1数据集的样本输出均值，c2为D2数据集的样本输出均值。
对于决策树建立后做预测的方式，上面讲到了CART分类树采用叶子节点里概率最大的类别作为当前节点的预测类别。而回归树输出不是类别，它采用的是用最终叶子的均值或者中位数来预测输出结果。

5.CART回归树的生成

最小二叉回归树生成算法
输入：训练数据集D；
输出：回归树f(x)

算法：

(1)在训练数据集所在的输入空间中，递归地将每个区域划分为两个子区域并决定每个子区域上输出值，构建二叉决策树：
选择最优切分变量j与切分点s，求解

![6](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/6.png)

遍历变量j，对固定的切分变量j扫描切分点s，选择使上式最小值的对(j,s)。其中Rm是被划分的输入空间，cm是空间Rm对应的固定输出值。
(2)用选定的对(j,s)划分区域并决定相应的输出值：

![7](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/7.png)

(3)继续对两个子区域调用步骤（1），（2），直至满足停止条件。
将输入空间划分为M个区域R1,R2,…,RM，生成决策树:

![8](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/8.png)

示例
上面的东西有点难以理解，下面举个例子来说明。

训练数据见下表，x的取值范围为区间[0.5,10.5],y的取值范围为区间[5.0,10.0],学习这个回归问题的最小二叉回归树。

![9](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/9.png)


求训练数据的切分点，根据所给数据，考虑如下切分点：
1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5
对各切分点，不难求出相应的R1 , R2 , c1 , c2及

![10](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/10.png)

例如，当s=1.5时，R1={1} , R2={2,3,…,10} , c1=5.56 , c2=7.50 

![11](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/11.png)

现将s及m(s)的计算结果列表如下：

![12](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/12.png)

由上表可知，当x=6.5的时候达到最小值，此时R1={1,2,…,6} , R2=7,8,9,10 , c1=6.24 , c2=8.9 , 所以回归树T1(x)为：

![13](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/13.png)

用f1(x)拟合训练数据的残差见下表，表中r2i=yi−f1(xi),i=1,2,…,10

![14](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/14.png)


用f1(x)拟合训练数据的平方误差：

![15](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/15.png)

第2步求T2(x).方法与求T1(x)一样，只是拟合的数据是上表的残差，可以得到

![16](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/16.png)

用f2(x)拟合训练数据的平方误差是：

![17](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/17.png)

继续求得

![18](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/18.png)

用f6(x)拟合训练数据的平方损失误差是

![19](https://github.com/wonderfultina/Machine-Learning-Algorithm/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/images/19.png)

假设此时已经满足误差要求，那么f(x)=f6(x)即为所求的回归树。


