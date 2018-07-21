# 卡尔曼滤波

[TOC]

## 卡尔曼滤波基本原理

[卡尔曼滤波原理说明](http://blog.chinaunix.net/uid-26694208-id-3184442.html)
思考一下下面简单的问题，给定一组噪声测量值，$z_1, z_2, z_3 ... z_k$，不连续的未知数，我们假设下面的模型

$$
z_k = x + v_k
$$

并且$v_k$是对应的噪声，大量数据最小二乘拟合

$$
\hat x_k = \frac{1}{k} \sum_{i=1}^{k} = \text {estimate of $x$ after $k$ measurement ts}
$$

## 基本思想

### 参考地址：[卡尔曼滤波](http://blog.csdn.net/heyijia0327/article/details/17487467)
卡尔曼的基本思想是：预测+测量反馈
在学习卡尔曼之前，需要知道一些基本的概念：什么是协方差，什么叫最小均方差估计，什么是多元高斯分布
**均方差**：它是“误差”的平方的期望值（误差就是估计值与真实值的差），多个样本的时候，均方差等于每个样本的误差平方乘以该样本出现概率的和
**方差**：方差是描述随机变量的离散程度，是变量离期望值的距离
**协方差**：两个变量之间的协方差

$$
cov(X,Y) = E( (X-\mu)(Y-v) )
$$

高斯分布

$$
\sum = 
\begin{matrix}
1 & 0 \\
0 & 1 \\
\end{matrix}
\\
\sum = 
\begin{matrix}
1 & 0.5 \\
0.5 & 1 \\
\end{matrix}
\\
\sum = 
\begin{matrix}
1 & 0.8 \\
0.8 & 1 \\
\end{matrix}
$$


### 卡尔曼滤波器是一个optimal recursive data processing algorithm（最优化自回归数据处理算法）

$$
X(k) = A X(k-1) + B U(k) + W(k) \\
Z(k) = H X(k) + V(k)
$$


## 具体的实例

### 匀加速实例
有一个匀加速运动的小车，它受到的合力为ft，由匀加速运动的位移和速度公式，能得到由t-1到t时刻的位移和速度变化公式：

$$
x_{t} = x_{t-1} + ( \dot{x_{t-1}} \times \Delta t) + \frac{ f_{t}(\Delta t)^2 }{2m}
\\
\dot{x_{t}} = \dot{x_{t-1}} + \frac{f_{t} \Delta t}{m}
$$

## 知乎回答

[尽可能简单的解释卡尔曼滤波](https://www.zhihu.com/question/23971601)















