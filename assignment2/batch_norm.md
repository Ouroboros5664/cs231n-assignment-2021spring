在写batch normalization的back propagation的时候发现，自己对batch的反向传播理解有问题，所以这篇先写一些和传统网络的backprop相关的知识点，加固一下自己的认识

比如现在得到了上一层的梯度dout，网络结构是batch_size=2，这一层的输入为3个feature，输出到2个隐层的节点

那么首先我们知道这一层input的size为$2\times 3$的，这一层的$W$的size为$3\times 2$的，$bais$的size为2，全部都数学化表示后就是
$$
\text{input} = 
\begin{bmatrix}
i_{11} & i_{12} & i_{13} \\
i_{21} & i_{22} & i_{23} \\
\end{bmatrix}

\quad
W=
\begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22} \\
w_{31} & w_{32} \\
\end{bmatrix}
\quad

\text{out}=
\begin{bmatrix}
o_{11} & o_{12} \\
o_{21} & o_{22} \\
\end{bmatrix}
$$
其中$\text{out}=\text{input}\times W$，$o_{ij}$为第i个样本在该隐层的第$j$个值

那么拆解一下这个也就是
$$
o_{11}=i_{11}w_{11}+i_{12}w_{21}+i_{13}w_{31}+b_1 \\
o_{12}=i_{11}w_{12}+i_{12}w_{22}+i_{13}w_{32}+b_2 \\
o_{21}=i_{21}w_{11}+i_{22}w_{21}+i_{23}w_{31}+b_1 \\
o_{22}=i_{21}w_{12}+i_{22}w_{22}+i_{23}w_{32}+b_2 \\
$$
得到的dout也就是这几个值的梯度，那么这个时候来考虑求$db$, $dw$, $dx$

假设这个out已经是最后的结果了，那么loss显然应该是$\sum$这些值得到的

我们现在要求$\dfrac{dl}{db}$也就是说$\dfrac{dl}{db_1}=\dfrac{dl}{do_{11}}+\dfrac{dl}{do_{21}}, \dfrac{dl}{db_2}=\dfrac{dl}{do_{12}}+\dfrac{dl}{do_{22}}$

所以$\dfrac{dl}{db}=\sum\dfrac{dl}{do_{i}}$

回到batch normalization

已知的是$x=\gamma x + b$

让我们直接把所有知道的式子都写下来，更清楚一些
$$
\mu = \dfrac{1}{m}\sum_i^m x_i \\
\sigma^2 = \dfrac{1}{m} \sum_i^m (x_i - \mu)^2 \\
\hat{x}_i = \dfrac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
y_i = \gamma \hat{x_i} + \beta
$$


那么$\dfrac{dl}{db}=\dfrac{dl}{dout} \cdot \dfrac{dout}{db}=\sum\dfrac{dl}{dout_i} \cdot \dfrac{dout_i}{db}=\sum\dfrac{dl}{dout_i}$，所以结果也就是把所有样本的dout相加

同样地可以求$\dfrac{dl}{d\gamma}=\dfrac{dl}{dout} \cdot \dfrac{dout}{d\gamma}=\sum\dfrac{dl}{dout_i} \cdot \dfrac{dout_i}{d\gamma}=\sum\dfrac{dl}{dout_i}\cdot \hat{x_i}$

![img](https://pic4.zhimg.com/80/v2-b8687c2cf7323223dff2eeff377bce8f_720w.jpg)

最后要求$\dfrac{dl}{dx}$，先扒一张计算图下来，这个部分有点复杂，先来找找哪些是和$x$有关的，有$\mu, \sigma, \hat{x}$，其中最变态的是$\hat{x}$和三者都有关，$\sigma$和两者都有关，所以这两个东西在求导的时候是个加法，需要把各个部分都加起来，但从计算图来分析的话就完全没有这么复杂了，按链式法则直接求下来就好了，记住已知的是$\dfrac{\part l}{\part y}$就好
$$
\dfrac{\part y_i}{\part \hat{x_i}} =& \gamma \\
\dfrac{\part \hat{x_i}}{\part x_i} =& \dfrac{1}{\sqrt{\sigma^2 + \epsilon}} \\
\dfrac{\part \hat{x_i}}{\part \mu} =& -\dfrac{1}{\sqrt{\sigma^2 + \epsilon}} \\
\dfrac{\part \hat{x_i}}{\part \sigma^2} =& (x_i-\mu)\cdot \dfrac{-1}{2} \cdot (\sigma^2 + \epsilon)^{-\frac{3}{2}} \\
\dfrac{\part \sigma^2}{\part x_i} =& \dfrac{2(x_i-\mu)}{m} \\
\dfrac{\part \sigma^2}{\part \mu} =& -\dfrac{2\sum (x_i-\mu)}{m} \\
\dfrac{\part \mu}{\part x_i} =& \dfrac{1}{m}
$$
这些就是计算图上节点各自的偏导了，那么我们要求$\dfrac{\part l}{\part x}$就把这些加起来就可以了，需要注意的是，有些偏导是需要对下标$\sum$的

**特别注意**，$\dfrac{\part l}{\part \sigma^2} = \sum \dfrac{\part l}{\part \hat{x_i}}\cdot\dfrac{\part \hat{x_i}}{\part \sigma^2}$，这里要记得求和。**时刻记得**导数的size要匹配。

因为$\dfrac{\part l}{\part \mu}$，显然每一个$\hat{x}_i$都与$\mu$有关

当然完全按照计算图来做的话会比较麻烦，因为求dx的过程太长了，可以利用向量运算简化掉这个过程 

![image-20211115213727569](https://i.loli.net/2021/11/15/mMwv42jrI1GLxg3.png)

当然也可以按照这张图推导，相关式子如下（大概这就是为什么我找到的资料都是两项加起来的了？）
$$
\begin{align}
& \mu=\frac{1}{N}\sum_{k=1}^N x_k  &  v=\frac{1}{N}\sum_{k=1}^N (x_k-\mu)^2 \\
& \sigma=\sqrt{v+\epsilon}         &  y_i=\frac{x_i-\mu}{\sigma}
\end{align}
$$
