Two layer net

affine_backward

先考虑求db，想象一下这个图就是，对所有隐层结点，后面再接一个add节点，得到最终的结果，因此
$$
\dfrac{\part L}{\part b_0} = \dfrac{\part L}{\part d_0}\cdot \dfrac{\part d_0}{\part b_0} = \text{dout}_0
$$
再考虑如何求dx![IMG_0011(20211105-153333)](https://i.loli.net/2021/11/05/T6vCaIf4j89dsPK.jpg)

假设网络长成这个样子，那么显然对tmp这一层来说，梯度就是out的梯度

现在来考虑如何求$\dfrac{\part L}{\part x}=\dfrac{\part L}{\part out} \cdot \dfrac{\part out}{\part x}$，这个$\dfrac{\part L}{\part out}$是上一层的梯度，是$n \times m$的，我们需要求的是$\dfrac{\part L}{\part x}$，这个东西跟$x$是同shape的，也就是$n\times d$的，那么我们需要乘一个$m \times d$的东西就可以得到了，然后发现，$W$这个矩阵本来就是$m \times d$的就解决了。

当然不能这样来理解这个问题，换一些公式推导来解决他。

先考虑$\dfrac{\part L}{\part out_0}$也就是上图的第一个点，显然由于$out = W_0X$，因此$\dfrac{\part out_0}{\part x}=W_0$，以此类推可以得到$\dfrac{\part out_i}{\part x} = W_i$，而$\dfrac{\part L}{\part x}=\sum_i \dfrac{\part L}{\part out_i}\cdot \dfrac{\part out_i}{\part x}=\sum_i dout_i \cdot W_i$

如果熟悉矩阵就会发现，这就是$dout \times W^T$

最后要求dw，先从维度上看，目标是$d\times m$的，我们已有的是$n \times m$的，那么需要一个$d\times n$的就可以搞到一个$d\times m$的了，发现$x$是满足这个要求的，当然还是需要一些数学推导来说明这个问题。

先考虑单独的$\dfrac{\part out_0}{\part W_{0,0}}=x_0, \dfrac{\part out_1}{\part W_{0,1}}=x_0,\dfrac{\part out_0}{\part W_{1,0}}=x_1$

所以可以发现$\dfrac{\part out_0}{\part W}=x, \dfrac{\part out_1}{\part W}=x$

因此$\dfrac{\part out}{\part W}=\sum_i \dfrac{\part out_i}{\part W}=\sum_i x,\dfrac{\part L}{\part W}=\sum_i x\cdot dout_i=X^T \cdot dout$

