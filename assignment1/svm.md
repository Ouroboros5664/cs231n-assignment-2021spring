硬间隔的SVM的目标为
$$
min \dfrac{1}{2}||w||^2 \\
s.t. \quad y\cdot(w^Tx+b)\ge 1
$$
软间隔SVM为
$$
min \dfrac{1}{2}||w||^2+C\cdot\sum_i \xi_i \\
s.t. \quad y\cdot(w^Tx+b) \ge 1 - \xi_i \\
\xi_i \ge 0
$$
即到超平面的function distance可以$\le 1$，转换一下式子就是
$$
min( \dfrac{1}{2}||w||^2 + C\cdot \sum_i max\{0,1-y\cdot(w^Tx+b)\})
$$
通过改变C这个式子可以变成
$$
min( \lambda||w||^2 +\sum_i max\{0,1-y\cdot(w^Tx+b)\})
$$
于是这个就变成损失函数了

多分类的SVM与这个式子类似，记
$$
L_i=\sum_{j\ne y_i} max(0,f(x_i,W)_j-f(x_i,W)y_i+\Delta)
$$
$f(x_i,W)_j$为第$j$类的得分，$\Delta$通常取1，损失即考虑正确的分数与其他类的分数差要大于$\Delta$

则总体的Loss为
$$
L=\dfrac{1}{N}\sum_iL_i+\lambda ||W||^2
$$
我们需要求$\dfrac{\partial L}{\partial W_j}$，那么先考虑$\dfrac{\partial L_i}{\partial W_j}$即可，因为$\dfrac{\partial L_i}{\partial W_j}=\dfrac{1}{N}\sum L_i+2\lambda W_j$，考虑分类讨论
$$
\text{若 }j\ne y_i \quad  
\dfrac{\partial L_i}{\partial W_j}= \dfrac{\partial \sum_{j\ne y_i W_jX_i-W_{y_i}X+\Delta}}{\partial W_j}=X_i \\
\text{若 }j= y_i \quad  \dfrac{\partial L_i}{\partial W_j}=\sum-X_i
$$
于是两个导数就求完了

考虑如何用矩阵直接求到$dW$，考虑每个$X_i$的贡献，当$W_jX_i-W_{y_i}X_i+\Delta\ge0$时，$X_i$就对$dW_j$这一列有贡献，对$dW_{y_i}$这一列也有贡献分别为$1,-1$，于是可以统计这个贡献

$X\cdot W$可以直接得到score矩阵，维度为$n\cdot c$，对每一行可以统计有贡献的数量。

那么$X^T\cdot M$就直接得到$dW$了

