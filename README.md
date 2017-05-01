模式分类作业
==========

### Linear SVM 

在linear SVM中，不同的参数C对算法性能有不同的影响。我将C的值设置在 10^-5 .... 10^4，并通过不同性能量度对C进行筛选，
分别有(accuracy, precision, recall, f1-score)


> 不同的参数C的交叉验证准确率

![不同的参数C的交叉验证准确率](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-1.png)

> 在测试集合当中不同参数C的准确率

![在测试集合当中不同参数C的准确率](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-2.png)

> 各个参数C对应的Precision-Recall曲线

![各个参数C对应的Precision-Recall曲线](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-3.png)

> 各个参数C对应的F1-score值

| Value of C | F1-score |
| -----------|:--------:|
|   0.00001  |**0.7728**|
|   0.00010  | 0.7682   |
|   0.00100  | 0.7647   |
|   0.01000  | 0.7581   |
|   0.10000  | 0.6393   |
|   1.00000  | 0.4518   |
|   10.0000  | 0.6921   |
|   100.000  | 0.6651   |
|   1000.00  | 0.6401   |

最后发现无论是交叉验证、准确率还是F1-score的值，当c=10^-5都是最好。一开始我尝试的最小C值为10^-4，但C的值变小，lsvm的性能貌似越来越好。
但是按照理论:
- C值过小时，分类器就会过于“不在乎”分类错误，于是分类性能就会较差。（欠拟合）
- C值过大时，分类器就会竭尽全力地在训练数据上少犯错误。（过拟合）

但在这里未能体现出来，可能是因为C的值还不足够小。