模式分类作业
==========

根据给定的数据集合，我挑选出了几个分类器模型进行评估，并根据不同的度量属性判断选取哪个模型以及其参数。在这里我挑选了Linear SVM, Guassian 
bayes, adaBoost, decision tree 模型作为候选模型。在评估当中，我使用主要使用10-folds cross-validation作为主要的评估方法。另外我使用 
average accuracy, precision, recall, F1-Score作为主要的评估属性。

### 对于Linear SVM 

在linear SVM中，不同的参数C对算法性能有不同的影响。我将C的值设置在 10^-5 .... 10^4，并通过不同性能量度对C进行筛选，
分别有(accuracy, precision, recall, f1-score)


> 不同的参数C的交叉验证准确率(均值)

![不同的参数C的交叉验证准确率](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-1.png)

> 各个参数C对应的Precision-Recall曲线(10-fold cross-validation的均值)

![各个参数C对应的Precision-Recall曲线](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-3.png)

> 各个参数C对应的F1-score值（10-fold cross-validation）

| Value of C | F1-score  |
| -----------|:---------:|
|   0.00001  |**0.72376**|
|   0.00010  | 0.71877   |
|   0.00100  | 0.70159   |
|   0.01000  | 0.68769   |
|   0.10000  | 0.67633   |
|   1.00000  | 0.66822   |
|   10.0000  | 0.69414   |
|   100.000  | 0.67586   |
|   1000.00  | 0.69348   |

> 在测试集合当中不同参数C的准确率

![在测试集合当中不同参数C的准确率](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-2.png)

最后发现无论是交叉验证、准确率还是F1-score的值，当c=10^-5都是最好。一开始我尝试的最小C值为10^-4，但C的值变小，lsvm的性能貌似越来越好。
但是按照理论:
- C值过小时，分类器就会过于“不在乎”分类错误，于是分类性能就会较差。（欠拟合）
- C值过大时，分类器就会竭尽全力地在训练数据上少犯错误。（过拟合）

但在这里未能体现出来，可能是因为C的值还不足够小。

### 对于RandomForest

##### 森林中树的个数：20

| max features | spend time (s) | f1      | average accuracy |
|:------------:|:--------------:|:-------:|:----------------:|
| 10           |    2.1         | 0.81852 |     0.9165       |
| log2(n)      |    3.0         | 0.81845 |     0.9168       |
| sqrt(n)      |    14          | 0.86445 |     0.9349       |
| 100          |    32.21       | 0.87659 |     0.9400       |
| 200          |    72.2        | 0.87623 |     0.9398       |

> 不同属性数量的Precision-Recall曲线(10-fold cross-validation的均值)

![不同属性数量的Precision-Recall曲线](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-5.png)


##### 森林中树的个数：50

| max features | spend time (s) | f1      | average accuracy |
|:------------:|:--------------:|:-------:|:----------------:|
| 10           |    7.0         | 0.85661 |     0.9327       |
| log2(n)      |    7.5         | 0.86382 |     0.9356       |
| sqrt(n)      |    36.7        | 0.89114 |     0.9470       |
| 100          |    82.21       | 0.89546 |     0.9485       |
| 200          |    176.1       | 0.89676 |     0.9490       |

> 不同属性数量的Precision-Recall曲线(10-fold cross-validation的均值)

![不同属性数量的Precision-Recall曲线](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-6.png)

结论：从平均训练花费时间，f1和平均准确率来看，树的数量：50、最大属性个数：log2(dim_n)的效果比较理想。我们将它作为随机森林的代表加入到
不同训练模型当中进行对比。

### 不同的模型

对于之前Linear SVM不同的参数C的评估，我们选择最优的参数C（10^-5）作为Linear SVM的参数加入到模型集合中一起进行评估。

> 各个模型对应的Precision-Recall曲线(10-fold cross-validation的均值)

![各个模型对应的Precision-Recall曲线](https://github.com/Salon-sai/Pattern-Recognition-hw/blob/master/figure_1-4.png)


|  Model         | F1-score  | average accuracy | accuracy (Test Set) |
| -----------    |:---------:|:----------------:|:-------------------:|
|bayes/GaussianNB|  0.60013  |     0.7402       |        0.72844      |
|   decisionTree |  0.57864  |   0.7961         |        0.78080      |
|   adaBoost     |**0.73040**|   **0.8690**     |      **0.88436**    |
|   lsvm_1e-05   |  0.72376  |     0.8629       |        0.87055      |