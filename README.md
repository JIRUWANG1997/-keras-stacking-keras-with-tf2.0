# stacking-keras-with-tf2.0
本文完全翻译自博客https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
##做以下修改：
- 代码兼容tensorflow2.0
- 修复一些bug

## 代码思想：
- 首先使用Keras生成数据集，这是一个共有1100个sample，3分类的数据集，我们使用1000个做训练集，100个做测试集
- **底层**：先搭建一个模型，进行一次训练，看一下结果。
- **底层**：接着同样的模型，训练5次，得到5个不通过的模型参数，保存起来。
- **meta层**：分为两部分，1、参数准备。2、训练。
- 参数准备：加载5个训练好的模型，feed进test数据，得到5个1000*3的预测结果矩阵
- meta层训练：reshape成1000*15的矩阵，输入进逻辑回归层，进行最后的分类。

## 输出结果如下：
5个单独模型：
> Model Accuracy: 0.814
> Model Accuracy: 0.811
> Model Accuracy: 0.817
> Model Accuracy: 0.811
> Model Accuracy: 0.801
融合后模型
> Stacked Test Accuracy: 0.834
可见准确率明显提升
