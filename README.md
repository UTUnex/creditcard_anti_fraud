# Project：[Kaggle Credit Card Anti Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
> Description：Apply machine learning and deep learning techniques to credit card fraud detection  
## Dataset  
> Dataset in this project is a typical imbalanced dataset which contains two classes with quite imbalanced distribution
## Possible Schemes  
> 1. Traditional machine learning -- supervised learning  
> 2. ML -- unsupervised  
> 3. Deep learning -- supervised learning  
> 4. DL -- unsupervised  
## Tools  
> 1. ML  
>> I. scikit-learn + imbalanced learn + xgboost + lightgbm
> 1. scikit-learn + imbalanced learn + xgboost + lightgbm + borutapy(for feature selection)  
> 2. pytorch(neural net design
> 2. 使用Pyspark和基于或兼容Pyspark的相关工具来实现
>>> A.Pyspark ML自带的Logistic Regression, LinearSVC, RandomForest, Naive Bayes, GBT等分类器自身带有一个参数weightCOL用于解决非均衡数据的问题;  
>>> B.Pyspark ML缺少与之对应的第三方特征选择工具，相比scikit-learn基于scikit-learn-contrib的强大生态，Pyspark ML的功能有些单薄;  
>>> C.正在寻找能够使scikit-learn并行运算的第三方库.  
## 代码快速浏览  
> [实现代码](https://nbviewer.jupyter.org/github/UTUnex/creditcard_anti_fraud_zh_CN/blob/master/%E6%9C%89%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/with_feature_scaling_with_feature_selection.ipynb)  
## 注意  
> github有时候可能无法正常显示JupyterNotebook文件，此时请在浏览器中打开**https://nbviewer.jupyter.org/** ，然后将github中.ipynb文件的地址复制到搜索框中，然后点'go',进行浏览
