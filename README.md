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
>>> 1). Main: scikit-learn + imbalanced learn + xgboost + lightgbm  
>>> 2). Others: category_encoders(for categorical feature encoding) + borutapy(feature selection) + hyperopt(hyperparameters tuning)
> 2. DL  
>>> 1). Main: pytorch(neural net design) + skorch(trainining)  
>>> 2). Others: category_encoders(for categorical feature encoding) + daskML(tuning) / microsoft nni(tuning + visualization & monitoring)  
>>> 3). Potential: optuna(tuning), wandb(tuning + visualization & monitoring),ray tune-sklean(tuning)
> 3. Pyspark (future)  
## Code fast overview  
> [Code](https://nbviewer.jupyter.org/github/UTUnex/creditcard_anti_fraud_zh_CN/blob/master/%E6%9C%89%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/with_feature_scaling_with_feature_selection.ipynb)  
## Attention  
> Sometimes github cannot display JupyterNotebook. Please use nbviewer.
