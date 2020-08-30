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
> [Code for ml classification](https://github.com/UTUnex/creditcard_anti_fraud/blob/master/ml/supervised/with_feature_scaling_with_feature_selection.ipynb)  
> [Code for dl classification]()  
## Attention  
> Sometimes github cannot display JupyterNotebook. Please use nbviewer.
