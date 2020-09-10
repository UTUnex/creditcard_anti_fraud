## The parameters to be tuned in LGBMClassifier  
1. __num_leaves__: max number of leaves for base learners(decision tree). This is the main parameter to control the __complexity__ of the tree model. Increasing this value may help increase the accuracy of model but also may cause overfitting. It should be less than 2^(max_depth).  
2. __max_depth__: max tree depth for base learner(decision tree).   
3. __n_estimators__: number of boosted trees to fit. Should be set to a large value and use the early_stopping_rounds to stop the training when it's nor learning anything useful.  
4. __min_child_weight__: minimum sum of instance weight(heissan) needed in a child(leaf). Used to deal with overfitting.  
5. __subsample__: subsample ratio of the training instance (this will randomly select part of data without resampling). Used to deal with overfitting and speed up training.  
6. __subsample_freq__: frequency for bagging, set to 1.  
7. __colsample_bytree__: subsamples ratio of columns when constructing each tree.  
## Important reference  
1. [Understanding LightGBM Parameters (and How to Tune Them)](https://towardsdatascience.com/understanding-lightgbm-parameters-and-how-to-tune-them-6764e20c6e5b)  
2. [hyper parameter optimization - suggested parameter grid](https://github.com/microsoft/LightGBM/issues/695)
