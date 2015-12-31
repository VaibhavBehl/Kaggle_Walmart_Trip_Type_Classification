(timeline: Fall 2015)

# Kaggle_Walmart_Trip_Type_Classification  
Competition Link: [https://www.kaggle.com/c/walmart-recruiting-trip-type-classification]
Team: Vaibhav Behl(members:1)
**Private Leaderboard Rank: #159/1000**   


In this kaggle competition we had to classify customer trips using a transactional dataset of the items they've purchased at Wallmart. This was a multi-class classification algorithm with highly skewed classes. Some classes were also difficult to differentiate as seen from the confusion matrix. Multinomial log-loss was used as the metric to measure the performance of the model. The top models were able to report a mlogloss of less than 0.5, meaning on an average they were able to report a probability of ~0.61 for the correct class. This is a big improvement as mlogloss metric heavily penalizes low probability values using the log function.  

My method involved lots of feature engineering and ensemble of xgboost models. My top model was an ensemble of two xgboost model, namely: 
- One made through (v5.3) of the feature generation file(see 'code' for more details).  
- Other made through (v5.4) of the feature generation file.  
Both of these were  run for 300 rounds with 50 depth and 0.1 eta using the following parameters: 'max_delta_step':1 'subsample':0.8 'colsample_bytree':0.8, 'min_child_weight':3, 'lambda':8, 'alpha':3, 'gamma':1. These parameters were found thorugh CV 3-folds on train dataset.  

With this ensemble my mlogloss score on test set was: 0.64139, meaning on an average I was assigning a probability of ~0.53 for the correct class.  

I was also observed during the competition that Neural Nets were outperforming other models and the results indicated that most of the top solutions were an ensemble of these two models. Towards the end of the competition I tried running Lasagne+Theano for this dataset but was not successful in running it before the deadline ended. Post competition I switched to Keras(a NN library)+Theano configuration and was succeful in running it within set time. I will be experimenting with this configuration and will be updating the score achieved with NN and its ensemble with xgboost.  

For some insights about the data see - 'data.pdf' or 'TODO_wallmart.txt'(this file is not formatted properly, so will be difficult to read!)  
