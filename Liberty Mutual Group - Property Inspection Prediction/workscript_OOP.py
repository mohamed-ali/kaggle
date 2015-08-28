#workscript OOP

import pandas as pd
import numpy as np 
import argparse 
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt  
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from time import time 
from operator import itemgetter
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

#a proposition to the Kaggle competition: 
# Liberty Mutual Group: Property Inspection Prediction

class KagglePropertyInspectionPrediction():
	def __init__(self):
		pass

	def loadData(self,data_file):
		return pd.read_csv(data_file, header=0)

	def filterOutNonImportantFeatures(self,data,non_important_feature_list):
		#non_important_feature_list must be a list of valid column names 
		return data[data.columns.difference(non_important_feature_list)]

	def DataPreparation(self,data):
		data[["T1_V4","T1_V5","T1_V6","T1_V7","T1_V8","T1_V9","T1_V11","T1_V12","T1_V15","T1_V16","T1_V17","T2_V3","T2_V5","T2_V11","T2_V12","T2_V13"]] = data[["T1_V4","T1_V5","T1_V6","T1_V7","T1_V8","T1_V9","T1_V11","T1_V12","T1_V15","T1_V16","T1_V17","T2_V3","T2_V5","T2_V11","T2_V12","T2_V13"]].apply(LabelEncoder().fit_transform)
		return data		

	def FeaturePermutation(self,data):
		#do the experiment of permutating columns and evaluating the relative importance. 
		# use this as a reference 
		# http://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
		pass 

	def ApplySVM_SVC(self,train,test, cross_validation, full_train):
		svm = SVC(cache_size=700,verbose=True)
		target_train = train[['Hazard']]
		cross_validation_test = cross_validation[['Hazard']]
		prepared_train = train[train.columns.difference(['Id','Hazard'])]
		svm.fit(prepared_train,target_train)
		dt = svm.predict(test[test.columns.difference(['Id'])])
		print "prediction score on cross validation"
		print svm.score(cross_validation[cross_validation.columns.difference(['Id','Hazard'])],cross_validation_test)
		dt_cv = svm.predict(cross_validation[cross_validation.columns.difference(['Id','Hazard'])])
		
		test['Hazard'] = dt
		cross_validation['predicted_Hazard'] = dt_cv
		return test,cross_validation

	def ApplySVM_LinearSVC(self,train,test, cross_validation, full_train):
		svm = LinearSVC(verbose=1)
		target_train = train[['Hazard']]
		cross_validation_test = cross_validation[['Hazard']]
	
		svm.fit(train[train.columns.difference(['Id','Hazard'])],target_train)
		dt = svm.predict(test[test.columns.difference(['Id'])])
		#test = test[test.columns.difference(['Id'])]

		print "prediction score on cross validation"
		print svm.score(cross_validation[cross_validation.columns.difference(['Id','Hazard'])],cross_validation_test)
		dt_cv = svm.predict(cross_validation[cross_validation.columns.difference(['Id','Hazard'])])
		
		test['Hazard'] = dt
		cross_validation['predicted_Hazard'] = dt_cv
		return test,cross_validation

	def EnsembleRandomForestRegressor(self,train,test, cross_validation, full_train,config):
		ERF = RandomForestRegressor(n_estimators=config['n_estimators'],criterion = config['criterion'], max_depth=None,verbose=True, n_jobs=3,max_features=None,min_samples_split=10,min_samples_leaf=10,bootstrap=False)
		target_train = train[['Hazard']]
		cross_validation_test = cross_validation[['Hazard']]
		prepared_train = train[train.columns.difference(['Id','Hazard'])] 

		print "prepared_train meta"
		print "shape", prepared_train.shape
		print prepared_train.head(3)

		ERF.fit(prepared_train,target_train)
		dt = ERF.predict(test[test.columns.difference(['Id'])])
		print "prediction score on cross validation"
		print ERF.score(cross_validation[cross_validation.columns.difference(['Id','Hazard'])],cross_validation_test)
		dt_cv = ERF.predict(cross_validation[cross_validation.columns.difference(['Id','Hazard'])])
		test['Hazard'] = self.clipForecastValue(dt)
		cross_validation['predicted_Hazard'] = self.clipForecastValue(dt_cv)

		names=  prepared_train.columns.values 

		print "sorted feature importance"
		print sorted(zip(map(lambda x: round(x, 4), ERF.feature_importances_), names), 
             reverse=True)

		#computing the Gini score 
		print "the Gini score"
		print self.gini_normalized(np.ravel(cross_validation[['Hazard']]),np.ravel(cross_validation[['predicted_Hazard']]))


		return test,cross_validation

	def ApplyBayesianRidge(self,train,test, cross_validation, full_train,config):
		BR = BayesianRidge(verbose=True,n_iter=1000,tol=0.00001)
		target_train = train[['Hazard']]
		cross_validation_test = cross_validation[['Hazard']]
		prepared_train = train[train.columns.difference(['Id','Hazard'])] 

		print "prepared_train meta"
		print "shape", prepared_train.shape
		print prepared_train.head(3)

		BR.fit(prepared_train,target_train)
		dt = BR.predict(test[test.columns.difference(['Id'])])
		print "prediction score on cross validation"
		print BR.score(cross_validation[cross_validation.columns.difference(['Id','Hazard'])],cross_validation_test)
		dt_cv = BR.predict(cross_validation[cross_validation.columns.difference(['Id','Hazard'])])
		test['Hazard'] = self.clipForecastValue(dt)
		cross_validation['predicted_Hazard'] = self.clipForecastValue(dt_cv)

		# print "sorted feature importance"
		# print sorted(zip(map(lambda x: round(x, 4), BR.feature_importances_), names), 
  		#	     reverse=True)

		print "regression model coefficients"
		print BR.coef_

		print "estimated precision of the noise"
		print BR.alpha_

		print "estimated precision of the weights"
		print BR.lambda_

		print "value of the objective function"
		print BR.scores_


		return test,cross_validation

	def clipForecastValue(self,data):
		#if a value is under a threshhold, replace it with the value of the threshhold 
		#in this competition no target value under 1
		data[data<1] = 1 
		return data

	def ApplyGradientBoostingRegressor(self,train,test, cross_validation, full_train,config):
		GBR = GradientBoostingRegressor(verbose=True,n_estimators=config['n_estimators'],learning_rate=1.5,max_depth=5)
		target_train = train[['Hazard']]
		cross_validation_test = cross_validation[['Hazard']]
		prepared_train = train[train.columns.difference(['Id','Hazard'])] 

		print "prepared_train meta"
		print "shape", prepared_train.shape
		print prepared_train.head(3)

		GBR.fit(prepared_train,target_train)
		dt = GBR.predict(test[test.columns.difference(['Id'])])
		print "prediction score on cross validation"
		print GBR.score(cross_validation[cross_validation.columns.difference(['Id','Hazard'])],cross_validation_test)
		dt_cv = GBR.predict(cross_validation[cross_validation.columns.difference(['Id','Hazard'])])
		test['Hazard'] = self.clipForecastValue(dt)
		cross_validation['predicted_Hazard'] = self.clipForecastValue(dt_cv)



		names=  prepared_train.columns.values 

		print "sorted feature importance"
		print sorted(zip(map(lambda x: round(x, 4), GBR.feature_importances_), names), 
             reverse=True)

		print "The i-th score train_score_[i] is the deviance (= loss) of the model at iteration i on the in-bag sample. If subsample == 1 this is the deviance on the training data."
		print GBR.train_score_

		# print "The collection of fitted sub-estimators."
		# print GBR.estimators_

		#computing the Gini score 
		print "the Gini score"
		print self.Gini(np.ravel(cross_validation[['Hazard']]),np.ravel(cross_validation[['predicted_Hazard']]))
		print self.gini_normalized(np.ravel(cross_validation[['Hazard']]),np.ravel(cross_validation[['predicted_Hazard']]))

		return test,cross_validation

	def ApplyAdaBoostRegressor(self,train,test, cross_validation, full_train,config):
		ABR = AdaBoostRegressor(loss=config['loss'],n_estimators=config['n_estimators'])
		target_train = train[['Hazard']]
		cross_validation_test = cross_validation[['Hazard']]
		prepared_train = train[train.columns.difference(['Id','Hazard'])] 

		print "prepared_train meta"
		print "shape", prepared_train.shape
		print prepared_train.head(3)

		ABR.fit(prepared_train,target_train)
		dt = ABR.predict(test[test.columns.difference(['Id'])])
		print "prediction score on cross validation"
		print ABR.score(cross_validation[cross_validation.columns.difference(['Id','Hazard'])],cross_validation_test)
		dt_cv = ABR.predict(cross_validation[cross_validation.columns.difference(['Id','Hazard'])])
		test['Hazard'] = self.clipForecastValue(dt)
		cross_validation['predicted_Hazard'] = self.clipForecastValue(dt_cv)

		names=prepared_train.columns.values 

		print "sorted feature importance"
		print sorted(zip(map(lambda x: round(x, 4), ABR.feature_importances_), names), 
             reverse=True)

		#computing the Gini score 
		print "the Gini score"
		print self.gini_normalized(np.ravel(cross_validation[['Hazard']]),np.ravel(cross_validation[['predicted_Hazard']]))

		return test,cross_validation


	def GridSearchForHyperparameterOptimization(self, clf, param_grid, labels, target):

		#clf : model 
		#param_grid : hyperparameter space 

		grid_search = GridSearchCV(clf, param_grid=param_grid,verbose=True, n_jobs=5)
		start = time()
		grid_search.fit(labels, target)

		print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
		      % (time() - start, len(grid_search.grid_scores_)))
		self.report(grid_search.grid_scores_)
		
		return 

	def gini(self,actual, pred, cmpcol = 0, sortcol = 1):		
	     assert( len(actual) == len(pred) )
	     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
	     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
	     totalLosses = all[:,0].sum()
	     giniSum = all[:,0].cumsum().sum() / totalLosses
	 
	     giniSum -= (len(actual) + 1) / 2.
	     return giniSum / len(actual)
 
 	def gini_normalized(self,a, p):
    	 return self.gini(a, p) / self.gini(a, a)

	# from http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
	# Utility function to report best scores
	def report(self,grid_scores, n_top=10):
	    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
	    for i, score in enumerate(top_scores):
	        print("Model with rank: {0}".format(i + 1))
	        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
	              score.mean_validation_score,
	              np.std(score.cv_validation_scores)))
	        print("Parameters: {0}".format(score.parameters))
	        print("")
	    return 

	def dataVisualization(self,dataFrame):

		dataFrame.plot()
		plt.show(block=False)

		figure_name = '_'.join(dataFrame.columns.values)

		plt.savefig("feature_hazard_vis/"+figure_name+".svg",format="svg")

		return 

	def PrepareOutputFile(self,data):
		data[['Id','Hazard']].to_csv('submission.csv',index=False)
		return 

	def ApplyLearningAlgorithm(self, learning_alg, ):
		pass 
		#if learning_alg == 1:
			



	def processInputArguments(self,args):
		parser = argparse.ArgumentParser(description="Kaggle property inspection prediction solver")

		#define input parameter for know products csv file 
		parser.add_argument('-la',
							'--learning-algorithm',
							type=int,
							dest='learning_algorithm',
							help='Define the learning strategy to use, 1: EnsembleRandomForestRegressor, 2: GradientBoostingRegressor, 3: ApplyBayesianRidge'
							)



		#apply the parser to the argument list 
		options = parser.parse_args(args)
		return vars(options)

	#implementing business logic 
	def main(self):
		
		options = self.processInputArguments(sys.argv[1:])
		#learning_algorithm = options["learning_algorithm"]

		print "loading data files"

		train_file = "train1.csv"
		test_file = "test.csv"
		cross_validation_file = "cross_validation.csv"
		full_train_file = "train.csv"

		non_important_feature_list = ["T2_V10","T2_V7","T1_V13","T1_V10","T2_V8","T1_V12","T1_V7","T1_V15"]
		#non_important_feature_list = []
		
		train = self.loadData(train_file)
		test = self.loadData(test_file)
		cross_validation = self.loadData(cross_validation_file)
		full_train = self.loadData(full_train_file)		
		
		#train, test, cross_validation, full_train = self.loadData(train_file, test_file,cross_validation_file, full_train) 

		train = self.DataPreparation(train)
		full_train = self.DataPreparation(full_train)
		test = self.DataPreparation(test)
		cross_validation = self.DataPreparation(cross_validation)

		


		#data exploration 

		#self.dataVisualization(cross_validation[['Hazard','T2_V12']])

		all_features = cross_validation.columns.difference(["Id","Hazard"])

		# for feature in all_features:
		# 	print "visualizing Hazard and",feature
		# 	self.dataVisualization(cross_validation[['Hazard',feature]])

		# print "finished visualizing features"

		train = self.filterOutNonImportantFeatures(train,non_important_feature_list)
		full_train = self.filterOutNonImportantFeatures(full_train,non_important_feature_list)
		test = self.filterOutNonImportantFeatures(test,non_important_feature_list)
		cross_validation = self.filterOutNonImportantFeatures(cross_validation,non_important_feature_list)

		#shuffle columns in the training data and see if that affects the accuracy
		

		# column_names = train.columns.values 

		# np.random.shuffle(column_names)

		# column_names = column_names.tolist()
		# print "colmns shuffled"
		# print column_names

		# train = train[column_names]
		# cross_validation = cross_validation[column_names]
		# full_train = full_train[column_names]
		# column_names.remove("Hazard")
		# #np.delete(column_names,["Hazard"])
		# test = test[column_names]

		#test1 = self.ApplyRandomForest(train,test, cross_validation, full_train)
		
		#takes very long time 
		#test = self.ApplySVM_SVC(train,test, cross_validation, full_train)
		#RF configuration object 






		## applying gridsearch to find optimal hyperparameters

		# use a full grid over all parameters
		# param_grid = {"max_depth": [None],
		#               "max_features": [1, 3, 10, 15, 20],
		#               "min_samples_split": [1, 3, 10, 15, 20],
		#               "min_samples_leaf": [1, 3, 10, 15, 20],
		#               "n_estimators":[600], #100,200,300,400,500
		#               "bootstrap": [False],
		#               "criterion": ["mse"]}

		# #RF_config1 = {'n_estimators':200, 'criterion':'entropy'}
		



		# clf = RandomForestRegressor(n_jobs=-1,verbose=True)

		# self.GridSearchForHyperparameterOptimization(clf, param_grid, train[train.columns.difference(['Id','Hazard'])] , np.ravel(train[['Hazard']]))


		#TODO: 
		# 
		# add adaboostRegressor by RandomForestRegressor as base for it 
		# http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#example-ensemble-plot-adaboost-regression-py


		#test2,cross_validation_predict = self.ApplyRandomForest(train,test, cross_validation, full_train, RF_config1)
		
		RF_config = {'n_estimators':1000, 'criterion':'mse'}
		test2,cross_validation_predict = self.EnsembleRandomForestRegressor(train,test, cross_validation, full_train, RF_config)
		
		# test2,cross_validation_predict = self.ApplyBayesianRidge(train,test, cross_validation, full_train, RF_config)
		
		## gradient Boosting regressor 
		#GBR_config = {'n_estimators':1000}

		#test2,cross_validation_predict = self.ApplyGradientBoostingRegressor(train,test, cross_validation, full_train, GBR_config)

		## Ada boost regressor 
		#ABR_config = {"loss":"square","n_estimators":200}
		#test2,cross_validation_predict = self.ApplyAdaBoostRegressor(train,test, cross_validation, full_train, ABR_config)

		#grid search parameter optimization for gradientBoostingRegression 
		
		# print "output cross_validation_predict"
		# print cross_validation_predict.shape 
		# print cross_validation_predict.head(3)

		# print "silicing output cross validation"
		# print cross_validation_predict[['Id','Hazard','predicted_Hazard']].shape
		# print cross_validation_predict[['Id','Hazard','predicted_Hazard']].head(3)
		
		# print "columns in cross validation"
		# print cross_validation_predict.columns.values 

		cross_validation_predict[['Id','T1_V1','T1_V1','Hazard','predicted_Hazard']] = cross_validation_predict[['Id','T1_V1','T1_V1','Hazard','predicted_Hazard']].astype(int)

		print "end slicing output cross validation"


		cross_validation_predict[["predicted_Hazard"]] = cross_validation_predict[["predicted_Hazard"]]


		self.dataVisualization(cross_validation_predict[['Hazard','predicted_Hazard']])

		#test2[["Hazard"]] = np.around(test2[["Hazard"]])
		test2[["Hazard"]] = test2[["Hazard"]]

		self.PrepareOutputFile(test2)

		# #keep graphs open 
		plt.show()
		return 

if __name__ == '__main__':
	proposition1 = KagglePropertyInspectionPrediction()
	proposition1.main()

