"""
@author: Mohamed Ali Jamaoui 
@description: Santander classification problem  

"""
from __future__ import print_function 
import pandas as pd 
import argparse 
import sys 
import logging 
import json
import numpy as np 
from sklearn.preprocessing import LabelEncoder 
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.decomposition import TruncatedSVD
import random 

"""
The scientist guidelines: 

step1: 
	create a RandomForest based model: DONE 
		split the data to train and test (10%) and use a RandomForest from sklearn with default parameters (n_jobs=-1 for mutlithreading) 

step2: 
	remove constant columns with coefficient of variation: DONE  
	how does it improve the performance? 

step3: 
	remove highly correlated features: DONE (didn't improve)
	how does that improves the perforamance
		=> didn't improve the results on the leaderboard! 
			=> need further study to get more understanding out of it

step4:

	remove features with low importance based on random forest score 


step5:
	DONE: didn't improve 
	use hyperparameter optimization to get the best out of a single model. 

step6: 
	train several models and assess their correlation => ensemble non correlated models 

step7: compare gridseach and randomizedsearch results to bayesian optimization results 

TIPS: 

use cross validation to ensure stability 

TODO next: 
	ensemble models with votingClassifier : DONE 
	add zero count column : 
	add bagging : https://www.kaggle.com/skyjiao/santander-customer-satisfaction/xgboost-bagging/code 
		=> check difference between this manual bagging and the baggingClassifier from sklearn 

	add zero count as a feature: 
		https://www.kaggle.com/avishek/santander-customer-satisfaction/xgb-lalala-in-python/code 

	clipping min max values on test based on values from train 
	https://www.kaggle.com/yangnanhai/santander-customer-satisfaction/xgb-lalala-cap-lims2/run/214900 

	check PCA and calibrated classifier idea from here: 
		https://www.kaggle.com/godbless/santander-customer-satisfaction/feature-selection-1/code 

	check the domain knowledge from here: 
		
		https://www.kaggle.com/zfturbo/santander-customer-satisfaction/to-the-top-v3/run/219762 

	check the submitted blending stuff: 
		https://www.kaggle.com/cast42/santander-customer-satisfaction/blending-2xrf-2xet-xgb/code 

	tried this with no improvement: 
	https://www.kaggle.com/cast42/santander-customer-satisfaction/blending-2xrf-2xet-xgb/code 


	try to create an automation interface to set up experiments and run them in parallel. 

	>> understanding feature ranges is useful: 

		https://www.kaggle.com/zfturbo/santander-customer-satisfaction/feature-ranges 

TODO: 
	spit the pre analysis out into a seperate file: correlation, variance, etc. 

"""

# fix the seed to ensure that experiments are reproducible 
np.random.seed(1234) #keep this to be consistent with other competitors in Kaggle 
#np.random.seed(4294967295)
#random.seed(1234)# xgb uses this (appearently) => to be verified 

class SantanderClassifier():

	def __init__(self):
		self.ID = ""
		self.target = "" 
		self.columns_to_exclude = []
		self.correlation_threshold = 0.5
		self.saved_Models_Folder = "./"
		self.ensemble_models = []

	def normalizeData(self, train, test):
		ss = StandardScaler()
		train[train.columns.difference([self.ID, self.target])] = ss.fit_transform(train[train.columns.difference([self.ID, self.target])])
		test[test.columns.difference([self.ID])] = ss.transform(test[test.columns.difference([self.ID])])
		#print(test.head())
		return train, test

	def reduceDimensionality(self, train, test):
		pca = TruncatedSVD(n_components=60, n_iter=5, random_state=1234)
		
		train = pca.fit_transform(train[train.columns.difference([self.ID, self.target])])
		test = pca.transform(test[test.columns.difference([self.ID])])
		
		print(train.shape, test.shape )
		return train, test

	def LabelEncodeCategoricals(self, data, categorical_features):
		data[categorical_features] = data[categorical_features].apply(LabelEncoder().fit_transform)
		return data

	def oneHotEncodeCategoricals(self, data, categorical_features_to_one_hot):
		#TODO 
		pass

	def customDataPreprocessing(self, train, test):

		"""
		1- what to do with var3 == -999999

		https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999/comments
		https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19367/var3/111169#post111169

		"""
		#train = train.replace(-999999,2)
		#test = test.replace(-999999,2)

		### add zero count per row 
		#--add zero count per row--
		logging.info("adding zero counts per row")
		def zero_count(x):
		    return sum(x==0)
		train['zero_count'] = train.apply(zero_count,axis=1)
		test['zero_count'] = test.apply(zero_count,axis=1)

		return train, test


	def readConfigurationFile(self, configuration_file_name):
		"""
		@description: read a json configuration file 
		@param : configuration file name 
		@return : the configuration object  
		"""
		with open(configuration_file_name, "r") as config_file:
			configuration_info = json.load(config_file)

		return configuration_info

	def trainWithKFolding(self, data):
		logging.info("preparing KFolds")
		kf = KFold(train.shape[0], n_folds=5)

		for train_index, test_index in kf:
			current_train, current_test = train.ix[train_index],train.ix[test_index]
			print(current_train.head())
			print(current_test.head())

		return

	def ModelerScientist(self, data):
		#TODO
		pass

	"""
	def xgbOriginal(self, train, cross_validation, full_train, mode="train"):
		logging.info("starting XGboost training")
		logging.info("preparing Target Variable for train and cross validation")
		train_Y = train[self.target]
		cross_validation_Y = cross_validation[self.target]
		logging.info("preparing train and CV data")
		train_X = train[train.columns.difference([self.ID, self.target])]
		cross_validation_X = cross_validation[cross_validation.columns.difference([self.ID,self.target])]
		
		logging.info("Preparing xgboost model")

		params = {}
	    params["objective"] = "binary:logistic"
	    params["eta"] = 0.03
	    params["subsample"] = 0.8
	    params["colsample_bytree"] = 0.7
	    params["silent"] = 1
	    params["max_depth"] = 5
	    params["min_child_weight"] = 1
	    params["eval_metric"] = "auc"

	    dtrain = xgb.DXMatrix(train_X, label=train_Y)
	    dtest = xgb.DXMatrix(cross_validation_X, label=cross_validation_Y)


	    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        clf = xgb.train(params, train_X, 10,
                        evals=watchlist, early_stopping_rounds=50,
                        verbose_eval=False)

        clf.

        return 

        """ 


	def XGboostTrain(self, train, cross_validation, full_train, mode="train"):
		logging.info("starting XGboost training")
		logging.info("preparing Target Variable for train and cross validation")
		train_Y = train[self.target]
		cross_validation_Y = cross_validation[self.target]
		logging.info("preparing train and CV data")
		train_X = train[train.columns.difference([self.ID, self.target])]
		cross_validation_X = cross_validation[cross_validation.columns.difference([self.ID,self.target])]
		
		logging.info("Preparing xgboost model")
		
		"""clf = xgb.XGBClassifier(
			    max_depth = 5,
		        n_estimators=500,
		        learning_rate=0.03, 
		        nthread=-1,
		        subsample=0.7815,
		        colsample_bytree=0.801
		        )
		{'max_depth': (5, 10),
		                          'learning_rate': (0.01, 0.3),
		                          'n_estimators': (50, 1000),
		                          'gamma': (1., 0.01),
		                          'min_child_weight': (2, 10),
		                          'max_delta_step': (0, 0.1),
		                          'subsample': (0.7, 0.8),
		                          'colsample_bytree' :(0.5, 0.99)
		
	                          }
	            """
		
		"""
		experiment timestamp: 24 / 04 / 2016 : 6:18AM 
		clf = xgb.XGBClassifier(
			    max_depth = 6,
		        n_estimators=460,
		        learning_rate=0.240, 
		        nthread=-1,
		        subsample=0.7712,
		        colsample_bytree=0.7255,
		        gamma=0.1, 
		        min_child_weight=4,
		        )
		"""
		# missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242 
		clf = xgb.XGBClassifier(
				missing=np.nan,
			    max_depth =5,
		        n_estimators=350,
		        learning_rate=0.03, 
		        nthread=-1,
		        subsample=0.95,
		        colsample_bytree=0.85)
		        #seed=4242)

		self.ensemble_models.append(clf)

		if mode != "train":
			return clf

		#logging.info("applying dimensionality reduction")
		#train, test = self.reduceDimensionality(train, test)

		clf.fit(train_X, train_Y, eval_metric="auc", early_stopping_rounds=20, eval_set=[(train_X, train_Y)])
		#clf.fit(train_X, train_Y, eval_metric="auc", early_stopping_rounds=20, eval_set=[(cross_validation_X, cross_validation_Y)])

		print('Overall AUC on whole train set:', roc_auc_score(train_Y, clf.predict_proba(train_X)[:,1]))
		print('Overall AUC on whole cross_validation set:', roc_auc_score(cross_validation_Y, clf.predict_proba(cross_validation_X)[:,1]))
		
		logging.info('classification_report on train')
		print(classification_report(train_Y, clf.predict(train_X)))
		
		logging.info('classification_report on cross_validation')
		print(classification_report(cross_validation_Y, clf.predict(cross_validation_X)))
		
		logging.info("Preparing Target Variable for full train")
		train_Y = full_train[self.target]
		logging.info("Preparing train and CV data")
		train_X = full_train[train.columns.difference([self.ID, self.target])]
		
		#clf.fit(train_X, train_Y, eval_metric="auc", eval_set=[(train_X, train_Y)])

		
		return clf


	def quickInitialTrain(self, train, cross_validation, mode="train"):		
		# perform a quick random forest training 
		logging.info("Starting quick training")
		logging.info("preparing Target Variable for train and cross validation")
		train_Y = train[self.target]
		cross_validation_Y = cross_validation[self.target]
		logging.info("preparing train and CV data")
		train_X = train[train.columns.difference([self.ID, self.target])]
		cross_validation_X = cross_validation[cross_validation.columns.difference([self.ID,self.target])]

		logging.info("Preparing the RandomForestClassifier model")
		clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_features=0.33, n_jobs=-1, verbose=1)

		self.ensemble_models.append(clf)

		if mode != "train":
			return clf

		logging.info("fitting the RandomForestClassifier model")
		clf.fit(train_X,train_Y)
		
		print("accurary on cross_validation set", clf.score(cross_validation_X, cross_validation_Y))
		print('Overall RFC AUC on whole train set:', roc_auc_score(train_Y, clf.predict_proba(train_X)[:,1]))
		print('Overall RFC AUC on whole cross_validation set:', roc_auc_score(cross_validation_Y, clf.predict_proba(cross_validation_X)[:,1]))

		return clf

	def ensembleModel(self, list_of_models, train, cross_validation):
		logging.info("preparing Target Variable for train and cross validation")
		train_Y = train[self.target]
		cross_validation_Y = cross_validation[self.target]
		logging.info("preparing train and CV data")
		train_X = train[train.columns.difference([self.ID, self.target])]
		cross_validation_X = cross_validation[cross_validation.columns.difference([self.ID,self.target])]

		clf = VotingClassifier(estimators=list_of_models, voting='soft', weights=[1,5])
		clf.fit(train_X,train_Y)

		self.saveModel(clf, "rfc_and_xgb_model")
		
		print("accurary on cross_validation set", clf.score(cross_validation_X, cross_validation_Y))
		print('Overall RFC AUC on whole train set:', roc_auc_score(train_Y, clf.predict_proba(train_X)[:,1]))
		print('Overall RFC AUC on whole cross_validation set:', roc_auc_score(cross_validation_Y, clf.predict_proba(cross_validation_X)[:,1]))

		return clf

	def baggingOfXbg(self, train, cross_validation, full_train, mode="train"):

		logging.info("preparing Target Variable for train and cross validation")
		train_Y = train[self.target]
		cross_validation_Y = cross_validation[self.target]
		logging.info("preparing train and CV data")
		train_X = train[train.columns.difference([self.ID, self.target])]
		cross_validation_X = cross_validation[cross_validation.columns.difference([self.ID,self.target])]

		from sklearn.ensemble import BaggingClassifier

		clf = xgb.XGBClassifier(
			    max_depth = 5,
		        n_estimators=300,
		        learning_rate=0.2, 
		        nthread=-1,
		        subsample=0.7815,
		        colsample_bytree=0.801
		        )

		"""
		clf = xgb.XGBClassifier(
			    max_depth = 7.4833,
		        n_estimators=807,
		        learning_rate=0.1995, 
		        nthread=-1,
		        subsample=0.7712,
		        colsample_bytree=0.7255,
		        gamma=1.0000, 
		        max_delta_step=0.0417, 
		        min_child_weight=4.0151, 
		        )
		""" 
		clf.fit(train_X, train_Y, eval_metric="auc", eval_set=[(train_X, train_Y)])

		logging.info("bagging classifiers")
		bagging = BaggingClassifier(clf, max_samples=0.3, max_features=0.7, warm_start=True, verbose=1)

		logging.info("fitting the bag of classifiers")
		bagging.fit(train_X, train_Y)
		print("accurary on cross_validation set", clf.score(cross_validation_X, cross_validation_Y))
		print('Overall RFC AUC on whole train set:', roc_auc_score(train_Y, clf.predict_proba(train_X)[:,1]))
		print('Overall RFC AUC on whole cross_validation set:', roc_auc_score(cross_validation_Y, clf.predict_proba(cross_validation_X)[:,1]))

		logging.info("Preparing Target Variable for full train")
		train_Y = full_train[self.target]
		logging.info("Preparing train and CV data")
		train_X = full_train[train.columns.difference([self.ID, self.target])]

		logging.info("continue fitting with cross_validation data")
		#bagging.fit(train_X, train_Y, eval_metric="auc", eval_set=[(train_X, train_Y)])
		bagging.fit(cross_validation_X, cross_validation_Y)

		return bagging 


	def variance_analysis(self, data, variance_threshold=0.0):
		#TODO document parameters 
		logging.info("data assessment based on variance threshold")
		from sklearn.feature_selection import VarianceThreshold
		data = data[data.columns.difference([self.ID, self.target])]
		selector = VarianceThreshold(threshold=variance_threshold)
		selector.fit(data)
		mask = ~selector.get_support()
		self.columns_to_exclude = self.columns_to_exclude + list(data.columns[mask])
		
		print(self.columns_to_exclude)
		
		return

	def correlation_analysis(self, data):

		print("duplicate columns")
		from pandas.core.common import array_equivalent
		# Kaggle forums way to remove duplicates 
		def duplicate_columns(frame):
		    groups = frame.columns.to_series().groupby(frame.dtypes).groups
		    dups = []

		    for t, v in groups.items():

		        cs = frame[v].columns
		        vs = frame[v]
		        lcs = len(cs)

		        for i in range(lcs):
		            ia = vs.iloc[:,i].values
		            for j in range(i+1, lcs):
		                ja = vs.iloc[:,j].values
		                if array_equivalent(ia, ja):
		                    dups.append(cs[i])
		                    break

		    return dups

		redundant_cols = duplicate_columns(data)
		self.columns_to_exclude = self.columns_to_exclude + redundant_cols
		return 
		"""
		Performing correlation analysis to find redundant, linearly correlated features 
		"""
		logging.info("redundancy analysis based on correlation analysis")
		data = data[data.columns.difference([self.ID, self.target])]
		correlation = data.corr()
		np.fill_diagonal(correlation.values, 0)

		attrs = correlation.iloc[:,:] # all except target
		threshold = self.correlation_threshold

		important_corrs = (attrs[abs(attrs) > threshold]).unstack().dropna().to_dict()

		#print(important_corrs)
		#     attribute pair  correlation
		# 0     (AGE, INDUS)     0.644779
		# 1     (INDUS, RAD)     0.595129
		# ...
		
		unique_important_corrs = pd.DataFrame(
		    list(set([(key[0],key[1], important_corrs[key]) for key in important_corrs])), columns=['column','correlated_column', 'correlation'])
		
		# sorted by absolute value
		unique_important_corrs = unique_important_corrs.ix[
		    abs(unique_important_corrs['correlation']).argsort()[::-1]]


		unique_important_corrs = unique_important_corrs[unique_important_corrs["correlation"] == 1]
		unique_important_corrs.to_csv("redundant_columns.csv", sep="|", index=False)
		redundant_column = pd.unique(unique_important_corrs["correlated_column"].values.ravel())
		
		print("correlation based")
		print(list(redundant_column))

		self.columns_to_exclude = self.columns_to_exclude + list(redundant_column)
		
		return 

	def saveModel(self, model, filename="filename"):
		return joblib.dump(model, self.saved_Models_Folder+filename+'pkl') 

	def loadModel(self, model, filename="filename"):
		return joblib.load(model, self.saved_Models_Folder+filename+'pkl')

	def tune_xgb_parameters(self, train, y):

		def xgboostcv(
			  n_estimators,
		      learning_rate,
		      silent=True,
		      nthread=-1):
			return cross_val_score(xgb.XGBClassifier(missing=np.nan,
											     max_depth=5,
			                                     learning_rate=learning_rate,
			                                     n_estimators=int(n_estimators),
			                                     silent=silent,
			                                     nthread=nthread,
			                                     subsample=0.95,
			                                     colsample_bytree=0.85),
			                   train,
			                   y,
			                   "roc_auc",
			                   cv=5).mean()


		# Load data set and target values
		"""
				clf = xgb.XGBClassifier(
				missing=np.nan,
			    max_depth =5,
		        n_estimators= 300,
		        learning_rate=0.04, 
		        nthread=-1,
		        subsample=0.95,
		        colsample_bytree=0.85)
		        #seed=4242)

		"""

		xgboostBO = BayesianOptimization(xgboostcv,
		                         {'n_estimators': (350, 420),
		                          'learning_rate':(0.03,0.21)
		                          })

		xgboostBO.maximize()
		print('-'*53)

		print('Final Results')
		print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])

		return

	def deepLearningKeras(self):

		"""
		implement a classifier with keras 
		"""

		return 

	def prepareSubmission(self, data, model):
		logging.info("predicting on test data")
		proba = model.predict_proba(data[data.columns.difference([self.ID])])
		data["TARGET"] = proba[:,1]

		#didn't improve 
		#data = self.DomainKnowledge(data)

		#didn't improve 
		#data[data["TARGET"]<0.02]["TARGET"] = 0
		logging.info("exporting kaggle submission results")
		data[["ID","TARGET"]].to_csv("Santander_submission.csv", index=False)
		return

	def DomainKnowledge(self, prediction):

		"""
		#link: 
		https://www.kaggle.com/cvikas/santander-customer-satisfaction/people-with-loans-never-complain/code

		# Under 23 year olds are always happy
		preds[var15 < 23] = 0
		preds[saldo_medio_var5_hace2 > 160000]=0
		preds[saldo_var33 > 0]=0
		preds[var38 > 3988596]=0
		preds[V21>7500]=0
		"""

		# Under 23 year olds are always happy
		prediction[prediction["var15"] < 23]["TARGET"] = 0  
		prediction[prediction["saldo_medio_var5_hace2"] > 160000]["TARGET"] = 0  
		prediction[prediction["saldo_var33"] > 0]["TARGET"] = 0  
		prediction[prediction["var38"] > 3988596]["TARGET"] = 0  
		#prediction[prediction["V21"] > 7500]["TARGET"] = 0

		return prediction 

	def processInputArguments(self,args):
		parser = argparse.ArgumentParser(description="Starter code for data science projects")

		#train data file name 
		parser.add_argument('-td',
							'--training-data',
							type=str,
							dest='training_data',
							help='training data file'
							)

		#test data file name 
		parser.add_argument('-tsd',
							'--test-data',
							type=str,
							dest='test_data',
							help='test data file'
							)
		
		#test data file name 
		parser.add_argument('-jc',
							'--json-configuration-file',
							type=str,
							dest='json_configuration_file',
							help='json configuration file'
							)

		## show help if no arguments passed 
		if len(sys.argv)==1:
			parser.print_help()
			sys.exit(1)

		#apply the parser to the argument list 
		options = parser.parse_args(args)
		return vars(options)

	def main(self):

		#########################################
		##### Set up python logging format ######
		#########################################

		log_format='%(asctime)s %(levelname)s %(message)s'
		logging.basicConfig(format=log_format, level=logging.INFO)

		#######################################################
		##### Set up command command line parsing config ######
		####################################################### 
		options = self.processInputArguments(sys.argv[1:])
		training_data = options["training_data"]
		test_data = options["test_data"]
		json_configuration_file = options["json_configuration_file"]

		##################################################
		##### loading json configuration parameters ######
		################################################## 
		logging.info("parsing json configuration file")
		json_configs = self.readConfigurationFile(json_configuration_file)
		self.ID = json_configs["ID"]
		self.target = json_configs["target"]
		self.saved_Models_Folder = json_configs["saved_models_folder"]

		###############################
		##### load training data ######
		############################### 

		logging.info("loading training data")
		train = pd.read_csv(training_data, delimiter=",")
		logging.info("loading test data")
		test = pd.read_csv(test_data, delimiter=",")
		
		###############################
		##### data preprocessing ######
		###############################

		#TODO update code to not have to recompute the data preocessing steps each new run 
		#train, test = self.customDataPreprocessing(train, test)

		###############################

		logging.info("performing variance analysis")
		self.variance_analysis(train, 0.0)
		
		# Note: 16 04 2016 correlation analysis degraded the results. 
		self.correlation_analysis(train) 
		
		return 

		logging.info("dropping_excluded_columns")
		train.drop(self.columns_to_exclude, axis=1, inplace=True)
		test.drop(self.columns_to_exclude, axis=1, inplace=True)


		logging.info("normalizing the data")
		train, test = self.normalizeData(train, test)


		# perform an intial train based on the data as is without much
		#logging.info("Now the modeler scientist is tackling your data, let him do the magic") 
		#self.ModelerScientist(data)

		##########################################
		####### run parameter tuning on xgb ######
		
		#logging.info("parameter tuning")
		#self.tune_xgb_parameters(train.drop([self.ID, self.target], axis=1), train["TARGET"])

		#logging.info("finished parameter tuning")
		### results of parameter tuning 
		"""
		 results of parameter tuning 

		 Step |   Time |      Value |   colsample_bytree |     gamma |   learning_rate |   max_delta_step |   max_depth |   min_child_weight |   n_estimators |   subsample | 
		   16 | 20m50s |    0.84038 |             0.7255 |    1.0000 |          0.1995 |           0.0417 |      7.4833 |             4.0151 |       806.8057 |      0.7712 | 
		
		Final Results
		XGBOOST: 0.840376
		2016-04-21 11:41:04,674 INFO finished parameter tuning

		real	393m4.161s
		user	1195m38.296s
		sys	2m59.100s

		best settings:

		clf = xgb.XGBClassifier(
			    max_depth = 8,
		        n_estimators=807,
		        learning_rate=0.1995, 
		        nthread=-1,
		        subsample=0.7712,
		        colsample_bytree=0.7255,
		        gamma=1.0000, 
		        max_delta_step=0.0417, 
		        min_child_weight=4.0151, 
		        )


		"""

		#return 
		###########################################
		###########################################

		logging.info("splitting data into train and cross validation set")
		train_, cv = train_test_split(train, test_size=0.1)

		#model1 = self.quickInitialTrain(train_, cv, "create")
		#final_model = self.quickInitialTrain(train_, cv, "train")
		final_model = self.XGboostTrain(train_, cv, train, "train")
		#final_model = self.baggingOfXbg(train_, cv, train, "train")
		#self.ensemble_models = [('rfc', model1),('xgbc', model2)]
		#final_model = self.ensembleModel(self.ensemble_models, train_, cv)

		#############################################################################################################
		###### this data set should only be used for the final validation of the model, don't use it for the CV #####
		#############################################################################################################

		self.prepareSubmission(test, final_model)

		return 


if __name__ == '__main__':
	bp = SantanderClassifier()
	bp.main()