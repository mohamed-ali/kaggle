"""
@author: Mohamed Ali Jamaoui 
@description: starter code for data science projects - sample randomForest learner  

"""

from __future__ import print_function 
import pandas as pd 
import argparse 
import sys 
import logging 
import json 

"""
TODO: 

	0- prepare the submission and skelton for the solution 


	1- Features ranges: 

		- for each feature, find the min and the max and median and mean values 
		example: https://www.kaggle.com/zfturbo/santander-customer-satisfaction/feature-ranges 
		- export the results into a file and share it with us via FB or github 

	2- Value counts: 

		- For each categorical features, count the number of instances (rows) associated with each value 
		export the results into a file and share it with us via FB or github 
		- export the results into a file and share it with us via FB or github 
	3- Compute the percentange of missing values in each data columns:

		- for each column compute the percentage of missing values. 
		- note that missing values can be encoded as 99999 or -1 or else. Check the forums and the data description 
		- export the results as a report and share it with us 

	4- Normalize the data: 

		- For each column perform standard normalization of the data 
		- export the normalized data set 

	5- Extract feature importance with Random Forest 

		- train a random forest model and extract the feature importance values
		- export the importance as a score and share it with us 

	6- Include the PCA components 2 or 3 of them instead of all the destination variables values for a given hotel 

		- train a PCA and extract the 2 or 3 components and add them to the data set. 
		- the train the model and see how the results of the machine learning algorithm change with and without PCA 

	7- Use gridsearch to find the best parameters for a given machine learning algorithm 

		- use a hyper parameter optimization method to find the best parameters for a given machine learning model 

	8- Use randomized search to find the best parameters for a given machine learning algorithm 

		- use a hyper parameter optimization method to find the best parameters for a given machine learning model 

	9- Use bayesian optimization to find the best parameters for a given machine learning algorithm 

		- use a hyper parameter optimization method to find the best parameters for a given machine learning model 

	10- change the learning model from RanfomForestClassifier to AdaBoostClassifier and compare results 

		- change the learning model from one classifier to another and compare results

	11- Create a deep learning classifier with Keras, (using Torch or tensorflow as the underlying engine)

		- change the RandomForestClassifier to use deep learning classifier with Keras instead. 
		- compare the results 

	12- use linear blending of many classifiers and see the results 

	



"""

class ExpediaRecommender():
	def __init__(self):
		self.chunk_size=1000000

	def readConfigurationFile(self, configuration_file_name):
		"""
		@description: read a json configuration file 
		@param : configuration file name 
		@return : the configuration object  
		"""
		with open(configuration_file_name, "r") as config_file:
			configuration_info = json.load(config_file)

		return configuration_info

	def importDestination(self, data):

		logging.info("transforming destination data")
		
		pass 

	def addDestinationInfo(self, train, test, destinations):

		train = pd.merge(train, destinations, on="srch_destination_id", how="inner")
		test = pd.merge(train, destinations, on="srch_destination_id", how="inner")

		return train, test

	def removeClickData(self, train):
		logging.info("removing click data")		
		return train[train["is_booking"] == True]


	def preprocessData(self, data_file, output_file_name):
		logging.info("preprocessing the data")


		logging.info("removing click data")
		for index, chunk in enumerate(pd.read_csv(data_file, chunksize=self.chunk_size)):
			
			logging.info("processing chunk "+str(index))
			
			print(index)
			print(chunk.head(1))

			chunk = self.removeClickData(chunk)
			
			if index == 0: 
				chunk.to_csv(output_file_name, sep=",", index=False)
			else:
				chunk.to_csv(output_file_name, sep=",", mode='a', index=False, header=None)
		

		return 


	def processInputArguments(self,args):
		parser = argparse.ArgumentParser(description="Starter code for data science projects")

		#train data file name 
		parser.add_argument('-td',
							'--training-data',
							type=str,
							dest='training_data',
							help='training data file'
							)
		
		parser.add_argument('-dst',
							'--destinations',
							type=str,
							dest='destinations',
							help='destinations data file'
							)

		#test data file name 
		parser.add_argument('-tsd',
							'--test-data',
							type=str,
							dest='test_data',
							help='test data file'
							)
		
		#configuration file name 
		parser.add_argument('-jc',
							'--json-configuration-file',
							type=str,
							dest='json_configuration_file',
							help='json configuration file'
							)

		# 
		parser.add_argument('-pr',
							'--preprocessing',
							type=int,
							default=0,
							dest='preprocessing',
							help='preprocessing'
							)

		## show help if no arguments passed 
		if len(sys.argv)==1:
			parser.print_help()
			sys.exit(1)

		#apply the parser to the argument list 
		options = parser.parse_args(args)
		return vars(options)

	def main(self):

		##### Set up python logging format ###### 
		log_format='%(asctime)s %(levelname)s %(message)s'
		logging.basicConfig(format=log_format, level=logging.INFO)


		##### Set up command command line parsing config ###### 
		options = self.processInputArguments(sys.argv[1:])
		training_data = options["training_data"]
		test_data = options["test_data"]
		destinations_file = options["destinations"]
		json_configuration_file = options["json_configuration_file"]
		preprocessing = options["preprocessing"]

		##### loading json configuration parameters ###### 
		json_configs = self.readConfigurationFile(json_configuration_file)
		self.chunk_size = json_configs["chunk_size"]

		if preprocessing == 1: 
			self.preprocessData(training_data,training_data+"_without_click")
			return 

		##### load training data ###### 
		logging.info("loading training data")
		train = pd.read_csv(training_data, delimiter=",")
		print(train.head(3))
		
		logging.info("loading test data")
		test = pd.read_csv(test_data, delimiter=",")
		print(test.head(3))
		
		logging.info("loading destinations data")
		destinations = pd.read_csv(destinations_file, delimiter=",")
		print(destinations.head(3))

		###### aggregating destinations information ##########

		logging.info("adding destinations information")

		train, test = self.addDestinationInfo(train, test, destinations)

		print(train.head(3))
		print(test.head(3))

		########################################
		
		logging.info("doing some data cleansing stuff")
		logging.info("doing some cool machine learning ")		

		logging.info("finishing and writing out results")

		return 


if __name__ == '__main__':
	bp = ExpediaRecommender()
	bp.main()