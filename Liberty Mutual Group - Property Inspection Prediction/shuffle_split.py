import pandas as pd
import numpy as np 
import sys
import argparse 

# data_path = "/home/medalijamaoui/Desktop/workspace/sku_clone/data/"

# df = pd.read_csv(data_path+"product.dlm", header=0, delimiter="|")
# ts =  df.shape 
# # print "data dimension", ts
# # print "product attributes \n", train.columns.values 


# #shuffle data set, and split to train and test set. 
# new_train = df.reindex(np.random.permutation(df.index))

# indice_90_percent = int((ts[0]/100.0)* 90)


# new_train[:indice_90_percent].to_csv('train_products.txt',header=True, sep="|", index=False)
# new_train[indice_90_percent:].to_csv('test_products.txt',header=True, sep="|", index=False)

class prepareTestTrainSets():
	def __init__(self): 
		pass

	def loadData(self,data_file_path): 
		input_data = pd.read_csv(data_file_path, header=0, delimiter="|",dtype=object)
		print "metadata: ",input_data.shape
		return input_data


	def shuffleToRandomize(self,input_data):
		shuffled_data = input_data.reindex(np.random.permutation(input_data.index))
		return shuffled_data 


	def exportCSVData(self, train_percent, shuffled_data,train_file_name, test_file_name):

		indice_percent = int((shuffled_data.shape[0]/100.0)* train_percent)

		shuffled_data[:indice_percent].to_csv('train_products.txt',header=True, sep="|", index=False)
		shuffled_data[indice_percent:].to_csv('test_products.txt',header=True, sep="|", index=False)
		return 


	def processInputArguments(self, args):

		parser = argparse.ArgumentParser(description="shuffles and splits a given data files into train and test files with a given percent")


		#
		parser.add_argument('-df',
							'--data-file',
							type=str,
							dest='input_data_file',
							help='Define the path (absolute or relative) of the csv products file'
							)

		parser.add_argument('-tsfn',
							'--output-test-name',
							type=str,
							default='test_file.csv',
							dest='output_test_file_name',
							help='the output test file name'
							)


		parser.add_argument('-trfn',
							'--output-train-name',
							type=str,
							default='train_file.csv',
							dest='output_train_file_name',
							help='the output train file name'
							)

		parser.add_argument('-pr',
							'--train-percent',
							type=float,
							default=90,
							dest='train_set_percent',
							help='an integer describing the percentage of data to keep for training set')


		#apply the parser to the argument list 
		options = parser.parse_args(args)
		return vars(options)


	#Description: shuffle and split business logic 
	def main(self):
		options = self.processInputArguments(sys.argv[1:])
		input_data_file = options["input_data_file"]
		output_test_file_name = options["output_test_file_name"]
		output_train_file_name = options["output_train_file_name"]
		train_set_percent = options["train_set_percent"]

		print "loading data.."
		data_dataframe = self.loadData(input_data_file)
		print "shuffling data.."
		shuffled_data = self.shuffleToRandomize(data_dataframe)

		print "explorting csv files"
		self.exportCSVData(train_set_percent, shuffled_data, output_train_file_name, output_test_file_name)

		return 


if __name__ == '__main__':
	prepareTestTrainSetsObject = prepareTestTrainSets()
	prepareTestTrainSetsObject.main()
	

