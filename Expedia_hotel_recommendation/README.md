## Kaggle competition: Expedia Hotel Recommendation 

### Data Preprocessing 

uncompressed training data 3.8G 

```
$ ll -h train.csv 
-rw-rw-r-- 1 dalijamaoui dalijamaoui 3.8G Apr 21 11:36 train.csv
```

uncompressed training data without click information: 310M 

```
$ ll -h train.csv_without_click 
-rw-rw-r-- 1 dalijamaoui dalijamaoui 310M May  8 19:21 train.csv_without_click
```

### Command line arguments 

```
$ time python script/ExpediaRecommender.py 
usage: ExpediaRecommender.py [-h] [-td TRAINING_DATA] [-dst DESTINATIONS]
                             [-tsd TEST_DATA] [-jc JSON_CONFIGURATION_FILE]
                             [-pr PREPROCESSING]

Starter code for data science projects

optional arguments:
  -h, --help            show this help message and exit
  -td TRAINING_DATA, --training-data TRAINING_DATA
                        training data file
  -dst DESTINATIONS, --destinations DESTINATIONS
                        destinations data file
  -tsd TEST_DATA, --test-data TEST_DATA
                        test data file
  -jc JSON_CONFIGURATION_FILE, --json-configuration-file JSON_CONFIGURATION_FILE
                        json configuration file
  -pr PREPROCESSING, --preprocessing PREPROCESSING
                        preprocessing

real	0m0.335s
user	0m0.276s
sys	0m0.052s
```


### example

```
time python ExpediaRecommender.py -td ../data/train.csv -dst ../data/destinations.csv.gz -tsd ../data/test.csv.gz -jc config.json -pr 1 
```  