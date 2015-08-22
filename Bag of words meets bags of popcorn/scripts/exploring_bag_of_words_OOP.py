
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 

data_path="C:\\Users\\MedAli\\Desktop\\kaggle_competitions\\Bag of words meets bags of popcorn\\data\\"

train = pd.read_csv(data_path+"labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
#quoting=3 to ignore the doubled quotes

###print the dimensions of the data: rows and columns
##mm = train.shape
##print mm[0]
###see the column names
##print train.columns.values
##
###see few reviews
##print train["review"][0]
##
###data cleaning and text processing
##
###[train["review"][i] = BeautifulSoup(train["review"]).get_text() for i in xrange(mm[0])]
##
##print train["review"][0]

def review_to_words(raw_review):

    review_text = BeautifulSoup(raw_review).get_text()

    review_text = re.sub("[0-9\.]+", "NUM", review_text) 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    
    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if w not in stops]

    return " ".join(meaningful_words)



#clean_review = review_to_words(train["review"][0])



num_reviews = train["review"].size

clean_train_reviews = []

print "Cleaning and parsin the training set movie reviews...\n"
for i in xrange(0, num_reviews):
    if((i+1)%1000 == 0):
        print "review %d of %d\n" % (i+1, num_reviews)
    clean_train_reviews.append(review_to_words(train["review"][i]))
    


print "creating the bag of words \n"
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word", tokenizer = None, preprocessor= None, stop_words = None, max_features = 1000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

print "dimension of train_data_features:", train_data_features.shape

#vocab = vectorizer.get_feature_names()

print "training the random forest"

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data_features, train["sentiment"])

print "preparing a submission"

test= pd.read_csv(data_path+"testData.tsv", header=0, delimiter="\t", quoting=3)

print "test shape", test.shape

num_reviews = len(test["review"])
clean_test_reviews = []

print "clearning and parsing the test set movie reviews..\n"
for i in xrange(0, num_reviews):
    if((i+1)%1000 == 0):
        print "review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id":test["id"], "sentiment": result})
 
output.to_csv("Bag_of_words_model.csv", index=False, quoting=3)

