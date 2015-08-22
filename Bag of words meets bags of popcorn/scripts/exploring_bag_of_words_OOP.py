
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
#for bag of words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

class KaggleBagOfPopcorn():
    def __init__(self):
        pass

    #load csv to pandas dataframe 
    def loadData(self,data_file):
        return pd.read_csv(data_path+"labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)


    def ReviewToWords(self):
        review_text = BeautifulSoup(raw_review).get_text()
        review_text = re.sub("[0-9\.]+", "NUM", review_text) 
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if w not in stops]
        return " ".join(meaningful_words)

    def ApplyBagOfWords(self,cleaned_data):
        vectorizer = CountVectorizer(analyzer="word", tokenizer = None, preprocessor= None, stop_words = None, max_features = 1000)
        train_data_features = vectorizer.fit_transform(cleaned_data)
        train_data_features = train_data_features.toarray()
        return train_data_features


    def ApplyRandomForestClassifier(self,train, train_data_features, test_data_features):
        forest = RandomForestClassifier(n_estimators = 100)
        forest = forest.fit(train_data_features, train["sentiment"])
        result = forest.predict(test_data_features)
        return result

    def PrepareSubmission(self,test, result):
        output = pd.DataFrame(data={"id":test["id"], "sentiment": result}) 
        output.to_csv("Bag_of_words_model.csv", index=False, quoting=3)


    def main():
        #load train set 
        data_path="C:\\Users\\MedAli\\Desktop\\kaggle_competitions\\Bag of words meets bags of popcorn\\data\\"
        train = self.loadData(data_path)

        #load test set        
        test= self.loadData(data_path+"testData.tsv")

        num_reviews = train["review"].size

        #cleaning train set 
        clean_train_reviews = []
        print "Cleaning and parsing the training set movie reviews...\n"
        for i in xrange(0, num_reviews):
            if((i+1)%1000 == 0):
                print "review %d of %d\n" % (i+1, num_reviews)
            clean_train_reviews.append(self.ReviewToWords(train["review"][i]))

        print "creating the bag of words \n"
        train_data_features = self.ApplyBagOfWords(clean_train_reviews)

        print "dimension of train_data_features:", train_data_features.shape

        #vocab = vectorizer.get_feature_names()

        #cleaning testset 
        num_reviews = len(test["review"])
        clean_test_reviews = []

        print "clearning and parsing the test set movie reviews..\n"
        for i in xrange(0, num_reviews):
            if((i+1)%1000 == 0):
                print "review %d of %d\n" % (i+1, num_reviews)
            clean_review = self.ReviewToWords(test["review"][i])
            clean_test_reviews.append(clean_review)

        #applying bag of words on the test data
        test_data_features = self.ApplyBagOfWords(clean_test_reviews)

        #applying random forest 
        result = self.ApplyRandomForestClassifier(train, train_data_features, test_data_features)

        #prepare Csv file for submission
        self.PrepareSubmission(test, result)

        return 


if __name__ == '__main__':
    bagOfpopcorn = KaggleBagOfPopcorn()
    bagOfpopcorn.main()

