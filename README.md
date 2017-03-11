# Amazon-Review-Classification-using-KNN-Classifier
Implemented a k-Nearest Neighbor Classifier to predict the sentiment for 18506 baby product reviews provided in the test file. Positive sentiment is represented by a review rating of +1 and negative sentiment is represented by a review rating of -1.

Training data consists of 18506 reviews as well, provided in the file train.dat. Each row begins with the sentiment score followed by the text associated with that rating.

## The objective of this assignment are the following:

* Implement the Nearest Neighbor Classification algorithm
* Handle text data (reviews of Amazon baby products)
* Design and engineer features from text data.
* Choose the best model, i.e., parameters of a nearest neighbor classifier, features and similarity functions

## Preprocessing steps followed on the train and test datasets:

* Remove special characters and convert all words to lowercase
* Use filterLen function to filter out words that have less than 4 letters except for the word “bad”
* Implemented Inverse document frequency function to decrease the importance of popular words.
* Remove suffixes (and in some cases prefixes) in order to find the root word or stem of a given word using Potter2 stemmer.
* K-mer implementation with K=2 and K=3: Every document is passed through grouper function which groups 2 consecutive and 3 consecutive words and adds them to the original list of words (features)

## Results obtained:
* Cross validation was used to find the Best K value for KNN classification. 
* Best K value obtained for the last run was K = 19. Accuracy significantly increased from 71.85% to 79.94% when Stemmers, Inverse Document Frequency and K-mer steps were included.
* Final accuracy obtained on 100% test data = 79.89%


Markup : ![picture alt](https://github.com/nivedithabhandary/Amazon-Review-Classification-using-KNN-Classifier/blob/master/results.png)
