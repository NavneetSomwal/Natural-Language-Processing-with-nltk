# -*- coding: utf-8 -*-
"""
Created on Mon May  25 18:03:25 2018

@author: Navneet
"""
#Inspecting the Movie Reviews Dataset first
import nltk
nltk.download("movie_reviews")
from nltk.corpus import movie_reviews
len(movie_reviews.fileids())

#In particular in the movie_reviews dataset we have 2000 text files,
#each of them is a review of a movie, and they are already split in a neg folder
#for the negative reviews and a pos folder for the positive reviews:
movie_reviews.fileids()[:5]
movie_reviews.fileids()[-5:]
negative_fileids = movie_reviews.fileids('neg')
positive_fileids = movie_reviews.fileids('pos')
len(negative_fileids), len(positive_fileids)

#We can inspect one of the reviews using the raw method of movie_reviews,
#each file is split into sentences, the curators of this dataset also removed
#from each review from any direct mention of the rating of the movie.
print(movie_reviews.raw(fileids=positive_fileids[0]))

#Tokenize Text in Words
romeo_text = """Why then, O brawling love! O loving hate!
O any thing, of nothing first create!
O heavy lightness, serious vanity,
Misshapen chaos of well-seeming forms,
Feather of lead, bright smoke, cold fire, sick health,
Still-waking sleep, that is not what it is!
This love feel I, that feel no love in this."""

romeo_text.split()

#`nltk` has a sophisticated word tokenizer trained on English named `punkt`,
# we first have to download its parameters: 
nltk.download("punkt")

#Then we can use the `word_tokenize` function to properly tokenize this text, compare to the whitespace splitting we used above:
romeo_words = nltk.word_tokenize(romeo_text)
#romeo_words
romeo_words.shape

#Good news is that the movie_reviews corpus already has direct access to tokenized
#text with the words method:
movie_reviews.words(fileids=positive_fileids[0])

#Build a bag-of-words model
{word:True for word in romeo_words}
type(_)
def build_bag_of_words_features(words):
    return {word:True for word in words}

build_bag_of_words_features(romeo_words)

#This is what we wanted, but we notice that also punctuation like "!" and words
#useless for classification purposes like "of" or "that" are also included
#.Those words are named "stopwords" and nltk has a convenient corpus we can download:
nltk.download("stopwords")
import string
string.punctuation

#Using the Python string.punctuation list and the English stopwords we can build
#better features by filtering out those words that would not help in the classification:
useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
#useless_words
#type(useless_words)

def build_bag_of_words_features_filtered(words):
    return {
        word:1 for word in words \
        if not word in useless_words}

build_bag_of_words_features_filtered(romeo_words)

#Plotting Frequencies of Words
all_words = movie_reviews.words()
len(all_words)/1e6
filtered_words = [word for word in movie_reviews.words() if not word in useless_words]
type(filtered_words)
len(filtered_words)/1e6

#The `collection` package of the standard library contains a `Counter` class that
#is handy for counting frequencies of words in our list:
from collections import Counter
word_counter = Counter(filtered_words)
most_common_words = word_counter.most_common()[:10]
most_common_words

%matplotlib inline
import matplotlib.pyplot as plt

sorted_word_counts = sorted(list(word_counter.values()), reverse=True)

plt.loglog(sorted_word_counts)
plt.ylabel("Freq")
plt.xlabel("Word Rank");

plt.hist(sorted_word_counts, bins=50);
plt.hist(sorted_word_counts, bins=50, log=True);

#Train a Classifier for Sentiment Analysis
negative_features = [
    (build_bag_of_words_features_filtered(movie_reviews.words(fileids=[f])), 'neg') \
    for f in negative_fileids
]
print(negative_features[3])
positive_features = [
    (build_bag_of_words_features_filtered(movie_reviews.words(fileids=[f])), 'pos') \
    for f in positive_fileids
]
print(positive_features[6])

from nltk.classify import NaiveBayesClassifier
split = 800
sentiment_classifier = NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])
nltk.classify.util.accuracy(sentiment_classifier, positive_features[:split]+negative_features[:split])*100

#accuracy check on test data
nltk.classify.util.accuracy(sentiment_classifier, positive_features[split:]+negative_features[split:])*100
sentiment_classifier.show_most_informative_features()