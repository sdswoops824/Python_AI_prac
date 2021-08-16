from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# define the category map
category_map = {'talk.religion.misc':'Religion','rec.autos':'Autos',
               'rec.sport.hockey':'Hockey','sci.electronics':'Electronics','sci.space':'Space'}

# create the training set
training_data = fetch_20newsgroups(subset='train',
                                   categories=category_map.keys(), shuffle=True, random_state=5)

# build a count vectorizer and extract the term counts
vectorizer_count = CountVectorizer()
train_tc = vectorizer_count.fit_transform(training_data.data)
print("\nDimensions of training data:", train_tc.shape)

# create tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

# define test data
input_data = [
    'Discovery was a space shuttle',
    'Hinduism, Christianity, Sikhism are all religions',
    'We must have to drive safely',
    'Puck is a disk made of rubber',
    'Television, Microwave, Refrigerator all use electricity']

# train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(train_tfidf, training_data.target)

# transform the input data using count vectorizer
input_tc = vectorizer_count.transform(input_data)

# transform the vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)

# predict the output categories
predictions = classifier.predict(input_tfidf)

# generate the output
for sent, category in zip(input_data, predictions):
    print('\nInput Data:', sent, '\n Category:',
          category_map[training_data.target_names[category]])


