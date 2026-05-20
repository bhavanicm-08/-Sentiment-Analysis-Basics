from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

reviews = [
    "This product is amazing",
    "I love this item",
    "Very good quality product",
    "Excellent and useful",
    "Best purchase ever",
    "I am very happy with this",

    "This product is bad",
    "I hate this item",
    "Very poor quality",
    "Worst product ever",
    "I am disappointed",
    "Not worth the money"
]

labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

model = MultinomialNB()
model.fit(X, labels)
test_reviews = ["I love this product", "This is the worst item"]
X_test = vectorizer.transform(test_reviews)
predictions = model.predict(X_test)

for review, sentiment in zip(test_reviews, predictions):
    print(review, "->", "Positive" if sentiment == 1 else "Negative")