from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import wikipedia
from kmeans import KMeans

titles = [
    'Linear algebra',
    'Data Science',
    'Artificial intelligence',
    'European Central Bank',
    'Financial technology',
    'International Monetary Fund',
    'Basketball',
    'Swimming',
    'Cricket'
]


def load_data():
    articles = [wikipedia.page(
        title, preload=True).content for title in titles]
    vectorizer = TfidfVectorizer(stop_words={'english'})
    x_train = vectorizer.fit_transform(articles).toarray()
    y_train = np.arange(len(titles))

    return (x_train, y_train), vectorizer


def main():
    (x_train, y_train), vectorizer = load_data()
    kmeans = KMeans(x_train, y_train, num_clusters=9,
                    seed='cluster', tol=1e-6, max_iter=100)
    kmeans.fit(verbose=False)
    x_query = vectorizer.transform(
        ['There was water in the swimming pool'])
    x_query = x_query.toarray()
    predicts = kmeans.predict(x_query)
    predict_titles = [titles[i] for i in predicts]
    print(predict_titles)


if __name__ == '__main__':
    main()
