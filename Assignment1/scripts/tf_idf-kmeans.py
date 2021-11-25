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
    print("Data loaded, Finding Clusters ...")
    k = [4, 8, 12]
    losses = []
    for num_clusters in k:
        kmeans = KMeans(x_train, y_train, num_clusters=num_clusters,
                        seed='cluster', tol=1e-7, max_iter=100)
        kmeans.fit(verbose=False)
        print("Clusters found, printing results ...")
        losses.append(kmeans.calc_loss())
        clusters = [[] for i in range(num_clusters)]
        for i, title in enumerate(titles):
            index = kmeans.cluster_labels[i]
            clusters[index].append(title)
        print("Clusters:")
        for i, cluster in enumerate(clusters):
            print("Cluster {}: {}".format(i, cluster))


if __name__ == '__main__':
    main()
