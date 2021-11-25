import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist
from kmeans import KMeans
import argparse


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize training and test data
    x_train = x_train / 255
    x_test = x_test / 255
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    digits = []
    targets = []
    for i in range(10):
        images = x_train[y_train == i]
        digits.append(images[np.random.choice(
            len(images), 100, replace=False)])
        targets.append(np.full((100,), i))

    x_train = np.vstack(digits)
    y_train = np.hstack(targets)

    # shuffle the data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    test_indices = np.random.choice(x_test.shape[0], 50)
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]
    return (x_train, y_train), (x_test, y_test)


def plot_centroids(kmeans, centroids):
    centroid_images = np.copy(centroids.reshape(kmeans.k, 28, 28))
    centroid_images = centroid_images * 255

    centroid_labels = kmeans.get_centroid_labels()

    fig = plt.figure(figsize=(20, 20))
    nrows = 5
    ncols = kmeans.k // nrows + kmeans.k % nrows
    for i in range(kmeans.k):
        fig.add_subplot(nrows, ncols, i+1)
        plt.imshow(centroid_images[i], cmap="gray")
        plt.title(f"Label: {centroid_labels[i]}", fontsize=15)
        plt.axis("off")
    plt.show()


def main(args: argparse.Namespace):
    # load the mnist data
    (x_train, y_train), (x_test, y_test) = load_data()
    # create a kmeans instance
    kmeans = KMeans(x_train, y_train,
                    num_clusters=args.num_clusters,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    seed=args.seed)

    kmeans.fit(verbose=args.verbose, plot=True)  # train the model
    # predict the labels from input labels and centroids
    predictions = kmeans.predict(x_test)
    print(f"Accuracy: {np.mean(predictions == y_test)}")  # print the accuracy
    plot_centroids(kmeans, kmeans.centroids)  # plot the centroids


def plot_jclust(args):
    k = np.arange(start=5, stop=21, step=1, dtype=int)
    (x_train, y_train), (x_test, y_test) = load_data()
    # create a kmeans instance
    jclust = []
    for num_clusters in k:
        kmeans = KMeans(x_train, y_train,
                        num_clusters=num_clusters,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        seed=args.seed)
        kmeans.fit(verbose=args.verbose)  # train the model
        jclust.append(kmeans.calc_loss())

    plt.plot(k, jclust)
    plt.xlabel("Number of Clusters")
    plt.ylabel("J-Clustering Loss")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=str,
                        default="cluster", help="cluster or random")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    seed_everything(35)
    
    # main(args)
    plot_jclust(args)
