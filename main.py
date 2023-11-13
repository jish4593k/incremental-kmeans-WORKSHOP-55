import random
import math
import matplotlib.pyplot as plt

class DataPoint:
    def __init__(self, a, b, c, d):
        self.features = [a, b, c, d]
        self.cluster_number = None

class Centroid:
    def __init__(self, a=0, b=0, c=0, d=0):
        self.features = [a, b, c, d]

class KMeans:
    def __init__(self, k=3):
        self.k = k
        self.centroids = [Centroid() for _ in range(k)]
        self.clusters = [[] for _ in range(k)]
        self.data_points = []

    def load_data(self, filename="iris.txt"):
        with open(filename) as file:
            content = file.readlines()
            for line in content:
                features = list(map(float, line.split()[:4]))
                self.data_points.append(DataPoint(*features))

    def initialize_centroids(self):
        for i in range(self.k):
            self.centroids[i] = Centroid(random.uniform(4, 8), random.uniform(1.5, 4.5), random.uniform(0.5, 7), random.uniform(0, 3))

    def get_distance(self, datapoint, centroid):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(datapoint.features, centroid.features)))

    def assign_to_clusters(self):
        for datapoint in self.data_points:
            distances = [self.get_distance(datapoint, centroid) for centroid in self.centroids]
            min_index = distances.index(min(distances))
            datapoint.cluster_number = min_index
            self.clusters[min_index].append(datapoint)

    def update_centroids(self):
        for i in range(self.k):
            if self.clusters[i]:
                for j in range(4):
                    self.centroids[i].features[j] = sum(datapoint.features[j] for datapoint in self.clusters[i]) / len(self.clusters[i])

    def kmeans(self, max_iterations=100):
        self.initialize_centroids()
        for _ in range(max_iterations):
            self.assign_to_clusters()
            self.update_centroids()

    def show_results(self):
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i + 1}: {len(cluster)} data points")

    def plot_clusters(self):
        for i, cluster in enumerate(self.clusters):
            plt.scatter([datapoint.features[0] for datapoint in cluster],
                        [datapoint.features[1] for datapoint in cluster],
                        label=f'Cluster {i + 1}')
        plt.scatter([centroid.features[0] for centroid in self.centroids],
                    [centroid.features[1] for centroid in self.centroids],
                    marker='X', color='black', label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-Means Clustering')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    kmeans = KMeans(k=3)
    kmeans.load_data()
    kmeans.kmeans()
    kmeans.show_results()
    kmeans.plot_clusters()
