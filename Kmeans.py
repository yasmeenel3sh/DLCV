import numpy as np
import matplotlib.pyplot as plt
import math
import collections
import sys
from sklearn.metrics import davies_bouldin_score 
np.set_printoptions(threshold=sys.maxsize)


def generate_x(num_img, img_size, img_path):
    result = np.zeros(shape=(num_img, img_size))
    for i in range(0, num_img):
        img = plt.imread(img_path + str(i + 1) + ".jpg", format="jpg")
        result[i] = np.reshape(img, img.size)
    return result


def cluster_dispersion(x, mean):
    return np.sum(np.linalg.norm((x-mean), axis=1))/x.shape[0]


def cluster_similarity(cluster1, cluster2):
    x1, mean1 = cluster1
    x2, mean2 = cluster2
    return (cluster_dispersion(x1, mean1) + cluster_dispersion(x2, mean2)) / np.linalg.norm(mean1 - mean2)


def generate_labels(file_path):
    file = open(file_path, "r")
    labels_text = file.read()
    labels = labels_text.split("\n")
    labels = labels[:-1]
    labels = np.array(list(map(int, labels)))
    return labels


labels = generate_labels('Images/Training Labels.txt')
images = generate_x(2400, 784, 'Images/')
images = (images > 140).astype(int)
k = 10
min_db = math.inf
min_db_index = 0
result_membership = np.zeros(images.shape[0])
result_means = np.zeros(shape=(k, images.shape[1]))
np.random.seed(42)
for initialization in range(0, 30):
    # Get a random vector from the images
    candidate_image_index = np.random.randint(0, images.shape[0])
    candidate_image = images[candidate_image_index]
    # Get an array of the feature vectors not containing the U1
    comparison_images = np.copy(images)
    # Populate the rest of the cluster centroids
    means = np.zeros(shape=(k, images.shape[1]))
    means[0] = candidate_image
    #print("mean " + str(0) + " for run " + str(initialization + 1) + " is " + str(candidate_image_index))
    for i in range(1, k):
        
        # Erase the image that has been picked as a centroid
        comparison_images = np.delete(
            comparison_images, candidate_image_index, axis=0)
        # Get the euclidean distance between the rest of the images and the chosen centroid
        norms = np.linalg.norm((candidate_image-comparison_images), axis=1)
        # Get the farthest of those distances from your previous centroid
        # to represent the new centroid
        candidate_image_index = np.argmax(norms)
        candidate_image = comparison_images[candidate_image_index]
        means[i] = candidate_image
        #print("mean " + str(i) + " for run " + str(initialization + 1) + " is " + str(candidate_image_index))

    # Start by running K-means algorithm to update centroids till convergence
    new_means = np.zeros(means.shape)
    counter = 0
    while not np.equal(new_means, means).all():
        if counter != 0:
            means = np.copy(new_means)
        membership_norms = np.zeros(shape=(k, images.shape[0]))
        for i in range(0, means.shape[0]):
            membership_norms[i] = np.linalg.norm((means[i]-images), axis=1)
        membership = np.argmin(membership_norms, axis=0)
        for i in range(0, k):
            indices = membership == i
            new_means[i] = np.mean(images[indices], axis=0)
        counter += 1

    #print(counter)
    max_similarities = np.zeros(10)
    for i in range(k):
        for j in range(k):
            if(i != j): 
                current_similarity = cluster_similarity(
                    (images[membership == i], means[i]), (images[membership == j], means[j]))
                if(max_similarities[i] < current_similarity):
                    max_similarities[i] = current_similarity
    db = np.sum(max_similarities)/k
    
    #print("Sklearn: " + str(davies_bouldin_score(images, membership)) + "   DIY: " + str(db))
    if(db < min_db):
        min_db = db
        
        min_db_index = initialization
        result_means = np.copy(means)
        result_membership = np.copy(membership)

print(min_db)
print(min_db_index)
cluster_counts = np.zeros(shape=(k, k))
for i in range(0, k):
    cluster_labels = labels[result_membership == i]
    counts = np.zeros(k)
    for j in range(k):
        counts[j] = cluster_labels[cluster_labels == j].size
    cluster_counts[i] = counts
print(cluster_counts)
cluster_counts = np.max(cluster_counts, axis=0)
print(cluster_counts)
plt.bar(np.arange(0, k), cluster_counts, align="center")
plt.ylabel("Clustered Count")
plt.xlabel("Digit Class")
plt.title("Clustering counts for K-means")
plt.show()
plt.savefig("Counts.jpg")