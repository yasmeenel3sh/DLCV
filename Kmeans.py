import numpy as np
import matplotlib.pyplot as plt
import math
import collections
import sys
from scipy.io import loadmat
import random
import synthetic
from sklearn.cluster import KMeans
import confusion

np.set_printoptions(threshold=sys.maxsize)

def kmeansX(Data,K,C):
    #Data array, k classes, C means for classes
    #the following step to know the dimensions of the data
    dimSize=len(Data.shape)
    #creating empty array to save cluster number per pixel
    pixelClusterNo=np.zeros((Data.shape[0],Data.shape[1]))
    currentClusterMeans=C
    newClusterMeans=np.zeros(C.shape)
    #used to not update in the first iteration only
    counter =0
    while not np.equal(currentClusterMeans,newClusterMeans).all():
        
        if counter !=0:
            currentClusterMeans=newClusterMeans
        newClusterMeans=np.zeros(C.shape)
        clusterSum=np.zeros(C.shape[0])
    #Assuming only one image is entered, the first dim is the no. of rows
        #clustering the pixels
        for i in range(0,Data.shape[0]):
            for j in range(0,Data.shape[1]):
                shortestPixelDistance=1000
                currentCluster= -1
                for m in range(0,K):
                    distance= np.linalg.norm((Data[i][j]-currentClusterMeans[m]))
                    #print(distance)

                    if distance <= shortestPixelDistance:
                        currentCluster=m
                        pixelClusterNo[i][j]=currentCluster
                        shortestPixelDistance=distance
                #updating clusterMeansTotal and sum
                newClusterMeans[currentCluster]+=Data[i][j]
                clusterSum[currentCluster]+=1
        #print(clusterSum)                
        #formulting the new cluster mean
        for x in range(0,newClusterMeans.shape[0]):
            if(clusterSum[x]!=0):
                newClusterMeans[x]=np.divide(newClusterMeans[x],clusterSum[x])
            else:
                newClusterMeans[x]=np.zeros(newClusterMeans[x].size)    
        counter+=1
        # print(newClusterMeans)
        # print(clusterSum)
       
    finalClusteredImage=np.zeros(Data.shape)

   
    for y in range(0,Data.shape[0]):
        for p in range(0,Data.shape[1]):
            finalClusteredImage[y][p]=newClusterMeans[(pixelClusterNo[y][p]).astype(int)]
    #the finalclassImage is the clustered image each class has its mean color
    #pixelclusterno is an image with each pixel representing its class number mostly used in the hyper spectral   
    return finalClusteredImage , pixelClusterNo , newClusterMeans


def choose_initial_cluster_centers(points, clusters, dim):
    points = np.reshape(points,((42*43),points.shape[2]))
    us = np.zeros((clusters,dim))         # centers of each cluster
    indicies = np.zeros(clusters)

    index = np.copy(np.random.choice(points.shape[0], 1)[0])         # choose first point randomly
    indicies[0] = np.copy(index)
    us[0] =  np.copy(points[index])

    for i in range(1,clusters):
        max_distance = 0
        for j in range(points.shape[0]):    # choose the center with the biggest distancce form last center
            if j not in indicies:           # check if the the point is already a center to cluster
                point = np.copy(points[j])
                if point not in us:
                    last_point = np.copy(us[i-1])
                    distance = np.linalg.norm(point-last_point)     # Eucledian distance

                    if distance > max_distance:
                        max_distance = distance
                        index = j
                        u = np.copy(point)
        
        
        indicies[i] = np.copy(index)
        us[i] = np.copy(u)

    return us

def map_classes(clustered_data, labeled_data, num_classes):
    indicies = []
    predict = clustered_data.copy()
    labels = labeled_data.copy()
    
    all_counts = []
    for i in range(num_classes):
        counts = []
        uniqueTruthvalues=np.unique(labeled_data)
        unique, counts = np.unique(clustered_data[np.where(labeled_data == uniqueTruthvalues[i])], return_counts=True)
        counts = dict(zip(unique, counts))
        for k in range(num_classes):
            if k not in counts.keys():
                counts[k] = 0
        keys = sorted(counts.keys())
        counts_list = []
        for key in keys:
            counts_list.append(counts[key])

        all_counts.append(counts_list)
    
    all_counts = np.array(all_counts)

    for i in range(num_classes):
        labels[labeled_data == uniqueTruthvalues[np.where(all_counts[0:,i:(i+1)] == np.amax(all_counts[0:,i:(i+1)]))[0][0]]] = i-(num_classes - 1)

    labels += 4

    return predict, labels

#genrates initial means randomly
def randomClassMeanGenrator(img,numberOfClasses,dim):
    #gets all unique values used in the image sorted
    uniqueValues=np.unique(img)
    clusterMeansInitial=np.zeros((numberOfClasses,dim))
    for i in range(0,numberOfClasses):
        for j in range(0,dim):
            clusterMeansInitial[i][j]=random.randint(1,uniqueValues[-1])
    return clusterMeansInitial
# Generate synthetic images
height = 512
width = 512
num_color_range = 256

line1_start = (width/2) + int(width*.10)
line1_range = int(width*.01)
line2_start = (width/2) + int(width*.30)
line2_range = int(width*.05)

sigma = int(num_color_range / 6)
noise_probability = 0.9

# img1 = synthetic.syn_quarter_image(height,width,num_color_range,0,0)
# img5 = synthetic.syn_image_lines(height,width,num_color_range,line1_start,line1_range,line2_start,line2_range,0,0)
# img1_testGrayImage_hi = synthetic.syn_quarter_image(height,width,num_color_range,sigma,noise_probability)
# img5_testGrayImage_hi = synthetic.syn_image_lines(height,width,num_color_range,line1_start,line1_range,line2_start,line2_range,sigma,noise_probability)

# plt.imsave("img1.png", img1, cmap='gray', vmin=0, vmax=255)
# plt.imsave("img5.png", img5, cmap='gray', vmin=0, vmax=255)
# plt.imsave("img1_testGrayImage_hi.png", img1_testGrayImage_hi, cmap='gray', vmin=0, vmax=255)
# plt.imsave("img5_testGrayImage_hi.png", img5_testGrayImage_hi, cmap='gray', vmin=0, vmax=255)
######

#for rgb
# img = plt.imread("res/star.jpg", format="jpg")
# print(img.shape)
# arr=randomClassMeanGenrator(img,4,3)
# kmImage,clusterNoImage,newClusterMeans=kmeans(img,4,arr)
# plt.imshow(kmImage.astype(np.uint8))
# plt.savefig("outputrgb.png")

#for gray
# img = plt.imread("res/starg.jpg", format="jpg")
# print(img.shape)
# arrg=randomClassMeanGenrator(img,2,1)
# kmImage,clusterNoImage,newClusterMeans=kmeans(img,2,arrg)
# plt.imshow(kmImage.astype(np.uint8),cmap="gray")
# plt.savefig("outputgray.png")


#for spectral
img= loadmat('res/SalinasA_Q3.mat')
groundtruth=loadmat('res/SalinasA_GT3.mat')
imgdata = img['Q3']
#print(imgdata)


imgdata = (imgdata / np.max(imgdata)) * 255
imgdatak = np.reshape(imgdata, (imgdata.shape[0] * imgdata.shape[1], imgdata.shape[2]))

kmeans = KMeans(n_clusters=5)
kmeans.fit(imgdatak)
cl=kmeans.cluster_centers_

groundtruthdata=groundtruth['Q3_GT']
#[  0 182 219 237 255] it contains 5 unique classes not 7
# print(groundtruthdata[5])
arrspec =np.zeros((5,204))
arrspec[0]=cl[0]
arrspec[1]=cl[1]
arrspec[2]=cl[2]
arrspec[3]=cl[3]
arrspec[4]=cl[4]
# arrspec=randomClassMeanGenrator(imgdata,5,imgdata.shape[2])
arrspec=choose_initial_cluster_centers(imgdata,5,imgdata.shape[2])
# print(arrspec)

# for i in range(arrspec.shape[0]):
#     for j in range(i+1,arrspec.shape[0]):
#         print(str(i) + " " + str(j))
#         print((arrspec[j] == arrspec[i]).all())




kmImage,clusterNoImage,newClusterMeans= kmeansX(imgdata,5,arrspec)

predict, labels = map_classes(clusterNoImage, groundtruthdata, 5)

# reIndexed=np.zeros(clusterNoImage.shape)
# norms=np.zeros(5)
# for i in range(5):
#     norms[i]=np.linalg.norm(newClusterMeans[i]) 
# normssorted=np.unique(norms)
# uniqueTruthvalues=np.unique(groundtruthdata)    
# # print(normssorted)
# # print(uniqueTruthvalues)
# for j in range(0,clusterNoImage.shape[0]):
#     for k in range(0,clusterNoImage.shape[1]):
#         classNumber=clusterNoImage[j][k]
#         classMeanNorm=norms[(classNumber).astype(int)]
#         toBeAddedClassNo=uniqueTruthvalues[np.where(normssorted==classMeanNorm)]
#         reIndexed[j][k]=toBeAddedClassNo
# print(reIndexed[5])
# print(groundtruthdata[0])
# print(clusterNoImage[0])
# print(groundtruthdata[0])
# print(kmImage.shape)
# print(imgdata.shape[2])

plt.figure()
plt.imshow(clusterNoImage.astype(np.uint8))
plt.savefig("number.png")

# plt.imshow(reIndexed.astype(np.uint8))
# plt.savefig("number4.png")

# plt.figure()
# plt.imshow(groundtruthdata.astype(np.uint8))
# plt.savefig("number2.png")


# from sklearn.cluster import KMeans

# plt.scatter(imgdata[:,0],imgdata[:,1], label='True Position')

# print(img['Q3'])

# kmeans = KMeans(n_clusters=5)
# kmeans.fit(imgdata)
# print(kmeans.cluster_centers_)
# print(kmeans.labels_)

# plt.figure()
# plt.imshow(np.reshape(kmeans.labels_,(42,43)).astype(np.uint8))
# plt.savefig("number3.png")

cm, acc = confusion.confusion_matrix_compute(predict,labels)
print(acc)
confusion.confusion_matrix_save(cm)
