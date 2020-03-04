import numpy as np
import matplotlib.pyplot as plt
import math
import collections
import sys
np.set_printoptions(threshold=sys.maxsize)

def kmeans(Data,K,C):
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
                    if distance <= shortestPixelDistance:
                        currentCluster=m
                        pixelClusterNo[i][j]=currentCluster
                        shortestPixelDistance=distance
                #updating clusterMeansTotal and sum
                newClusterMeans[currentCluster]+=Data[i][j]
                clusterSum[currentCluster]+=1        
        #formulting the new cluster mean
        for x in range(0,newClusterMeans.shape[0]):
            newClusterMeans[x]=np.divide(newClusterMeans[x],clusterSum[x])
        print(newClusterMeans)    
        counter+=1

    finalClusteredImage=np.zeros(img.shape)
   
    for y in range(0,img.shape[0]):
        for p in range(0,img.shape[1]):
            finalClusteredImage[y][p]=newClusterMeans[(pixelClusterNo[y][p]).astype(int)]
       
    return finalClusteredImage  




img = plt.imread("res/starg.jpg", format="jpg")

#for rgb
# arr=np.zeros((2,3))
# arr[0]=np.array([255,0,0])
# arr[1]=np.array([0,255,0])
#plt.imshow(km.astype(np.uint8))


#for gray
arr=np.zeros(2)
arr[0]=20
arr[1]=140
km=kmeans(img,2,arr)

plt.imshow(km.astype(np.uint8),cmap="gray")


plt.savefig("output.png")
