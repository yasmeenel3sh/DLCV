import numpy as np
import matplotlib.pyplot as plt
import math
import collections
import sys
from scipy.io import loadmat
import random
import synthetic
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
    #the finalclassImage is the clustered image each class has its mean color
    #pixelclusterno is an image with each pixel representing its class number mostly used in the hyper spectral   
    return finalClusteredImage , pixelClusterNo 


#genrates initial means randomly
def randomClassMeanGenrator(img,numberOfClasses,dim):
    #gets all unique values used in the image sorted
    uniqueValues=np.unique(img)
    clusterMeansInitial=np.zeros((numberOfClasses,dim))
    for i in range(0,numberOfClasses):
        for j in range(0,dim):
            clusterMeansInitial[i][j]=random.randint(uniqueValues[0],uniqueValues[-1])
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

img1 = synthetic.syn_quarter_image(height,width,num_color_range,0,0)
img5 = synthetic.syn_image_lines(height,width,num_color_range,line1_start,line1_range,line2_start,line2_range,0,0)
img1_testGrayImage_hi = synthetic.syn_quarter_image(height,width,num_color_range,sigma,noise_probability)
img5_testGrayImage_hi = synthetic.syn_image_lines(height,width,num_color_range,line1_start,line1_range,line2_start,line2_range,sigma,noise_probability)

plt.imsave("img1.png", img1, cmap='gray', vmin=0, vmax=255)
plt.imsave("img5.png", img5, cmap='gray', vmin=0, vmax=255)
plt.imsave("img1_testGrayImage_hi.png", img1_testGrayImage_hi, cmap='gray', vmin=0, vmax=255)
plt.imsave("img5_testGrayImage_hi.png", img5_testGrayImage_hi, cmap='gray', vmin=0, vmax=255)
######

#for rgb
# img = plt.imread("res/star.jpg", format="jpg")
# arr=randomClassMeanGenrator(img,4,3)
# kmImage,clusterNoImage=kmeans(img,4,arr)
# plt.imshow(kmImage.astype(np.uint8))
# plt.savefig("outputrgb.png")

#for gray
#img = plt.imread("res/starg.jpg", format="jpg")
#arrg=randomClassMeanGenrator(img,2,1)
# kmImage,clusterNoImage=kmeans(img,2,arrg)
# plt.imshow(kmImage.astype(np.uint8),cmap="gray")
# plt.savefig("outputgray.png")

#for spectral
img= loadmat('res/SalinasA_Q3.mat')
groundtruth=loadmat('res/SalinasA_GT3.mat')
imgdata = img['Q3']
groundtruthdata=groundtruth['Q3_GT']
#[  0 182 219 237 255] it contains 5 unique classes not 7
#print(np.unique(groundtruthdata))
arrspec=randomClassMeanGenrator(imgdata,5,imgdata.shape[2])
kmImage,clusterNoImage=kmeans(imgdata,5,arrspec)
print(kmImage.shape)
#print(imgdata.shape[2])

