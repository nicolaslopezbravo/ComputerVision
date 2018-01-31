import cv2
import numpy as np
import PIL
import math
from numpy import pi, exp, sqrt, linalg
import matplotlib.pyplot as plt

class Program1(object):
    def __init__(self):
        print("Uncomment the question you wish to see, note, you will have to replace",
        "image locations - Nicolas Lopez Bravo")
        #Question1()
        #Question2()
        #Question3()
        #Question4()
        #Question5()
        #Question6()
        #Question7()
        #Question8()
        #Question9()

class Question9(object):
    def __init__(self):
        # Load up image        
        image1 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image1.png'))
        image2 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image2.png'))
        image4 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image4.png'))
        
        # Call function
        bin1 = self.maxEntropy(image1)
        bin2 = self.maxEntropy(image2)
        bin4 = self.maxEntropy(image4)
        
        #write images
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\binary1.png",bin1)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\binary2.png",bin2)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\binary4.png",bin4)
    
    def maxEntropy(self,image):
        # obtain histogram
        hist = np.histogram(image, bins=256, range=(0, 256))[0]
        
        # get threshold
        t = self.calculateT(hist)
        
        # return new images
        return self.binary(image,t)   
    
    def calculateT(self,data):
        # calculate Cumulative density function
        cdf = data.astype(np.float).cumsum()
    
        # find histogram's nonzero area
        temp = np.nonzero(data)[0]
        firstBin = temp[0]
        lastBin = temp[-1]
    
        # initialize search for maximum
        maxE, threshold = 0, 0
    
        for i in range(firstBin, lastBin + 1):
            # Background (dark)
            hRange = data[:i + 1]
            hRange = hRange[hRange != 0] / cdf[i]  # normalize within selected range & remove all 0 elements
            totalE = -np.sum(hRange * np.log(hRange))  # background entropy
    
            # Foreground/Object (bright)
            hRange = data[i + 1:]
            
            # normalize within selected range & remove all 0 elements
            hRange = hRange[hRange != 0] / (cdf[lastBin] - cdf[i])
            totalE -= np.sum(hRange * np.log(hRange))  # accumulate object entropy
    
            # find max
            if totalE > maxE:
                maxE, threshold = totalE, i
    
        return threshold
        
    def binary(self,image,t):
        result = np.zeros((image.shape[0], image.shape[1]))
        for x in range(image.shape[0]):     
                for y in range(image.shape[1]):
                    if image[x,y] >= t:
                        # assign a value of 1 (aka 256)
                        result[x,y] = 256
                    else:
                        result[x,y] = 0   
        return result        

class Question8(object):
    def __init__(self):
        # Load up image        
        image = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image4.png'))
        
        # Ask for T
        t = input("please enter a number between 0 and 256: ")
        
        # Call function
        binary = self.binary(image,int(t))
        
        #write image
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\binaryImage.png",binary)
    
    def binary(self,image,t):
        result = np.zeros((image.shape[0], image.shape[1]))
        for x in range(image.shape[0]):     
                for y in range(image.shape[1]):
                    if image[x,y] >= t:
                        # assign a value of 1 (aka 256)
                        result[x,y] = 256
                    else:
                        result[x,y] = 0   
        return result

class Question7(object):
    def __init__(self):
        # Load up image        
        image = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image4.png'))
        # Create the histogram
        self.histogram(image,64)
        self.histogram(image,128)
        self.histogram(image,256)
    
    def histogram(self,image,size):
        bins = [None]*size
        for x in range(size):
            bins[x] = 0
        
        for x in range(image.shape[0]):     
                for y in range(image.shape[1]):
                    bins[image[x,y]%size] += 1
        plt.bar(np.arange(size),bins)
        plt.show()
        
class Question6(object):
    def __init__(self):
        # Load up image        
        image1 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image1.png'))
        image2 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image2.png'))
                 
        #filter image1 with a gaussian filterwith sigma=3 
        dst3 = self.fastgaussianfilter(image1,3)
        dst5 = self.fastgaussianfilter(image1,5)
        dst10 = self.fastgaussianfilter(image1,10)
        
        #write image1
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\fastgaussianfiltered1-sigma3.png",dst3)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\fastgaussianfiltered1-sigma5.png",dst5)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\fastgaussianfiltered1-sigma10.png",dst10)
        
        #filter image2     
        dst3 = self.fastgaussianfilter(image2,3)
        dst5 = self.fastgaussianfilter(image2,5)
        dst10 = self.fastgaussianfilter(image2,10)
        
        #write image2
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\fastgaussianfiltered2-sigma3.png",dst3)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\fastgaussianfiltered2-sigma5.png",dst5)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\fastgaussianfiltered2-sigma10.png",dst10)
        
    def fastgaussianfilter(self,image,sigma):
        #create an output of the same size of image
        output = np.zeros_like(image)
        boxSize = sigma
        
        # we must pad image to account for edges
        image_padded = np.zeros((image.shape[0] + boxSize-1, image.shape[0] + boxSize-1))
        image_padded[1:-(boxSize-2), 1:-(boxSize-2)] = image
        
        #generate the kernel
        kernel = self.kernel(sigma)
        
        # Loop in first direction
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                # we first flatten the array since convolution allows it
                flatI = image_padded[y:y+boxSize,x:x+boxSize].ravel()
                # then we store it to our ouput matrix
                output[y,x]=np.convolve(kernel,flatI,'valid')
        
        # Loop in the other direction
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # we first flatten the array since convolution allows it
                flatI = image_padded[y:y+boxSize,x:x+boxSize].ravel()
                # then we store it to our ouput matrix
                output[y,x]=np.convolve(kernel,flatI,'valid')
        
        return output
        
    def kernel(shape=(3,3),sigma=0.5):
    #1D gaussian mask 
        #  generate a gaussian kernel with mean=0 and sigma = s
        s = sigma        
        probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(s*s)] 
        return probs

class Question5(object):
    def __init__(self):
        # Load up image        
        image1 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image1.png'))
        image2 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image2.png'))
                       
        #filter image1 and image2 using sobel
        h = [[-1,-2,-1],[0,0,0],[1,2,1]]
        v = [[-1,0,1],[-2,0,2],[-1,0,1]]
        
        dst1 = self.sobelfilter(image1,h)
        dst2 = self.sobelfilter(image2,h)
        
        #write image1 and 2
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\sobelfiltered1-x.png",dst1)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\sobelfiltered2-x.png",dst2)
        
        dst1 = self.sobelfilter(image1,v)
        dst2 = self.sobelfilter(image2,v)
        
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\sobelfiltered1-y.png",dst1)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\sobelfiltered2-y.png",dst2)
    
    def sobelfilter(self,image,kernel):
        #create an output of the same size of image
        boxSize = 3
        output = np.zeros_like(image) 
        kernel = np.array(kernel)
        flatk = kernel.ravel()
        # we must pad image to account for edges 
        image_padded = np.zeros((image.shape[0] + boxSize-1, image.shape[0] + boxSize-1))
        image_padded[1:-(boxSize-2), 1:-(boxSize-2)] = image
        
        # Loop over every pixel of the image
        for x in range(image.shape[0]):     
            for y in range(image.shape[1]):
                # we first flatten the array since convolution allows it
                flatI = image_padded[y:y+boxSize,x:x+boxSize].ravel()
                # then we store it to our ouput matrix
                output[y,x]=np.convolve(flatk,flatI,'valid')
        return output
        
        

class Question4(object):  
    def __init__(self):
        # Load up image        
        image = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image4.png'))
        
        #filter image1 with their different derivatives
        self.forward(image)
        self.backward(image)
        self.central(image)
        
    def magnitude(self,fx,fy):
        f = linalg.norm(fx)*linalg.norm(fy)
        return f
        
    def forward(self,array):
        mask = [1,-1]
        fX = array.copy()
        fY = array.copy()
        
        for row in range(array.shape[0]): #rows
            for x in range(1,array.shape[1]): #columns
                fX[row,x] = array[row,x-1]*mask[0] + array[row,x]*mask[1]
            
        for col in range(array.shape[1]): #columns
            for x in range(1,array.shape[0]): #rows
                fY[x,col] = array[x-1,col]*mask[0] + array[x,col]*mask[1]
        
        #write images
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gradient-forward-X.png",fX)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gradient-forward-Y.png",fY)
        print(self.magnitude(fX,fY))
                
    def backward(self,array):
        mask = [-1,1]
        fX = array.copy()
        fY = array.copy()
        
        for row in range(array.shape[0]): #rows
            for x in range(1,array.shape[1]): #columns
                fX[row,x] = array[row,x-1]*mask[0] + array[row,x]*mask[1]
            
        for col in range(array.shape[1]): #columns
            for x in range(1,array.shape[0]): #rows
                fY[x,col] = array[x-1,col]*mask[0] + array[x,col]*mask[1]
        
        #write images
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gradient-backward-X.png",fX)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gradient-backward-Y.png",fY)
        print(self.magnitude(fX,fY))
        
        
    def central(self,array):
        mask = [-1,0,1]
        fX = array.copy()
        fY = array.copy()
        
        for row in range(array.shape[0]): #rows
            for x in range(1,array.shape[1]-1): #columns
                fX[row,x] = array[row,x-1]*mask[0] + array[row,x]*mask[1] + array[row,x+1]*mask[2]
            
        for col in range(array.shape[1]): #columns
            for x in range(1,array.shape[0]-1): #rows
                fY[x,col] = array[x-1,col]*mask[0] + array[x,col]*mask[1] + array[x+1,col]*mask[2]
        
        #write images
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gradient-central-X.png",fX)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gradient-central-Y.png",fY)
        
        print(self.magnitude(fX,fY))
        
class Question3(object):
    def __init__(self):
        # Load up image        
        image1 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image1.png'))
        image2 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image2.png'))
                       
        #filter image1 with a gaussian filterwith sigma=3 
        dst3 = self.gaussianfilter(image1,3)
        dst5 = self.gaussianfilter(image1,5)
        dst10 = self.gaussianfilter(image1,10)
        
        #write image1
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gaussianfiltered1-sigma3.png",dst3)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gaussianfiltered1-sigma5.png",dst5)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gaussianfiltered1-sigma10.png",dst10)
        
        #filter image2     
        dst3 = self.gaussianfilter(image2,3)
        dst5 = self.gaussianfilter(image2,5)
        dst10 = self.gaussianfilter(image2,10)
        
        #write image2
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gaussianfiltered2-sigma3.png",dst3)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gaussianfiltered2-sigma5.png",dst5)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\gaussianfiltered2-sigma10.png",dst10)
        
    
    def gaussianfilter(self,image,sigma):
        #create an output of the same size of image
        output = np.zeros_like(image) 
        boxSize = sigma*3
        
        # we must pad image to account for edges 
        image_padded = np.zeros((image.shape[0] + boxSize-1, image.shape[0] + boxSize-1))
        image_padded[1:-(boxSize-2), 1:-(boxSize-2)] = image
        
        #generate the kernel
        kernel = self.kernel(sigma)
        flatk = kernel.ravel()
        
        # Loop over every pixel of the image
        for x in range(image.shape[0]):     
            for y in range(image.shape[1]):
                # we first flatten the array since convolution allows it
                flatI = image_padded[y:y+boxSize,x:x+boxSize].ravel()
                # then we store it to our ouput matrix
                output[y,x]=self.convolve(flatk,flatI)
        return output
    
        
    def kernel(shape=(3,3),sigma=0.5):
    #2D gaussian mask 
        #  generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
        s = sigma
        k = s*2
        probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
        kernel = np.outer(probs, probs)
        return kernel
    
    def convolve(self,k,i):
        result = 0
        for x in range(len(i)):
            result += k[x]*i[x]
        return result
        
class Question2(object):
    def __init__(self):
        # Load up image        
        image1 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image1.png'))
        image2 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image2.png'))
                       
        #filter image1     
        dst3 = self.medianfilter(image1,3)
        dst5 = self.medianfilter(image1,5)
        dst7 = self.medianfilter(image1,7)
        
        #write image1
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\medianfiltered1-3x3.png",dst3)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\medianfiltered1-5x5.png",dst5)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\medianfiltered1-7x7.png",dst7)
        
        #filter image2     
        dst3 = self.medianfilter(image2,3)
        dst5 = self.medianfilter(image2,5)
        dst7 = self.medianfilter(image2,7)
        
        #write image2
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\medianfiltered2-3x3.png",dst3)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\medianfiltered2-5x5.png",dst5)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\medianfiltered2-7x7.png",dst7)
    
    def medianfilter(self,image,boxSize):
        #create an output of the same size of image
        output = np.zeros_like(image) 
        # we must pad image to account for edges 
        image_padded = np.zeros((image.shape[0] + boxSize-1, image.shape[0] + boxSize-1))
        image_padded[1:-(boxSize-2), 1:-(boxSize-2)] = image
        
        # Loop over every pixel of the image
        for x in range(image.shape[0]):     
            for y in range(image.shape[1]):
                # we first flatten the array
                flatI = image_padded[y:y+boxSize,x:x+boxSize].ravel()
                # then we store it to our ouput matrix
                output[y,x]=self.median(flatI,boxSize)
        return output
    
    def median(self,array,boxSize):
        array.sort()
        return array[math.floor(len(array)-(boxSize/2))]
    
class Question1(object):
    def __init__(self):
        # Load up image        
        image1 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image1.png'))
        image2 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\pa1\\image2.png'))
        
        #Create the kernels, I am using a simple blur filter
        kernel3 = np.ones((3,3),np.float32)/9
        kernel5 = np.ones((5,5),np.float32)/25
                       
        #filter image1     
        dst3 = self.boxfilter(image1,kernel3,3)
        dst5 = self.boxfilter(image1,kernel5,5)
        
        #write image1
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\Boxfiltered1-3x3.png",dst3)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\Boxfiltered1-5x5.png",dst5)
        
        #filter image2     
        dst3 = self.boxfilter(image2,kernel3,3)
        dst5 = self.boxfilter(image2,kernel5,5)
        
        #write image2
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\Boxfiltered2-3x3.png",dst3)
        cv2.imwrite("C:\\Users\\nicol\\Desktop\\pa1\\Boxfiltered2-5x5.png",dst5)
    
    def boxfilter(self,image,kernel,boxSize):
        #create an output of the same size of image
        output = np.zeros_like(image) 
        flatk = kernel.ravel()
        # we must pad image to account for edges 
        image_padded = np.zeros((image.shape[0] + boxSize-1, image.shape[0] + boxSize-1))
        image_padded[1:-(boxSize-2), 1:-(boxSize-2)] = image
        
        # Loop over every pixel of the image
        for x in range(image.shape[0]):     
            for y in range(image.shape[1]):
                # we first flatten the array since convolution allows it
                flatI = image_padded[y:y+boxSize,x:x+boxSize].ravel()
                
                # then we store it to our ouput matrix
                output[y,x]=np.convolve(flatk,flatI,'valid')
        return output
    
program1 = Program1()