import cv2
import numpy as np
import PIL
import math
from numpy import pi, exp, sqrt, linalg
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
from scipy import ndimage
from random import randint

class PA3:
    def __init__(self):
        # Nicolas Lopez Bravo
        print('Lucas Kanade')
        im1 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\basketball1.png'))
        im2 = np.asarray(PIL.Image.open('C:\\Users\\nicol\\Desktop\\basketball2.png'))
        
        # optional image path overriding
        # im1 = input('please paste path to basketball1')
        # im2 = input('please paste path to basketball2')
        LucasKanade(im1,im2)
        
        im = 'C:\\Users\\nicol\\Desktop\\GMS.jpg'
        #im = input('please paste path to GMS')
        print('Gaussian MeanShift')
        GMS(im)
        print('done')
        
class GMS:

    def __init__(self,path):
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        
        self.Hr = 90
        self.Hs = 90
        self.Iter = 100
        self.opImg = np.zeros(img.shape,np.uint8)
        
        # Mean Shift
        clusters = 0
        vectors = self.createMatrix(img)
        
        while(len(vectors) > 0):
            # Generate a random index for seed (from feature matrix)
            randomIndex = randint(0,len(vectors)-1)
            seed = vectors[randomIndex]
            
            # Cache the seed as our initial mean
            initialMean = seed
            
            neighbors = self.getNeighbors(seed,vectors)
            
            # If we get only 1 neighbor, we can mark it 
            if(len(neighbors) == 1):
                vectors=self.markPixels([randomIndex],initialMean,vectors,clusters)
                clusters+=1
                continue
            # If we have multiple pixels, calculate the mean of all
            mean = self.calculateMean(neighbors,vectors)
            
            # Calculate mean shift based on the initial mean
            meanShift = abs(mean-initialMean)
            
            if(np.mean(meanShift)<self.Iter):
                vectors = self.markPixels(neighbors,mean,vectors,clusters)
                clusters+=1
        
        cv2.imshow('Original Image',img)
        cv2.imshow('Mean Shift',self.opImg)
        
        
    def getNeighbors(self,seed,matrix,mode=1):
        # searches feature matrix to get neighbors of a pixel
        neighbors = []
        for i in range(0,len(matrix)):
            pixel = matrix[i]
            r = math.sqrt(sum((pixel[:3]-seed[:3])**2))
            s = math.sqrt(sum((pixel[3:5]-seed[3:5])**2))
            if(s < self.Hs and r < self.Hr ):
                neighbors.append(i)
        return neighbors
    
    def markPixels(self,neighbors,mean,matrix,cluster):
        # Deletes the pixel from the matrix and marks it in the output
        for i in neighbors:
            pixel = matrix[i]
            x=pixel[3]
            y=pixel[4]
            self.opImg[x][y] = np.array(mean[:3],np.uint8)
        return np.delete(matrix,neighbors,axis=0)
    
    def calculateMean(self,neighbors,matrix):
        neighbors = matrix[neighbors]
        gaussian = signal.gaussian(len(neighbors[:,:1]), std=7)
        denominator = sum(gaussian)
        
        r=neighbors[:,:1]
        g=neighbors[:,1:2]
        b=neighbors[:,2:3]
        x=neighbors[:,3:4]
        y=neighbors[:,4:5]
        
        sumr = 0
        sumg = 0
        sumb = 0
        sumx = 0
        sumy = 0
        
        for i in range(gaussian.shape[0]):
            sumr += r[i]*gaussian[i]
            sumg += g[i]*gaussian[i]
            sumb += b[i]*gaussian[i]
            sumx += x[i]*gaussian[i]
            sumy += y[i]*gaussian[i]
            
        mean = np.array([sumr[0],sumg[0],sumb[0],sumx[0],sumy[0]])/denominator

        return mean
    
    def createMatrix(self,img):
        # Creates a vector matrix of the image in the form of [r,g,b,x,y] for each pixel
        h,w,d = img.shape
        vectors = []
        for row in range(0,h):
            for col in range(0,w):
                r,g,b = img[row][col]
                vectors.append([r,g,b,row,col])
        vectors = np.array(vectors)
        return vectors        
                        
class LucasKanade:
    def __init__(self,image1,image2):        
        window = 7
        
        # Built in LK Pyramid
        self.builtIn(image1,image2,window)
        
        # Lucas Kanade using Good Features to Track
        feature_params = dict( maxCorners = 500,
                        qualityLevel = 0.1,
                        minDistance = 1,
                        blockSize = 1 )
        
        p0 = cv2.goodFeaturesToTrack(image1, mask = None, **feature_params)
        step = 2
        scale = .05
        name = 'LK Good Features to track'
        self.lkGFTT(image1,image2,window,p0,step,scale,name)
        
        # Lucas Kanade for every Pixel
        self.lk(image1,image2,window)
        
        # Lucas Kanade with Pyramid
        self.pyramidLK(image1,image2,window)
    
    def pyramidLK(self,im1,im2,win):
        feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
        
        second = cv2.pyrDown(im1)
        third = cv2.pyrDown(second)
        fourth = cv2.pyrDown(third)
        
        p0 = cv2.goodFeaturesToTrack(im1,mask = None, **feature_params)
        p1 = cv2.goodFeaturesToTrack(second,mask = None, **feature_params)
        p2 = cv2.goodFeaturesToTrack(third,mask = None, **feature_params)
        p3 = cv2.goodFeaturesToTrack(fourth,mask = None, **feature_params)
        
        p = np.concatenate((p0, p1), axis=0)
        
        # Using the second layer
        for item in p1:
            x, y = item[0]  
            temp = np.array([[[x+1,y+1]]])
            p = np.concatenate((p, temp), axis=0)
        
        # Using the third layer
        temp2 = p2.copy()
        p = np.concatenate((p,p2),axis=0)
        
        for item in p2:
            x, y = item[0]  
            temp = np.array([[[x+1,y+3]]])
            p = np.concatenate((p, temp), axis=0)
        
        for item in temp2:
            x, y = item[0]  
            temp = np.array([[[x+3,y+3]]])
            p = np.concatenate((p, temp), axis=0)
        
        # Using the fourth layer
        temp3 = p3.copy()
        p = np.concatenate((p,p3),axis=0)
        
        for item in p3:
            x, y = item[0]  
            temp = np.array([[[x+3,y+3]]])
            p = np.concatenate((p, temp), axis=0)
        
        for item in temp3:
            x, y = item[0]  
            temp = np.array([[[x+3,y+3]],[[x*9,y*9]]])
            p = np.concatenate((p, temp), axis=0)
        
        
        step = 2
        scale = .05
        name = 'Pyramid LK with Good Features to Track'
        self.lkGFTT(im1,im2,win,p,step,scale,name)
        
        
    def builtIn(self,im1,im2,win):
        imo = im1.copy()
        imi = im2.copy()
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (win,win),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        # Take first frame and find corners in it
        
        p0 = cv2.goodFeaturesToTrack(imo, mask = None, **feature_params)
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(imo)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(imo, imi, p0, None, **lk_params)
        
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(imo,(a,b),5,color[i].tolist(),-1)
        frame = cv2.add(frame,mask)
        cv2.imshow('Lucas Kanade BuiltIn',frame)
    
    def lkGFTT(self,im1,im2,win,p0,step,scale,name):
        
        im1 = im1.copy()
        im2 = im2.copy()
        
        # masks to get fx fy and ft       
        kx = np.array([[-1,1],[-1,1]])
        ky = np.array([[-1,-1],[1,1]])
        kt = np.array([[1,1],[1,1]])
        
        # if we want 3 by 3 window the span of the window would be 1, so we must divide
        win = math.floor(win/2)
        
        # normalize so we can use the kernels
        im1 = im1/255
        im2 = im2/255
        
        # create partial derivatives
        fx = signal.convolve2d(im1,kx,boundary='symm',mode='same')
        fy = signal.convolve2d(im1,ky,boundary='symm',mode='same')
        ft = signal.convolve2d(im2,kt,boundary='symm',mode='same') 
        ft = ft + signal.convolve2d(im1,-kt,boundary='symm',mode='same')
        
        u = np.zeros(im1.shape)
        v = np.zeros(im1.shape)
        
        for item in p0:
            j, i = item[0]
            i = int(i)
            j = int(j)
            
            Ix = fx[i-win:i+win+1, j-win:j+win+1].flatten()
            Iy = fy[i-win:i+win+1, j-win:j+win+1].flatten()
            It = ft[i-win:i+win+1, j-win:j+win+1].flatten()
            
            fxt = Ix*It
            fyt = Iy*It
            fy2 = Iy*Iy
            fx2 = Ix*Ix
            fxy = Ix*Iy
            fxy2 = fxy*fxy
            det = fx2*fy2 - fxy2
            
            if(len(Ix) > 0 and det[0] != 0):
                u[i,j] = ((-fy2*fxt + fxy*fyt)/det[0])[0]
                v[i,j] = ((fxt*fxy - fx2*fyt)/det[0])[0]
        
        # Create grid for display
        x = np.arange(0, im1.shape[1], 1)
        y = np.arange(0, im2.shape[0], 1)
        x, y = np.meshgrid(x, y)
        
        mag, angle = cv2.cartToPolar(u,v)
        # Normalize for appearance
        u = u/mag
        v = v/mag
        
        # Display
        plt.figure()
        plt.imshow(im2, cmap='gray', interpolation='bicubic')
        plt.title(name)
        plt.quiver(x[::step, ::step], y[::step, ::step],
                u[::step, ::step], v[::step, ::step],units ='dots',scale=scale,
                color='r', pivot='middle')
        plt.show()
        
    def lk(self,im1,im2,win): 
        # masks to get fx fy and ft       
        kx = np.array([[-1,1],[-1,1]])
        ky = np.array([[-1,-1],[1,1]])
        kt = np.array([[1,1],[1,1]])
        
        # if we want 3 by 3 window the span of the window would be 1, so we must divide
        win = math.floor(win/2)
        
        # normalize so we can use the kernels
        im1 = im1/255
        im2 = im2/255
        
        # create partial derivatives
        fx = signal.convolve2d(im1,kx,boundary='symm',mode='same')
        fy = signal.convolve2d(im1,ky,boundary='symm',mode='same')
        ft = signal.convolve2d(im2,kt,boundary='symm',mode='same') 
        ft = ft + signal.convolve2d(im1,-kt,boundary='symm',mode='same')
        
        u = np.zeros(im1.shape)
        v = np.zeros(im1.shape)
        
        for i in range(win,im1.shape[0]-win):
            for j in range(win,im1.shape[1]-win):
                
                Ix = fx[i-win:i+win+1, j-win:j+win+1].flatten()
                Iy = fy[i-win:i+win+1, j-win:j+win+1].flatten()
                It = ft[i-win:i+win+1, j-win:j+win+1].flatten()
                
                fxt = Ix*It
                fyt = Iy*It
                fy2 = Iy*Iy
                fx2 = Ix*Ix
                fxy = Ix*Iy
                fxy2 = fxy*fxy
                det = fx2*fy2 - fxy2
                
                if(det[0] != 0):
                    u[i,j] = ((-fy2*fxt + fxy*fyt)/det[0])[0]
                    v[i,j] = ((fxt*fxy - fx2*fyt)/det[0])[0]
            
        # Create grid for display
        x = np.arange(0, im1.shape[1], 1)
        y = np.arange(0, im2.shape[0], 1)
        x, y = np.meshgrid(x, y)
        
        # Display
        step = 3
        plt.figure()
        plt.imshow(im2, cmap='gray', interpolation='bicubic')
        plt.title('LK for every pixel')
        plt.quiver(x[::step, ::step], y[::step, ::step],
                u[::step, ::step], v[::step, ::step],
                color='r', pivot='middle')
        plt.show()
    
program3 = PA3()