import PIL
import numpy as np
from numpy import pi, exp, sqrt, zeros, arctan2, real
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import cv2
class Program2(object):
    
    # Nicolas Lopez Bravo
    # Robot Vision assignment 2 - Canny Edge Detection
    
    def __init__(self):        
        
        canny = input("Canny Edge detection (1) or image enhancing (2)? ")
        canny = int(canny)
        
        if(canny == 1):
            
            image = input("Please enter an image directory (full path): ")
            mult = input("Do you want to process at a range of sigmas? (y) : (n) ")
            
            if(mult == 'n'):
                
                sigma = input("Please enter the sigma you want to work with: ")
                print("Processing image", image, " at sigma = ", sigma)
                CannyEdgeDetection(image, int(sigma)) 
                print("Completed")
                
            elif(mult == 'y'):
                start = input("Please enter the starting Integer for sigma: ")
                end = input("Please enter the ending Integer for sigma: ")
                
                start = int(start)
                end = int(end)
                
                for sigma in range(start,end+1):
                    print("Processing image", image, " at sigma = ", sigma)
                    
                    # It will perform Canny Edge Detection at different sigmas
                    CannyEdgeDetection(image, sigma) #in case of manual sigma do int(sigma)
                print("Completed")
                
        elif(canny == 2):
            BonusQuestion()
            
class BonusQuestion(object):
    def __init__(self):
        image = input("Please enter an image directory (full path): ")
        image1 = np.asarray(PIL.Image.open(image))
        
        sigma = 3
        
        #Calculate Gaussian (smooth image)
        gauss = self.gaussian(image1,sigma)
        
        # Subtract Image - Gaussian
        sub = self.subtract(image1,gauss)
        
        # Create sharpened image with gaussian and alpha picture 4 => 1.08, 3 => 2, 2 => 1.1
        alpha = input("Please input alpha for Unmask Sharpening (pic 4-> 1.08, 3->2, 2->1.1: ")
        sharp = self.add(gauss,sub,float(alpha))
        
        #beta = input("Enter alpha fo")
        lapgauss = self.lapgaussian(image1)
        
        # Calculate Gradient
        grad = self.gradient(lapgauss)
        gradient = grad[0]
        orientation = grad[1] 
        
        # Apply non maximum suppression
        nms = self.maximum(gradient,orientation)
        
        # Apply Hysteresis
        canny = self.hysteresis(nms,sigma)
        
        # picture 4 => 8, 3 = > 8, 2 => 6
        alpha = input("Please input alpha for Laplacian of Gaussian Sharpening (pic 4,3 -> 8, 2->4: ")
        lapsharp = self.add(canny,image1,float(alpha))
        
        # Plot all
        plt.figure(1)
        plt.subplot(221)
        plt.title('Unsharp masking')
        plt.imshow(sharp, cmap='gray')
        plt.subplot(222)
        plt.title('Laplacian Gaussian Sharpening')
        plt.imshow(lapsharp, cmap='gray')
        plt.subplot(223)
        plt.title('normal')
        plt.imshow(image1, cmap='gray')
        plt.subplot(224)
        plt.title('gaussian')
        plt.imshow(gauss, cmap='gray')
        plt.show()
        
        
    def subtract(self,im,filtr):
        output = np.zeros_like(im)
        
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                output[i][j] = im[i][j] - filtr[i,j]

        return output
    
    def add(self,im,filtr,alpha):
        output = np.zeros_like(im)
        
        mmax = 1
        
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                output[i][j] = im[i][j] + (alpha * filtr[i][j])
                if output[i][j] > mmax: mmax = output[i][j]
        
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                output[i][j] *= (255/mmax)
        
                
        return output
        
    def gradient(self,array):
        # Sobel filtering
        h = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        v = np.array([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]])
        
        # Fourier transform of kernels
        kernel1 = zeros(array.shape)
        kernel1[:h.shape[0], :h.shape[1]] = h
        kernel1 = fft2(kernel1)
    
        kernel2 = zeros(array.shape)
        kernel2[:v.shape[0], :v.shape[1]] = v
        kernel2 = fft2(kernel2)
        
        # Apply kernels
        farray = fft2(array)
        Gx = real(ifft2(kernel1 * farray)).astype(float)
        Gy = real(ifft2(kernel2 * farray)).astype(float)
    
        G = sqrt(Gx**2 + Gy**2)
        Theta = arctan2(Gy, Gx) * 180 / pi
        
        cuadrant = np.zeros_like(array)
        
        # Save which quadrant it is on
        for x in range(cuadrant.shape[0]):
            for y in range(cuadrant.shape[1]):
                if (Gx[x,y] > 0):
                    if(Gy[x,y] > 0):
                        cuadrant[x,y] = 1
                    else:
                        cuadrant[x,y] = 3
                else:
                    if(Gy[x,y] > 0):
                        cuadrant[x,y] = 2
                    else:
                        cuadrant[x,y] = 4
        
        
        return G, Theta, cuadrant
    
    def hysteresis(self,array,sigma=1):
        # Do thresholding
        t  = np.zeros_like(array)        
        lo, hi = 100/sigma, 160/sigma 
        #lo, hi = 10/sigma,50/sigma #for picture 4 we neeed low thresholds
        edges = []
        
        # Set 255 (white) as a strong edge) and 0 as non edges
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] >= hi:
                    t[i][j] = 255
                    # Making a list of edges to traverse
                    edges.append((i, j))
                elif array[i][j] >= lo:
                    t[i][j] = .5*255
                    
        return self.connectEdges(t, edges)
        
    def connectEdges(self,array,edges):
        # Visited pixel queue
        vis = zeros(array.shape, bool)
        
        # To traverse the array
        dx = [1, 0, -1,  0, -1, -1, 1,  1]
        dy = [0, 1,  0, -1,  1, -1, 1, -1]
        
        for e in edges:
            #if it hasn't been visited
            if not vis[e]:                 
                # depth first search for edges
                q = [e]
                while len(q) > 0:
                    # Dequeue location tuple
                    s = q.pop()
                    vis[s] = True
                    array[s] = 255
                    
                    # Check pixels around edge
                    for k in range(len(dx)):
                        for c in range(1, 16):                            
                            nx, ny = s[0] + c * dx[k], s[1] + c * dy[k]
                            # Check if we are within bounds, haven't visted it, and is closer to high threshold                            
                            if self.exists(nx, ny, array) and (array[nx, ny] >= 0.5*255) and (not vis[nx, ny]):
                                # Add location touple to queue
                                q.append((nx, ny))
        #Set new edges
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if vis[i, j] : array[i, j] = 255
        
        return array
        
    def exists(self, x, y, array):
        return x >= 0 and x < array.shape[0] and y >= 0 and y < array.shape[1]
    
    def maximum(self,array, orientation):
        output = array
        # Traverse the array
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                
                # Make sure we are within bounds
                if((j+1) < array.shape[1]) and ((j-1) >= 0) and ((i+1) < array.shape[0]) and ((i-1) >= 0):
                    
                    # Traverse the pixels adjacent to it
                    for c in range(-1,1):
                        for k in range(-1,1):
                            item = array[i][j]
                            #If there is something bigger then suppress this one
                            if(item < array[i+c][j+k]): 
                                output[i][j] = 0
        return output
    
    def gaussian(self,array,sigma):
        # initialize return value
        output = np.zeros_like(array)
        window = sigma*6 + 1
        # get 1d gaussian vector from function
        kernel = self.kernel(sigma)        
        
        # initialize helper array
        padded = np.zeros((array.shape[0] + window - 1, array.shape[1] + window - 1))
        padded[sigma*3:-(sigma*3), sigma*3:-(sigma*3)] = array #window - 1 /2
        
        #convolve in x direction
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                col = padded[x + sigma*2 + 1,:] 
                flat = col[(y):(window + y)]
                output[x,y] = np.convolve(flat,kernel,'valid')
        
        # Set borders to be the same
        output[:,0] = array[:,0]
        output[0,:] = array[0,:]
        output[:,len(array)-1] = array[:,len(array)-1]
        output[len(array)-1,:] = array[len(array)-1,:]
        
        #reset padded (helper array)
        padded = np.zeros((array.shape[0] + window - 1, array.shape[1] + window - 1))
        padded[sigma*3:-(sigma*3), sigma*3:-(sigma*3)]  = output
            
        #convolve in y direction
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                col = padded[:,y + sigma*3 + 1]
                flat = col[(x):(window + x)]
                output[x,y] = np.convolve(flat,kernel,'valid') 
        
        return output
    
    def kernel(n=3,sigma=1,m = 1):
        #Generate gaussian kernel
        box = sigma*3
        r = range(-box,box + 1)
        return [m / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]   
    
    def lapgaussian(self,array):
        
        output = np.zeros_like(array)
        window = 3
        win = 3
        # Using Laplacian of Gaussian Kernel
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])          
        
        # initialize helper array
        padded = np.zeros((array.shape[0] + win - 1, array.shape[1] + win - 1))
        padded[1:-(1), 1:-(1)] = array #window - 1 /2
        
        #convolve in x direction
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                flat = padded[x:x+window,y:y+window].ravel()
                output[x,y] = np.convolve(flat,kernel.ravel(),'valid')
        
        return output        

class CannyEdgeDetection(object):
    def __init__(self,directory,sigma):
        # Load up pictures        
        image1 = np.asarray(PIL.Image.open(directory))
        
        # Use 1D Gaussians to implement 2D Gaussian filtering
        gauss = self.gaussian(image1,sigma)
        
        # Calculate Gradient
        grad = self.gradient(gauss)
        gradient = grad[0]
        orientation = grad[1] 
        cuadrant = grad[2]   
        
        # Show colored orientation
        coloredGradient = self.colorGradient(orientation,cuadrant,gradient)
        
        # Apply non maximum suppression
        nms = self.maximum(gradient,orientation)
        
        # Apply Hysteresis
        canny = self.hysteresis(nms,sigma)
        
        # Compare with built-in function
        ncanny = cv2.Canny(image1,100/sigma,200/sigma)
        
        # Plot all
        plt.figure(sigma)
        plt.subplots_adjust(hspace = 1)
        plt.subplot(231)
        plt.title('Gaussian')
        plt.imshow(gauss, cmap='gray')
        plt.subplot(232)
        plt.imshow(gradient, cmap='gray')
        plt.title('Gradient')
        plt.subplot(233)
        plt.title('Colored Gradient')
        plt.imshow(coloredGradient)
        plt.subplot(234)
        plt.title('Non Maximum Suppression')
        plt.imshow(nms, cmap = 'gray') 
        plt.subplot(235)
        plt.title('Canny Edge Detection')
        plt.imshow(canny, cmap = 'gray')   
        plt.subplot(236)
        plt.title('Built-in Canny')
        plt.imshow(ncanny, cmap = 'gray')  
        plt.show()            
    
    def gradient(self,array):
        # Sobel filtering
        h = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        v = np.array([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]])
        
        # Fourier transform of kernels
        kernel1 = zeros(array.shape)
        kernel1[:h.shape[0], :h.shape[1]] = h
        kernel1 = fft2(kernel1)
    
        kernel2 = zeros(array.shape)
        kernel2[:v.shape[0], :v.shape[1]] = v
        kernel2 = fft2(kernel2)
        
        # Apply kernels
        farray = fft2(array)
        Gx = real(ifft2(kernel1 * farray)).astype(float)
        Gy = real(ifft2(kernel2 * farray)).astype(float)
    
        G = sqrt(Gx**2 + Gy**2)
        Theta = arctan2(Gy, Gx) * 180 / pi
        
        cuadrant = np.zeros_like(array)
        
        # Save which quadrant it is on
        for x in range(cuadrant.shape[0]):
            for y in range(cuadrant.shape[1]):
                if (Gx[x,y] > 0):
                    if(Gy[x,y] > 0):
                        cuadrant[x,y] = 1
                    else:
                        cuadrant[x,y] = 3
                else:
                    if(Gy[x,y] > 0):
                        cuadrant[x,y] = 2
                    else:
                        cuadrant[x,y] = 4
        
        
        return G, Theta, cuadrant
    
    def hysteresis(self,array,sigma=1):
        # Do thresholding
        t  = np.zeros_like(array)        
        lo, hi = 100/sigma, 160/sigma 
        #lo, hi = 10/sigma,50/sigma #for picture 4 we neeed low thresholds
        edges = []
        
        # Set 255 (white) as a strong edge) and 0 as non edges
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] >= hi:
                    t[i][j] = 255
                    # Making a list of edges to traverse
                    edges.append((i, j))
                elif array[i][j] >= lo:
                    t[i][j] = .5*255
                    
        return self.connectEdges(t, edges)
        
    def connectEdges(self,array,edges):
        # Visited pixel queue
        vis = zeros(array.shape, bool)
        
        # To traverse the array
        dx = [1, 0, -1,  0, -1, -1, 1,  1]
        dy = [0, 1,  0, -1,  1, -1, 1, -1]
        
        for e in edges:
            #if it hasn't been visited
            if not vis[e]:                 
                # depth first search for edges
                q = [e]
                while len(q) > 0:
                    # Dequeue location tuple
                    s = q.pop()
                    vis[s] = True
                    array[s] = 255
                    
                    # Check pixels around edge
                    for k in range(len(dx)):
                        for c in range(1, 16):                            
                            nx, ny = s[0] + c * dx[k], s[1] + c * dy[k]
                            # Check if we are within bounds, haven't visted it, and is closer to high threshold                            
                            if self.exists(nx, ny, array) and (array[nx, ny] >= 0.5*255) and (not vis[nx, ny]):
                                # Add location touple to queue
                                q.append((nx, ny))
        #Set new edges
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if vis[i, j] : array[i, j] = 255
        
        return array
        
    def exists(self, x, y, array):
        return x >= 0 and x < array.shape[0] and y >= 0 and y < array.shape[1]
    
    def maximum(self,array, orientation):
        output = array
        # Traverse the array
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                
                # Make sure we are within bounds
                if((j+1) < array.shape[1]) and ((j-1) >= 0) and ((i+1) < array.shape[0]) and ((i-1) >= 0):
                    
                    # Traverse the pixels adjacent to it
                    for c in range(-1,1):
                        for k in range(-1,1):
                            item = array[i][j]
                            #If there is something bigger then suppress this one
                            if(item < array[i+c][j+k]): 
                                output[i][j] = 0
        return output
   
    
    def colorGradient(self,array,cuadrant, magnitude):
        # Create colored gradient
        colored = np.zeros([array.shape[0], array.shape[1], 3], dtype=np.uint8)
        
        for row in range(array.shape[0]):
            for x in range(array.shape[1]):
                # Calculate the angle to color
                colored[row,x] = self.angleToColor(array[row,x],cuadrant[row,x], magnitude[row,x])
            
        #return gradient to show
        return colored
        
    def angleToColor(self, angle, cuadrant, magnitude):
        # Reset to positive (we don't need negative since we have cuadrant info)
        if angle < 0:
            angle += 180
        color = 0
        
        if(cuadrant == 1):
            if (angle < 45):
                color = (255, 0, 0)
            elif (angle >= 45):
                color = (0, 0, 255)
        elif(cuadrant == 2):
            if (angle < 45):
                color =(180, 0, 180)
            elif (angle >= 45):
                color = (130, 78, 62)
        elif(cuadrant ==3):
            if (angle < 45):
                color = (255, 255, 0)
            elif (angle >= 45):
                color = (0, 255, 255)
        elif(cuadrant == 4):
            if (angle < 45):
                color = (255, 0, 255)
            elif (angle >= 45):
                    color = (0, 0, 255)
        
        return color
 
    def gaussian(self,array,sigma):
        # initialize return value
        output = np.zeros_like(array)
        window = sigma*6 + 1
        # get 1d gaussian vector from function
        kernel = self.kernel(sigma)        
        
        # initialize helper array
        padded = np.zeros((array.shape[0] + window - 1, array.shape[1] + window - 1))
        padded[sigma*3:-(sigma*3), sigma*3:-(sigma*3)] = array #window - 1 /2
        
        #convolve in x direction
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                col = padded[x + sigma*2 + 1,:] 
                flat = col[(y):(window + y)]
                output[x,y] = np.convolve(flat,kernel,'valid')
        
        # Set borders to be the same
        output[:,0] = array[:,0]
        output[0,:] = array[0,:]
        output[:,len(array)-1] = array[:,len(array)-1]
        output[len(array)-1,:] = array[len(array)-1,:]
        
        #reset padded (helper array)
        padded = np.zeros((array.shape[0] + window - 1, array.shape[1] + window - 1))
        padded[sigma*3:-(sigma*3), sigma*3:-(sigma*3)]  = output
            
        #convolve in y direction
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                col = padded[:,y + sigma*3 + 1]
                flat = col[(x):(window + x)]
                output[x,y] = np.convolve(flat,kernel,'valid') 
        
        return output
    
    def kernel(n=3,sigma=1,m = 1):
        #Generate gaussian kernel
        box = sigma*3
        r = range(-box,box + 1)
        return [m / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]      
    
    
program2 = Program2()