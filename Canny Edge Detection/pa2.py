import numpy as np
import PIL
from numpy import pi, exp, sqrt, array, zeros, abs, arctan2, arctan, real
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from matplotlib.pyplot import imshow, show, subplot, figure, title, axis
import cv2
class Program2(object):
    def __init__(self):
        done = 'n'
        
    #while(done == 'n'):
        #image = input("Please enter an image directory: ")
        directory = 'C:\\Users\\nicol\\Desktop\\pa2\\picture1.png'
        image = directory
        # It will perform Canny Edge Detection at different sigmas
        for sigma in range(1,2):
            print("Processing image", image, " at sigma = ", sigma)
            CannyEdgeDetection(image, sigma)
            
            #done = input("Done ? (y) : (n) ")

class CannyEdgeDetection(object):
    def __init__(self,directory,sigma):
        
        # Load up pictures        
        image1 = np.asarray(PIL.Image.open(directory))
        
        # Use 1D Gaussians to implement 2D Gaussian filtering
        gauss = self.gaussian(image1,sigma)
        
        # Plot it
        # plt.figure(sigma)
        # plt.subplot(151)
        # plt.title('Gaussian')
        # plt.imshow(gauss, cmap='gray')
        cv2.imwrite('C:\\Users\\nicol\\Desktop\\pa2\\gauss.png',gauss)
        
        grad = self.gradient(gauss)
        gradient = grad[0]
        orientation = grad[1] 
        cuadrant = grad[2]               
                                
        # Plot it
        # plt.subplot(152)
        # plt.imshow(gradient, cmap='gray')
        # plt.title('Gradient')
        # plt.show()
        
        cv2.imwrite('C:\\Users\\nicol\\Desktop\\pa2\\gradient.png',gradient)
        # Show colored orientation
        coloredGradient = self.colorGradient(orientation,cuadrant,gradient)
        cv2.imwrite('C:\\Users\\nicol\\Desktop\\pa2\\colored.png',coloredGradient)
        # Plot it
        # plt.subplot(153)
        # plt.title('Colored Gradient')
        # plt.imshow(coloredGradient)
        
        # Apply non maximum suppression
        nms = self.maximum(gradient,orientation)
        
        # Plot it
        # plt.subplot(154)
        # plt.title('Non Maximum Suppression')
        # plt.imshow(nms, cmap = 'gray')
        cv2.imwrite('C:\\Users\\nicol\\Desktop\\pa2\\nms.png',nms)
        # Apply Hysterisis
        canny = self.hysteresis(nms)
        # 
        #  # Plot it
        # plt.subplot(155)
        # plt.title('Canny Edge Detection')
        # plt.imshow(canny, cmap = 'gray')
        cv2.imwrite('C:\\Users\\nicol\\Desktop\\pa2\\canny.png',canny)
    
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
        Theta = arctan(Gy, Gx) * 180 / pi
        
        cuadrant = np.zeros_like(array)
        
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
    
    def hysteresis(self,array):
        # Do thresholding
        t  = np.zeros_like(array)
        mmax = 255
        lo, hi = 0.1 * mmax, 0.8 * mmax
        edges = []
        
        # Set 255 (white) as a strong edge) and 0 as non edges
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] >= hi:
                    t[i][j] = 255
                    # Making a list of edges to traverse
                    edges.append((i, j))
                elif array[i][j] >= lo:
                    t[i][j] = 0
                    
        return self.connectEdges(t, edges)
        
    def connectEdges(self,array,edges):
        # Visited queue
        vis = zeros(array.shape, bool)
        
        # To traverse the array
        dx = [1, 0, -1,  0, -1, -1, 1,  1]
        dy = [0, 1,  0, -1,  1, -1, 1, -1]
        
        for e in edges:
            if not vis[e]: #if it hasn't been visited
                
                # depth first search for edges
                q = [e]
                while len(q) > 0:
                    s = q.pop()
                    vis[s] = True
                    array[s] = 255
                    
                    for k in range(len(dx)):
                        for c in range(1, 16):
                            
                            nx, ny = s[0] + c * dx[k], s[1] + c * dy[k]
                            
                            if self.exists(nx, ny, array) and (array[nx, ny] >= 0.5*255) and (not vis[nx, ny]):
                                q.append((nx, ny))
                                
        #Set new edges
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i, j] = 255 if vis[i, j] else 0
        
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
        
    def angleToColor(self, theta, cuadrant, magnitude):
        
        
        if theta < 0:
            theta += 180
        angle = theta
        
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
                color = (180, 180, 180)
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
        box = sigma*3
        r = range(-box,box + 1)
        return [m / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]      
    
    
program2 = Program2()