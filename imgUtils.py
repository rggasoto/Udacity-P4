import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def diagPanel(main,diag1,diag2,diag3,diag4,diag5,diag6,diag7,diag8,diag9,inp1):
        # middle panel text example
    # using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated lane curvature: {}'.format(inp1), (30, 60), font, 1, (255,0,0), 2)
    #cv2.putText(middlepanel, 'Estimated Meters right of center: {}'.format(inp2), (30, 90), font, 1, (255,0,0), 2)


    # assemble the screen example
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = main
    diagScreen[0:240, 1280:1600] = cv2.resize(diag1, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[0:240, 1600:1920] = cv2.resize(diag2, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1280:1600] = cv2.resize(diag3, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1600:1920] = cv2.resize(diag4, (320,240), interpolation=cv2.INTER_AREA)*4
    diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)*4
    diagScreen[720:840, 0:1280] = middlepanel
    diagScreen[840:1080, 0:320] = cv2.resize(diag5, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[840:1080, 320:640] = cv2.resize(diag6, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[840:1080, 640:960] = cv2.resize(diag9, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[840:1080, 960:1280] = cv2.resize(diag8, (320,240), interpolation=cv2.INTER_AREA)
    return diagScreen

class imgUtils:

    def __init__(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        self.debug = True

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                #cv2.imshow(fname,img)
                #cv2.waitKey(-1)
            else:
                print(fname)

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
       
        #input()
    def undistort(self,img):
        und = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return und
    def setPerspectiveParams(self,src,dst):
        self.dst = dst
        self.src = src
        self.M = cv2.getPerspectiveTransform(src, dst)
    def perspectiveTransform(self,img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size)
    def perspectiveInverse(self,img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, np.linalg.inv(self.M), img_size)

    def abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0,255)):

        thresh_min = thresh[0]
        thresh_max = thresh[1]
        # Apply the following steps to img
        # 1) Convert to grayscale
        if(len(img.shape) > 2):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)[:,:,0]
        else:
            gray = img
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        x = orient == 'x'
        y = orient == 'y'
        # 3) Take the absolute value of the derivative or gradient
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, x, y,ksize=sobel_kernel))
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*sobel/np.max(sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
                # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        # 6) Return this mask as your binary_output image
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_output

    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):

        # Apply the following steps to img
        # 1) Convert to grayscale
        # 1) Convert to grayscale
        if(len(img.shape) > 2):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)[:,:,0]
        else:
            gray = img
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        #binary_output = np.copy(img) # Remove this line
        return binary_output

    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi/2)):

        # Apply the following steps to img
        # 1) Convert to grayscale
        if(len(img.shape) > 2):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)[:,:,0]
        else:
            gray = img
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    def sat_threshold(self,img,thresh=(125,255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        l_channel = hls[:,:,1]
        s_thresh_min = thresh[0]
        s_thresh_max = thresh[1]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        return s_binary


    def getMaskedWarpedImage(self,undist):
            
            ksize = 3
            undistyuv = cv2.cvtColor(undist,cv2.COLOR_BGR2YUV)
            undistgray = undistyuv[:,:,0]
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(3,3))
            #Normalize image histogram on Grayscale channel
            undistgray = clahe.apply(undistgray)
            
            undistyuv[:,:,0] = undistgray
            undistclean = undist

            undist = cv2.cvtColor(undistyuv,cv2.COLOR_YUV2BGR)
            gradx = np.zeros((undistclean.shape[0],undistclean.shape[1]))
            grady = np.zeros_like(gradx)
            mag_binary = np.zeros_like(grady)
            dir_binary = np.zeros_like(gradx)
            s_thresh = np.zeros_like(gradx)
            halfy = int(gradx.shape[0]/2)
            #Generate  filters for Gradient, magnitude and direction thresholds (apply only in half image as above horizon is not relevant
            gradx[halfy:,:] = self.abs_sobel_thresh(undistgray[halfy:,:], orient='x', sobel_kernel=ksize, thresh=(40, 130))
            grady[halfy:,:]  = self.abs_sobel_thresh(undistgray[halfy:,:], orient='y', sobel_kernel=ksize, thresh=(50, 100))
            mag_binary[halfy:,:] = self.mag_thresh(undistgray[halfy:,:], sobel_kernel=ksize, mag_thresh=(30, 900))
            dir_binary[halfy:,:] = self.dir_threshold(undistgray[halfy:,:],  sobel_kernel=5, thresh=(0.7, 1.3))
            # placeholder for compbined filters
            combined = np.zeros_like(gradx)
            combined[((gradx >= 1) & (grady >= 1)) | ((mag_binary >= 1) & (dir_binary >= 1))] = 1
            

            
            #new = np.zeros_like(dir_binary)
            #new[ ((mag_binary >= 1) & (dir_binary >= 1))] = 1
            #dir_binary = new
            
            # Color Threshold
            s_thresh[halfy:,:] = self.sat_threshold(undistclean[halfy:,:])
            kernel = np.ones((13,13),np.uint8)
            s_mask = cv2.morphologyEx(np.uint8(s_thresh[:,:]), cv2.MORPH_CLOSE, kernel)            
            s_mask = cv2.morphologyEx(np.uint8(s_mask), cv2.MORPH_OPEN, kernel)
            # Mask out large blobs
            s_unmasked = np.copy(s_thresh)
            s_thresh[s_mask > 0] = 0
            s_open = np.copy(s_thresh)
            s_thresh[halfy:,:] = cv2.morphologyEx(np.uint8(s_thresh[halfy:,:]), cv2.MORPH_CLOSE, kernel)
            #kernel = np.ones((10,10),np.uint8)
            #s_thresh = cv2.morphologyEx(np.uint8(s_thresh), cv2.MORPH_DILATE, kernel)
            #kernel = np.ones((5,5),np.uint8)
            #s_thresh = cv2.morphologyEx(np.uint8(s_thresh), cv2.MORPH_ERODE, kernel)
            
            combined[s_thresh >=1] = 1
            new_combined = np.copy(combined)

            #undistwarped = self.perspectiveTransform(undist)

            # filter out small noise
            kernel = np.ones((2,2),np.uint8)
            combined[halfy:,:] = cv2.morphologyEx(np.uint8(combined[halfy:,:]), cv2.MORPH_OPEN, kernel)
            # fill in gaps
            kernel = np.ones((5,5),np.uint8)
            combined[halfy:,:] = cv2.morphologyEx(np.uint8(combined[halfy:,:]), cv2.MORPH_CLOSE, kernel)
            combinedWarped = self.perspectiveTransform(combined)
            
            #kernel = np.ones((20,20),np.uint8)
            #combinedWarped = cv2.morphologyEx(np.uint8(combinedWarped), cv2.MORPH_CLOSE, kernel)
            #plt.figure(1)
            #plt.subplot(2,3,1)
            #plt.imshow(gradx, cmap = 'gray')
            #plt.subplot(2,3,2)
            #plt.imshow(grady, cmap = 'gray')
            #plt.subplot(2,3,3)
            #grad = np.zeros_like(gradx)
            #grad[((gradx >= 1) & (grady >= 1))]=1
            #plt.imshow(grad)
            #plt.subplot(2,3,4)
            #plt.imshow(mag_binary, cmap = 'gray')
            #plt.subplot(2,3,5)
            #plt.imshow(dir_binary, cmap = 'gray')
            #plt.subplot(2,3,6)
            #dir = np.zeros_like(gradx)
            #dir[((mag_binary >= 1) & (dir_binary >= 1))]=1
            #plt.imshow(dir)
            

            #plt.figure(2)
            #color_binary = np.dstack(( np.zeros_like(dir), dir, grad))
            #plt.subplot(2,3,1)
            #plt.imshow(grad,cmap = 'gray')
            #plt.subplot(2,3,2)
            #plt.imshow(dir,cmap = 'gray')
            #plt.subplot(2,3,3)
            #plt.imshow(color_binary)

            #plt.subplot(2,3,4)
            #plt.imshow(s_unmasked,cmap='gray')
            #plt.subplot(2,3,5)
            #plt.imshow(s_open,cmap='gray')
            #plt.subplot(2,3,6)
            #plt.imshow(s_thresh)
            

            #plt.figure(3)
            #color_binary = np.dstack(( s_thresh, dir, grad))
            #plt.subplot(2,2,1)
            #plt.imshow(color_binary)
            #plt.subplot(2,2,2)
            #plt.imshow(new_combined,cmap = 'gray')
            #plt.subplot(2,2,3)
            #plt.imshow(combined,cmap = 'gray')
            #plt.subplot(2,2,4)
            #plt.imshow(combinedWarped,cmap = 'gray')
            
            #plt.show()
            #if self.debug:
            #    combinedWarped[combinedWarped>=1]=255
            #    combined[combined >=1] = 255
            #    gradx[gradx > 0] = 255
            #    grady[grady > 0] = 255
            #    mag_binary[mag_binary>0] = 255
            #    dir_binary[dir_binary>0] = 255
            #    s_thresh[s_thresh>0] = 255
            #    s_unmasked[s_unmasked>0] = 255
            #    img = diagPanel(undist,cv2.cvtColor(np.uint8(gradx),cv2.COLOR_GRAY2BGR),
            #                           cv2.cvtColor(np.uint8(grady),cv2.COLOR_GRAY2BGR),
            #                           cv2.cvtColor(np.uint8(mag_binary),cv2.COLOR_GRAY2BGR),
            #                           cv2.cvtColor(np.uint8(dir_binary),cv2.COLOR_GRAY2BGR),
            #                           cv2.cvtColor(np.uint8(combined),cv2.COLOR_GRAY2BGR),
            #                           undistwarped,cv2.cvtColor(np.uint8(combinedWarped),cv2.COLOR_GRAY2BGR),
            #                           cv2.cvtColor(np.uint8(s_thresh),cv2.COLOR_GRAY2BGR),
            #                           cv2.cvtColor(np.uint8(s_unmasked),cv2.COLOR_GRAY2BGR),0)
            #    cv2.imshow('pipeline',img)
                # cv2.waitKey(-1)

            return undist, combinedWarped
