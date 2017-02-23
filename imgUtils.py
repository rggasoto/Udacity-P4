import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


class imgUtils:

    def __init__(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        self.debug = False

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
    def sat_threshold(self,img,thresh=(120,255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        l_channel = hls[:,:,1]
        s_thresh_min = thresh[0]
        s_thresh_max = thresh[1]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        return s_binary
    def getMaskedWarpedImage(self,undist):
        # for image in glob.glob('test_images/*.jpg'):

            # dist = cv2.imread(image)

            # if i < 510:
            #     continue
            # if i > 650:
            #     if i < 950:
            #         continue

            ksize = 3
            undistyuv = cv2.cvtColor(undist,cv2.COLOR_BGR2YUV)
            undistgray = undistyuv[:,:,0]
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(3,3))
            undistgray = clahe.apply(undistgray)
            undistyuv[:,:,0] = undistgray
            undistclean = undist
            undist = cv2.cvtColor(undistyuv,cv2.COLOR_YUV2BGR)

            gradx = self.abs_sobel_thresh(undistgray, orient='x', sobel_kernel=ksize, thresh=(40, 130))
            grady  = self.abs_sobel_thresh(undistgray, orient='y', sobel_kernel=ksize, thresh=(50, 100))
            mag_binary = self.mag_thresh(undistgray, sobel_kernel=ksize, mag_thresh=(40, 900))
            dir_binary = self.dir_threshold(undistgray,  sobel_kernel=21, thresh=(0.6, 1.4))
            # kernel = np.ones((8,8),np.uint8)
            # dir_binary = cv2.morphologyEx(np.uint8(dir_binary), cv2.MORPH_OPEN, kernel)
            #print(gradx)
            combined = np.zeros_like(dir_binary)
            combined[((gradx >= 1) & (grady >= 1)) | ((mag_binary >= 1) & (dir_binary >= 1))] = 1
            s_thresh = self.sat_threshold(undistclean)
            kernel = np.ones((10,10),np.uint8)
            s_mask = cv2.morphologyEx(np.uint8(s_thresh), cv2.MORPH_OPEN, kernel)
            s_unmasked = s_thresh
            s_thresh[s_mask > 0] = 0
            s_thresh = cv2.morphologyEx(np.uint8(s_thresh), cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((8,8),np.uint8)
            s_thresh = cv2.morphologyEx(np.uint8(s_thresh), cv2.MORPH_DILATE, kernel)
            kernel = np.ones((4,4),np.uint8)
            s_thresh = cv2.morphologyEx(np.uint8(s_thresh), cv2.MORPH_ERODE, kernel)

            combined[s_thresh >=1] = 1


            undistwarped = self.perspectiveTransform(undist)
            #undist = cv2.line(undist,top_left,top_right,thickness = 2,color = (255,0,0))
            #undist = cv2.line(undist,top_left,bottom_left,thickness = 2,color = (255,0,0))

            #undist = cv2.line(undist,bottom_left,bottom_right,thickness = 2,color = (255,0,0))
            #undist = cv2.line(undist,top_right,bottom_right,thickness = 2,color = (255,0,0))

            #undistwarped = cv2.line(undistwarped,wtop_left,wtop_right,thickness = 2,color = (255,0,0))
            #undistwarped = cv2.line(undistwarped,wtop_left,wbottom_left,thickness = 2,color = (255,0,0))

            #undistwarped = cv2.line(undistwarped,wbottom_left,wbottom_right,thickness = 2,color = (255,0,0))
            #undistwarped = cv2.line(undistwarped,wtop_right,wbottom_right,thickness = 2,color = (255,0,0))
            combinedWarped = self.perspectiveTransform(combined)
            kernel = np.ones((2,2),np.uint8)
            combinedWarped = cv2.morphologyEx(np.uint8(combinedWarped), cv2.MORPH_OPEN, kernel)
            kernel = np.ones((20,20),np.uint8)
            combinedWarped = cv2.morphologyEx(np.uint8(combinedWarped), cv2.MORPH_CLOSE, kernel)

            if self.debug:
                combinedWarped[combinedWarped>=1]=255
                combined[combined >=1] = 255
                gradx[gradx > 0] = 255
                grady[grady > 0] = 255
                mag_binary[mag_binary>0] = 255
                dir_binary[dir_binary>0] = 255
                s_thresh[s_thresh>0] = 255
                s_unmasked[s_unmasked>0] = 255
                img = diagPanel(undist,cv2.cvtColor(np.uint8(gradx),cv2.COLOR_GRAY2BGR),
                                       cv2.cvtColor(np.uint8(grady),cv2.COLOR_GRAY2BGR),
                                       cv2.cvtColor(np.uint8(mag_binary),cv2.COLOR_GRAY2BGR),
                                       cv2.cvtColor(np.uint8(dir_binary),cv2.COLOR_GRAY2BGR),
                                       cv2.cvtColor(np.uint8(combined),cv2.COLOR_GRAY2BGR),
                                       undistwarped,cv2.cvtColor(np.uint8(combinedWarped),cv2.COLOR_GRAY2BGR),
                                       cv2.cvtColor(np.uint8(s_thresh),cv2.COLOR_GRAY2BGR),
                                       cv2.cvtColor(np.uint8(s_unmasked),cv2.COLOR_GRAY2BGR),0)
                cv2.imshow('pipeline',img)
                # cv2.waitKey(-1)

            return np.dstack((s_thresh,combined,np.zeros_like(s_thresh)))*255, combinedWarped
