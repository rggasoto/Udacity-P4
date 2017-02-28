
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_cal.png "Undistorted"
[image2]: ./output_images/undist_1.png "Undistorted"
[pipeline1]: ./output_images/binary_1.png "Binary pipeline 1"
[pipeline2]: ./output_images/binary_S_1.png "Pipeline add Saturation filter 1"
[pipeline3]: ./output_images/binary_warped_1.png "Pipeline Morphology convolution to remove noise and warping to top view"
[fitted]: ./output_images/fitted_poly.png "Fitted polynomial to pixels detected"
[warped]: ./output_images/perspectiveTransf.png "Perspective transform for birds eye view"
[output]: ./output_images/markedOutput.png "Output"
[video1]: ./output_images/output.mov "Video"

###Camera Calibration

The camera was calibrated using the chessboard process, in which it contained tiles organized in 6 rows and 9 columns.
The first step to such process is to convert the image in grayscale and find the chessboard corners using opencv built-in function:

```python
 gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
```
As the image might be incomplete or have other issues that prevent it from finding the corners, `ret` stores whether it was successful in finding the corners.
If it was successful, the corners are appended in a list of points, that later are used in the other built-in function `cv2.calibrateCamera` to store the retifying matrix used to undistort images.
in order to simplify the usage of such matrix, a wrapper function was created in the class imgUtils such that when called it returns the undistorted image:

```python
def undistort(self,img):
        und = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return und
```

After running the calibration using the example images provided, 3 images were unable to find the correct amount of corners (images 1, 4 and 5), so they were discarded from the calibration process.
The output from caling the function undistort can be seen in the image below.
![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
After the camera is calibrated, the remainder of the pipeline is done over an image that is already undistorted. To demonstrate the result of undistortion on a lane image, see image below:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The pipeline for lane finding contains the following steps:
* Equalize Grayscale histogram
* Apply gradient filter with threshold over equalized image on axis X and Y
	* Merge X and Y filters together where both are positive
* Apply directional gradient and magnitude filter over equalized image
	* Merge Directional and Magnitude filters where both are positive
* Apply saturation Threshold over image
	* Remove large image blobs by applying a morphological OPEN and mask out the result.
	* Close gaps with morphological Close operation
* Merge Saturation, magnitude and gradient filters keeping all selected pixels.
* Filter small remaining noise with morphological Close
* Warp image to bird's view perspective of road.

#####Gradient Filter
The gradient Filter is performed in function `abs_sobel_thresh`, defined in class imgUtils as follows:
``` python
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

```
the same function is used to apply x and y directions. later, the result is combined with the code `grad[((gradx >= 1) & (grady >= 1))]=1`
The result is seen in the first line of the next image, where the first image is in X direction, the second is in Y direciton, and the purple-yellow image is the resulting combination, applied to test image 5.

#####Magnitude and Directional Gradient
The next step in the pipeline is compute the magnitude of change and the directional threshold. This is done using the `imgUtils` functions `mag_thresh` and `dir_threshold`, defined as follows:

``` python
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
```
The result is seen in the second row of the next image, where the last column is the resulting combination of both finters done with the code `dir[((mag_binary >= 1) & (dir_binary >= 1))]=1`

![alt text][pipeline1]

The results for gradient and magnitude filters are then merged together (first row of image).

#####Saturation Filter
In order to improve robustness, an additional filter of Saturation threshold is done over the image, as defined in function `sat_threshold` of class `imgUtils`:

``` python

    def sat_threshold(self,img,thresh=(125,255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        l_channel = hls[:,:,1]
        s_thresh_min = thresh[0]
        s_thresh_max = thresh[1]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        return s_binary

```
as can be seen in the first image of the second row, there is significant amount of noise using this threshold, so morphologicals operations of open and close are done to remove this noise in function `getMaskedWarpedImage` of class `imgUtils`:
``` python
	s_thresh[halfy:,:] = self.sat_threshold(undistclean[halfy:,:])
	kernel = np.ones((13,13),np.uint8)
	s_mask = cv2.morphologyEx(np.uint8(s_thresh[:,:]), cv2.MORPH_CLOSE, kernel)            
	s_mask = cv2.morphologyEx(np.uint8(s_mask), cv2.MORPH_OPEN, kernel)
	# Mask out large blobs            
	s_thresh[s_mask > 0] = 0            
	s_thresh[halfy:,:] = cv2.morphologyEx(np.uint8(s_thresh[halfy:,:]), cv2.MORPH_CLOSE, kernel)
```

![alt text][pipeline2]

The resulting filter is merged with the previous filter, and then another morphological operation is done to elminate small noise from the image before warping.


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In order to warp the image, helper functions were made in `imgUtils` to set the warping parameters, get perspective distorted and the reverse operation:
``` python
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
```
for the filter, the resulting warped image can be seen in the last image of figure 5

![alt text][pipeline3]



The warping parameters are the mapping of 4 Source points to 4 destination points, which were done empirically using the test road images as parameter, which resulted in the following warping coordinates


| Source        | Destination   |
|:-------------:|:-------------:|
| 580, 460      | 320, 0        |
| 206, 720      | 320, 720      |
| 1100, 720     | 960, 720      |
| 703, 460      | 960, 0        |

Another example can be seen here:
![alt text][warped]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The file `pipeline.py` contains the logic that gets the warped binary image from imgUtils and identify where the lanes are. In order to help with the process, a class `LineFit` was created, that stores information from the line, and perform actions such as obtaining a weighted average of the last valid measurements.
#####Line detection
Whenever it is the first time it's detecting a lane, the code must first know where to look for one. In order to do it, a histogram is taken, and the max peak for each side of the image is selected as starting search point for the lane, which is then conducted using a sliding window algorithm (if measures in the window where the center of mass is, and move the window center to that position).
This is done in line 109 of `pipeline.py`.
If a previous match already exists for the lane (individually for left and right lanes), then instead of sliding window, a region around the fitted line is selected for matching the next polynomial
```python
	margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    # Fit a second order polynomial to each
    left_fit_t = np.polyfit(lefty, leftx, 2)
```

once a polynomial is fitted into the detected lane, some sanity checks are performed.
First, due to the nature of the problem, it is expected that the base of the lane won't change much from previous frame. Therefore, if the change exceeds a given threshold, the frame will be dropped.
If both frame bases are accepted, the next check is whether the top of the lane has a similar width as the base (which means the lane will be parallel). If no lane was dropped, and the difference between the base and top exceeds a threshold, then both lane readings are dropped.

Once the lane receives an accepted measurement, its dropped frames counter is reset. If the counter reaches the limit of dropped frames, the entire reading buffer is reset and a new search for a lane needs to be performed.

![alt text][fitted]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

the radius of curvature of the lanes is computed using the formula provided in lecture, in with the following code:
```python
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
avg_curvature = (left_curverad + right_curverad)/2
```

Considering that the camera is centered with the vehicle, the car location with respect to center is done in a similar fashion, by computing the base of the polynomial, and comparing against the center of the picture:
```python
l_base = left_fit[0]*binary_warped.shape[0]**2 + left_fit[1]*binary_warped.shape[0] + left_fit[2]
r_base = right_fit[0]*binary_warped.shape[0]**2 + right_fit[1]*binary_warped.shape[0] + right_fit[2]
car_center = (binary_warped.shape[1]/2 - (l_base + r_base)/2)*xm_per_pix
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


![alt text][output]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/output.avi)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, a robust pipeline for lane detection was implemented. In some specific scenarios, however, it might still fail. Sudden curvature changes, faded lanes, or heavily ocluded lanes might make the detection fail. The persistence model should be able to handle some of it, however it will likely fail if the lanes are not detected for a long period.
Another point of failure is that the current perspective warping assumes that the lane is always plane to the vehicle. it can be seen that this is not entirely true, since the vehicle dampers show that the lanes wobble back and forth when the vehicle moves on a bumpy area.
One way to improve the pipeline would be implementing a second-order persistence model, where it tracks the rate of change of the lanes, such that it can predict with better accuracy where the lane will be in future frames.
Another improvements can be done with a more reliable segmentation. The current method relies on the ability to either detect the gradient of the lane, or in the saturation of it. However good in the general case, it can be seen that it wobbles a little bit in situations where neither are present.
In order to prevent the wobbling caused by the vehicle dampers and non-flat roads, a dynamic perspective transform based on the parallelism of the lanes could be implemented, such that it guarantees that the lanes detected will be parallel within a tolerance level, given that the algorithm has a minimal certainty about the lanes detected in previous measurements (that could eb achieved with computing the average pixel detection per lane and setting a threshold on it to give the desired certainty)
