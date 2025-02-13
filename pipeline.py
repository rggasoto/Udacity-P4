import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imgUtils import imgUtils
import glob


cap = cv2.VideoCapture('project_video.mp4')
i = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoout = cv2.VideoWriter('output.mov',fourcc, 30.0,( 1280,720))
utils = imgUtils()
makeReport = True;
# top_left = (631,427)
# top_right = (647,427)
# bottom_left = (257,684)
# bottom_right = (1049,684)
top_left = (580,460)
top_right = (703,460)
bottom_left = (206,720)
bottom_right = (1100,720)


poly = np.array([top_left,top_right,bottom_right,bottom_left],np.float32).reshape(-1,2)
#print (poly)
# undist = cv2.fillpoly(undist,poly,1,thickness = 2,color = (255,0,0,60))


wtop_left = (320,0)
wtop_right = (960,0)
wbottom_left = (320,720)
wbottom_right = (960,720)
wpoly = np.array([wtop_left,wtop_right,wbottom_right,wbottom_left],np.float32).reshape(-1,2)

utils.setPerspectiveParams(poly,wpoly)

class LineFit:
    def __init__(self):
        self.fit = None
        self.n = 10
        self.buffer = np.zeros((self.n,3))
        self.count = 0
        self.index = -1
        self.weights = [[10,1,2,3,4,5,6,7,8,9],
          [9,10,1,2,3,4,5,6,7,8],
          [8,9,10,1,2,3,4,5,6,7],
          [7,8,9,10,1,2,3,4,5,6],
          [6,7,8,9,10,1,2,3,4,5],
          [5,6,7,8,9,10,1,2,3,4],
          [4,5,6,7,8,9,10,1,2,3],
          [3,4,5,6,7,8,9,10,1,2],
          [2,3,4,5,6,7,8,9,10,1],
          [1,2,3,4,5,6,7,8,9,10]]
        self.weights = np.ones((self.n,self.n))
        self.frameskip = 0;
        self.curve_r = -1
        self.frames_dropped = 0
    def reset(self):
        self.fit = None;
        self.buffer = np.zeros((self.n,3))
        self.count = 0
        self.index = -1
        self.frames_dropped = 0
    def add(self,line):
        if not line is None:
            self.index = (self.index+1)%self.n
            self.buffer[self.index] = line
            self.count = min(self.n,self.count+1)
            #self.frames_dropped = 0
    def get(self):
        w = self.weights[self.index];
        total = np.sum(w[:self.count])
        return np.dot(w,self.buffer)/total
left = LineFit()
right = LineFit()
left_w = LineFit()
right_w = LineFit()
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
leftx_base = None
rightx_base = None
i = -1
avg_curvature = None;
avg_curve = None;
while cap.isOpened():
    ret,dist = cap.read()
    if dist is None:
        break
#for image in glob.glob('test_images/*.jpg'):
    i = i+1
    #if i < 1030:
    #    continue
    #if i > 650:
    #    if i < 950:
    #        continue
    #dist = cv2.imread(image)
    # dist = cv2.cvtColor(dist,cv2.COLOR_BGR2RGB)
    undist = utils.undistort(dist)
    warped,binary_warped = utils.getMaskedWarpedImage(undist)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
 # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    if (left.count == 0):
        if(leftx_base is None):
            leftx_base = np.argmax(histogram[:midpoint])
        # Choose the number of sliding windows
        nwindows = 8
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Current positions to be updated for each window
        leftx_current = leftx_base
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
          # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        # Fit a second order polynomial to each
        left_fit_t = np.polyfit(lefty, leftx, 2)
        left.add(left_fit_t);
        left_fit = left.get()
        left_w.add(np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2))
    else:
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        # Fit a second order polynomial to each
        left_fit_t = np.polyfit(lefty, leftx, 2)


    if right.count == 0 :
        if(rightx_base is None):
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 8
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Current positions to be updated for each window
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
         # Concatenate the arrays of indices
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        right_fit_t = np.polyfit(righty, rightx, 2)
        right.add(right_fit_t);
        right_fit = right.get()
        right_w.add(np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2))
    else:
        margin = 100
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        right_fit_t = np.polyfit(righty, rightx, 2)
    #If both lanes have a measurement and there's a previous curvature, perform sanity checks
    if left.count !=0 and right.count !=0 and not avg_curve is None:
        y_eval = np.max(ploty)
        left_curverad_t = ((1 + (2*left_fit_t[0]*y_eval + left_fit_t[1])**2)**1.5) / np.absolute(2*left_fit_t[0])
        right_curverad_t = ((1 + (2*right_fit_t[0]*y_eval + right_fit_t[1])**2)**1.5) / np.absolute(2*right_fit_t[0])
        l_fit_f = np.copy(left_fit_t)
        r_fit_f = np.copy(right_fit_t)
        tolerance = avg_curve*0.200
        #Make sure lanes didn't change much from previous frame (20% tolerance)
        # if abs(left_curverad_t - avg_curve) > tolerance:
        #     #left.frames_dropped+=1
        #     #if did, check if can use other lane as a rough estimate
        #     if abs(left_curverad_t - avg_curve) < (right_curverad_t - avg_curve) or (right_curverad_t - avg_curve)  >tolerance:
        #         l_fit_f = None;
        #     else:
        #         #If it's a reasonable match, use right lane as estimate, with start point as previous average was
        #         l_fit_f[0:2] = right_fit_t[0:2];
        #         l_fit_f[2] = left.get()[2]
        #
        # #Same process for Right lane
        # if abs(right_curverad_t - right.curve_r) > tolerance:
        #     if  abs(right_curverad_t - right.curve_r) < abs(left_curverad_t - avg_curve) or abs(left_curverad_t - avg_curve) >tolerance:
        #         r_fit_f = None;
        #     else:
        #         r_fit_f[0:2] = left_fit_t[0:2];
        #         r_fit_f[2] = right.get()[2]

        #If frame was still not dropped, check if base moved from original estimate by a tolerance pixels

        # if not (l_fit_f is None):

        leftx_base =left_fit[0]*binary_warped.shape[0]**2 + left_fit[1]*binary_warped.shape[0] + left_fit[2]
        rightx_base = right_fit[0]*binary_warped.shape[0]**2 + right_fit[1]*binary_warped.shape[0] + right_fit[2]
        l_base = l_fit_f[0]*binary_warped.shape[0]**2 + l_fit_f[1]*binary_warped.shape[0] + l_fit_f[2]
        ltip = l_fit_f[2]
        if abs(leftx_base - l_base) > 30 or abs(ltip-left.get()[2]) > 30:
            l_fit_f = None
        # if not (r_fit_f is None):
        r_base = r_fit_f[0]*binary_warped.shape[0]**2 + r_fit_f[1]*binary_warped.shape[0] + r_fit_f[2]
        rtip = r_fit_f[2]
        if abs(rightx_base - r_base) > 30 or abs(rtip - right.get()[2])>30:
            r_fit_f = None
        if abs((ltip + rtip) - (left_fit[2] + right_fit[2])) > 30 and not l_fit_f is None and not r_fit_f is None:
            l_fit_f = None
            r_fit_f = None
        #If frame was dropped (even if used other lane to estimate), increase frame drop
        if not np.array_equal(left_fit_t,l_fit_f):
            left_fit_t = l_fit_f;
            left.frames_dropped+=1;
        else:
            left.frames_dropped=0
        if not np.array_equal(right_fit_t ,r_fit_f):
            right_fit_t = r_fit_f;
            right.frames_dropped+=1
        else:
            right.frames_dropped=0;

        #if there's something to add to the line
        if not left_fit_t is None:
            left.add(left_fit_t)
            left_w.add(np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2))
        if not right_fit_t is None:
            right.add(right_fit_t)
            right_w.add(np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2))


    left_fit = left.get()
    right_fit = right.get()

    y_eval = np.max(ploty)
    left.curve_r = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right.curve_r = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    left_fit_cr = left_w.get()
    right_fit_cr = right_w.get()
    l_base = left_fit[0]*binary_warped.shape[0]**2 + left_fit[1]*binary_warped.shape[0] + left_fit[2]
    r_base = right_fit[0]*binary_warped.shape[0]**2 + right_fit[1]*binary_warped.shape[0] + right_fit[2]
    car_center = (binary_warped.shape[1]/2 - (l_base + r_base)/2)*xm_per_pix
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    left_w.curve = left_curverad
    right_w.curve = right_curverad
    avg_curvature = (left_curverad + right_curverad)/2
    avg_curve = (left.curve_r + right.curve_r)/2
    print( left_curverad, right_curverad, avg_curvature,car_center)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    # Create an image to draw on and an image to show the selection window
    out_img = warped
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-10, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+10, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-10, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+10, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Color in lane
    lane_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    lane_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    lane_pts = np.hstack((lane_window1, lane_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([lane_pts]), (0,255,0 ))
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,0, 255))

    out = utils.perspectiveInverse(window_img)

    out_img = cv2.addWeighted(undist, 1, out, 0.3, 0)


    # Recast the x and y points into usable format for cv2.fillPoly()
    curv = avg_curvature
    if curv < 2500:
        cv2.putText(out_img, 'Estimated lane curvature: {0:.2f}'.format(curv), (10, 30), 0, 1, (255,255,255), 2)
    else:
        cv2.putText(out_img, 'Estimated lane curvature: {}'.format('inf'), (10, 30), 0, 1, (255,255,255), 2)
    if(car_center < 0):
        cv2.putText(out_img, 'Car is {0:.2f}m to the left'.format(abs(car_center)), (10, 60), 0, 1, (255,255,255), 2)
    elif(car_center > 0):
        cv2.putText(out_img, 'Car is {0:.2f}m to the right'.format(abs(car_center)), (10, 60), 0, 1, (255,255,255), 2)
    else:
        cv2.putText(out_img, 'Car is centered'.format(abs(car_center)), (10, 60), 0, 1, (255,255,255), 2)

    cv2.imshow('result', out_img)
    videoout.write(out_img)



    if(left.frames_dropped >= 5):
        left.reset()


    if ( right.frames_dropped >= 5):
        right.reset()


    #plt.pause(0.001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
videoout.release()
