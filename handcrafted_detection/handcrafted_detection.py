import numpy as np
import cv2
import pyximport; pyximport.install()
from handcrafted_utils import *

if (__name__ == "__main__"):
    cap = cv2.VideoCapture('../../Training_Data/video_train_set/train-10.avi')
    ind = 0
    while (True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        #frame = cv2.putText(frame, str(ind), (50,250), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 5, cv2.LINE_AA)
        BI = calculate_block_mean_image(yuv_frame)
        th = 4
        if (ind == 0):
            BG = BI
            BD = calculate_block_diff(BI, BG)
            BB = calculate_block_th(BD, th)
            final_out = threshold_img(BB)
            last_BB = BB
        else:
            BD = calculate_block_diff(BI, BG)
            BB = calculate_block_th(BD, th)
            if (ind > 1):
                #BB = calculate_and(last_BB, BB)
                BB = BB


        final_out = threshold_img(BB)
        gray_out = cv2.cvtColor(final_out, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray_out, 1, 255,
            cv2.THRESH_BINARY_INV)[1] 
        threshold = cv2.bitwise_not(threshold)
        # Creating kernel
        kernel = np.ones((5, 5), np.uint8)
        # Using cv2.erode() method 
        threshold = cv2.erode(threshold, kernel, iterations = 1) 

        kernel = np.ones((5, 5), np.uint8)
        #threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)
        threshold = cv2.dilate(threshold, kernel, iterations=5)
        #frame = draw_blobs(frame, BB)
        contours, _ = cv2.findContours(threshold,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largestContour = np.array([[]])
        # create hull array for convex hull points
        hull = []
        
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        if len(contours) > 0:
            largestContour = max(contours, key=cv2.contourArea)
            #cv2.drawContours(frame, contours, -1, 255, 2)
            for i in range(len(contours)):
                cv2.drawContours(frame, hull, i, (0, 255, 0), 1, 8)
            # get the unrotated bounding box that surrounds the contour
            x,y,w,h = cv2.boundingRect(largestContour)

            # draw the unrotated bounding box
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

            last_BB = BB
        frame = cv2.putText(frame, "Frame: " + str(ind), (25,500), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
        cv2.imshow('video', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

        ind = ind + 1

    #cam.release()
    cv2.destroyAllWindows()
