import cython 
import numpy as np
import cv2

def calculate_block_mean_image(image):
    
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]
    
    out = np.zeros((int(h/2),int(w/2),1), np.intc)
    
    # loop over the image, pixel by pixel
    for x in range(0, int(w/2)-1):
        for y in range(0, int(h/2)-1):
            new_y = np.mean([image[2*y, 2*x][0], image[2*y+1, 2*x][0], image[2*y, 2*x+1][0], image[2*y+1, 2*x+1][0]])
            out[y, x] = int(new_y)
                
    
    return out

def calculate_block_diff(bi, bg):
    out = np.abs(np.subtract(bg, bi))
    return out


def calculate_block_th(bd, th):
    return (bd > th)
    
def calculate_and(bb_old, bb_new):
    old_bool = np.array(bb_old, dtype=bool)
    new_bool = np.array(bb_new, dtype=bool)
    bb_sum = np.logical_and(old_bool, new_bool)
    bb_sum = bb_sum.astype(int)
    return bb_sum

def draw_blobs(image, bb):
    h = bb.shape[0]
    w = bb.shape[1]
    
    for x in range(w):
        for y in range(h):
            if (bb[y, x] == 1):
                image = cv2.rectangle(image, (2*x-1, 2*y-1), (2*x+3, 2*y+3), (0, 0, 255), 2)
                
    return image

def threshold_img(bb):
    h = bb.shape[0]
    w = bb.shape[1]
    
    out = np.zeros((int(h*2),int(w*2),3), np.uint8)
    
    for x in range(w):
        for y in range(h):
            if (bb[y, x] == 1):
                out[2*y, 2*x] = (255, 255, 255)
                out[2*y+1, 2*x] = (255, 255, 255)
                out[2*y, 2*x+1] = (255, 255, 255)
                out[2*y+1, 2*x+1] = (255, 255, 255)
                
    return out