import cython 
import numpy as np
import cv2
import time

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, point):
        return Point(self.x + point.x, self.y + point.y)
    def __sub__(self, point):
        return Point(self.x - point.x, self.y - point.y)
    def cross(self, point):
        return (self.x * point.y, self.y * point.x)

def invert_polygon(points):
    out = []
    for point in points:
        out.append(Point(-point.x, -point.y))
    return out

def reorder_polygon(points):
    pos = 0
    for i in range(1, len(points)):
        if (points[i].y < points[pos].y or (points[i].y == points[pos].y and points[i].x < points[pos].x)):
            pos = i
    return np.roll(points, -1*pos)

def minkowski(polygon_a, polygon_b):
    polygon_b = invert_polygon(polygon_b)
    polygon_a = reorder_polygon(polygon_a)
    polygon_b = reorder_polygon(polygon_b)
    polygon_a.append(polygon_a[0])
    polygon_a.append(polygon_a[1])
    polygon_b.append(polygon_b[0])
    polygon_b.append(polygon_b[1])

    out = []
    i = 0
    j = 0
    while (i < polygon_a.size()-2 or j < polygon_b.size()-2):
        out.append(polygon_a[i] + polygon_b[j])
        cross = (polygon_a[i + 1] - polygon_a[i]).cross(polygon_b[j + 1] - polygon_b[j])
        if (cross >= 0):
            i = i + 1
        if (cross <= 0):
            j = j + 1
    return out

def calculate_block_mean_image(image):
    start = time.perf_counter()
    
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    out = cv2.resize(image, (int(w/2),int(h/2)))[:,:,0]
    out = out.astype(np.intc)
    
    """"
    out = np.zeros((int(h/2),int(w/2),1), np.intc)
    
    # loop over the image, pixel by pixel
    for x in range(0, int(w/2)-1):
        for y in range(0, int(h/2)-1):
            new_y = np.mean([image[2*y, 2*x][0], image[2*y+1, 2*x][0], image[2*y, 2*x+1][0], image[2*y+1, 2*x+1][0]])
            out[y, x] = int(new_y)
            """
                
    print("Block Mean Img.: " + str(round(start-time.perf_counter(), 2)))
    
    return out

def calculate_block_diff(bi, bg):
    start = time.perf_counter()
    out = np.abs(np.subtract(bg, bi))
    print("Block diff.: " + str(round(start-time.perf_counter(), 2)))
    return out


def calculate_block_th(bd, th):
    start = time.perf_counter()
    print("Block th.: " + str(round(start-time.perf_counter(), 2)))
    return (bd > th)
    
def calculate_and(bb_old, bb_new):
    start = time.perf_counter()
    old_bool = np.array(bb_old, dtype=bool)
    new_bool = np.array(bb_new, dtype=bool)
    bb_sum = np.logical_and(old_bool, new_bool)
    bb_sum = bb_sum.astype(int)
    print("Old new and: " + str(round(start-time.perf_counter(), 2)))
    return bb_sum

def draw_blobs(image, bb):
    start = time.perf_counter()
    h = bb.shape[0]
    w = bb.shape[1]
    
    for x in range(w):
        for y in range(h):
            if (bb[y, x] == 1):
                image = cv2.rectangle(image, (2*x-1, 2*y-1), (2*x+3, 2*y+3), (0, 0, 255), 2)
                
    print("Blobs: " + str(round(start-time.perf_counter(), 2)))
    return image

def threshold_img(bb):
    start = time.perf_counter()
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
                
    print("Threshold img.: " + str(round(start-time.perf_counter(), 2)))

    return out