# build mask dataset for gan
# just run it with out any argv
# note that you need to change the input and output path

import cv2
import dlib
import numpy

import sys
import os


def get_landmarks(im):
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])



def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
    

def get_face_mask(im, landmarks):
    # FEATHER_AMOUNT = 11
    
    # LEFT_EYE_POINTS = list(range(42, 48))
    # RIGHT_EYE_POINTS = list(range(36, 42))
    # LEFT_BROW_POINTS = list(range(22, 27))
    # RIGHT_BROW_POINTS = list(range(17, 22))
    # NOSE_POINTS = list(range(27, 35))
    # MOUTH_POINTS = list(range(48, 61))
    
    # Yunqiu Xu changed on 20171008
    # OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]

    mask = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    
    draw_convex_hull(mask,landmarks[48:61],color=1)

    # Yunqiu Xu
    cv2.imshow("convex hull", mask)

    xs=[]
    ys=[]

    # Yunqiu Xu 68 --> 48, 61
    for ind in range(48, 61):
        xs.append(landmarks[ind,0])
        ys.append(landmarks[ind,1])

    xmin,xmax,ymin,ymax = min(xs),max(xs),min(ys),max(ys)
    
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if(mask[y][x]!=1):
                im[y][x][0] = 0
                im[y][x][1] = 0
                im[y][x][2] = 0
    
    return im

def read_im_and_landmarks(fname):
    SCALE_FACTOR = 1
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    # im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def main():
    input_dir = "/home/venturer/COMP9517/generate_gan_dataset/img_dataset/"
    output_dir = "/home/venturer/COMP9517/generate_gan_dataset/mask/"
    for img_name in os.listdir(input_dir):
        print(img_name)
        im, landmarks = read_im_and_landmarks(input_dir+img_name)
        mask = get_face_mask (im,landmarks)
        print "mask shape : " + str(mask.shape)
        print("done")


        cv2.imshow("mask", mask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        # cv2.imwrite(output_dir+img_name, mask)

    print("All done")
    cv2.destroyAllWindows()

main()
