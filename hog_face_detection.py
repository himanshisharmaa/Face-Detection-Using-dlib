from pyimg.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2

# --upsample: Number of times to upsample 
# an image before applying face detection.
# The downside to upsampling is that it 
# creates more layers of our image pyramid,
#  making the detection process slower.

# For faster face detection, set the 
# --upsample value to 0, meaning that 
# no upsampling is performed (but you 
# risk missing face detections).
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())

#load dlib's HOG+Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")
detector=dlib.get_frontal_face_detector()

# load the input from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
image=cv2.imread(args["image"])
image=imutils.resize(image,width=600)
rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


#perform face detection using dlib's face detector
start=time.time()
print("[INFO] performing face detection with dlib")
rects=detector(rgb,args['upsample'])
end=time.time()
print(f"[INFO] face detection took {(end-start):.4f} seconds")


# convert the resulting dlib rectangle objects to bounding boxes,
# then ensure the bounding boxes are all within the bounds of the
# input image

boxes=[convert_and_trim_bb(image,r) for r in rects]

for (x,y,w,h) in boxes:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Output",image)
cv2.waitKey(0)

