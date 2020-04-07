# python decolorization2.py --image images/red-and-white-raccoon.jpg
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="Path to image")
args = vars(ap.parse_args())


colourimage = cv2.imread(args["image"])
grayimage = cv2.cvtColor(colourimage, cv2.COLOR_BGR2GRAY)
  
key = cv2.waitKey(1) & 0xFF
# if the `q` key was pressed, break from the loop



if key == ord("y"):



imagesave = input("Do you want to save this image? y/n: ")
if (imagesave == "y"):

	findString = "."
	inserttxt = "_bw"

	bwfile = args["image"]
	idx = bwfile.index(findString)
	bwfile = bwfile[:idx] + inserttxt + bwfile[idx:]
	#input(bwfile)
	cv2.imwrite(bwfile,grayimage)

cv2.waitKey(0)
cv2.destroyAllWindows()