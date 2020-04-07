import cv2

print(cv2.__version__)

colorPic = "D:\\coding\\python\\bw-colorization\\images\\" + input("enter your file name: ")
ss = colorPic.replace( "\ " , "\\")
pic=cv2.imread (colorPic,0)

findString = "."
inserttxt = "_bw"

idx = ss.index(findString)
bwPic = ss[:idx] + inserttxt + ss[idx:]

cv2.imwrite(bwPic,pic)
cv2.imshow("racoon",pic)
cv2.waitKey(5000)