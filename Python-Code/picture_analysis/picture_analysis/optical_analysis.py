import cv2
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import os

#Connect to Webcam
cap = cv2.VideoCapture(1)
#video = VideoWriter('webcam.avi', VideoWriter_fourcc(*'MP42'), 30.0 , (3840, 2160))

#Setup Directory for frames
try:
    if not os.path.exists("videoframes"):
        os.makedirs("videoframes")
        
except OSError:
    print("Error: Creating Directory of Data")

currentframe = 0
cap.set(3, 3840)
cap.set(4, 2160)
cap.set(cv2.CAP_PROP_FPS, 30.0)

lastframe = 100 
#cv2.destroyAllWindows()     
#cap.release()
#video.release()
    
currentframe = 0
#statt webcam.avi
while True:
    success, image = cap.read()
    name = "./videoframes/frame" + str(currentframe) + ".tiff"
    cv2.imwrite(name, image)
    currentframe +=1
    # cv2.waitKey(1) & falls fenster erwünscht
    if currentframe == lastframe: 
        break
            
cap.release()


