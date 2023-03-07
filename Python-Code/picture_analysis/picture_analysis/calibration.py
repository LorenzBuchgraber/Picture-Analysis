import cv2

#connect to camera

def take_calibration_photo():
    """
    Returns picture for the calibration, cell has to be free from crystals

    Parameter
    ---------
    none
    
    Returns
    -------
    calibration.tiff
        calibration image in the directory
    """
    cap = cv2.VideoCapture(1)   
    #connect to camera                
    cap.set(3,3840)
    cap.set(4,2160)
    ret, frame = cap.read()
    cv2.imwrite("calibration.tiff", frame)
    #deactivates camera
    cap.release()                               
    
take_calibration_photo()