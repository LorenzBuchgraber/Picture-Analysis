import cv2
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import color, measure
from scipy import ndimage as nd
from skimage.color import label2rgb
from skimage.filters import threshold_sauvola
from skimage.measure import label
from scipy.ndimage import label
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max

#ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import the picture for image processing
def readpicture (path, lower_x, upper_x):    
    """
    Returns cropped picture in grayscale for the analysis

    Parameter
    ---------
    path : str
        path to the pictures for the analysis
    lower_x : int
        lower boundary for the cropping
    upper_x : int
        upper boundary for the cropping
    Returns
    -------
    ndarray
        grayscale image
    """              
    img = cv2.imread(path)
    img = cv2.resize(img,(3840,2160))
    #crop the image/delete boundary areas which are not needed
    cropped = img[0:2160,lower_x:upper_x]   
    #turns colored image into grayscale                
    gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)         
    return gray

def thresholding (gray_scale, window_size, additional_thresh):
    """
    Returns thresholded and labeled image for the measuremnts

    Parameter
    ---------
    gray_scale : ndarry
        gray scale image which should be thresholded and labeled
    window_size : int
        windowsize for the local thresholding, could be in- or decreased for better thresholding
    additional_thresh : float
        value for a better separation between background and crystal, can be varied for better results

    Returns
    -------
    ndarray
        labeled image for the measurment
    """   
    #defines size of the local thresholding area. define not too small! - no differences in gray level can be detected anymore
    window_size = 155  
    #local thresholding according to Sauvola                                                 
    thresh = threshold_sauvola(gray_scale, window_size = window_size)   
    #converts gray scale image into binary image - additional term to
    binary = gray_scale > thresh + additional_thresh      
    #deletes every crystal which touches the border of the picture, so just full particles are measured               
    cleared = clear_border(binary)
    #labeling the detected particles
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]                               
    labeled, num_labels = label(cleared, structure=s)          #type: ignore
    return labeled


#reading of the picture to be analyzed
#defines path to the picture which should be measured
path = "test/*.*"   
#cut the edges of the picture which are not representative                     
lower_x = 800                                               
upper_x = 3240

#reading of the picture for the pixels to micron calculation/cut size must be the same!
#defines path to the picture which should be measured
path_tri = "calibration.tiff"                

#convert pictures into gray scale
gray_tri = readpicture(path_tri, lower_x, upper_x)

#define windowsize for the local thresholding and the additional value for fine tuning of the threshold
window_size = 155
additional_thresh_tri = 40

#thresholding of the picture to be analyzed
additional_thresh = 5.9

#actual thresholding of the two pictures
labeled_tri = thresholding (gray_tri, window_size, additional_thresh_tri)

#measuring for the pixels to micron size out of a picture without crystalls
props_tri = measure.regionprops_table(labeled_tri, gray_tri,
                                  properties = ['label', 
                                                'feret_diameter_max'])

dataframe_tri = pd.DataFrame(props_tri)

dataframe_tri = dataframe_tri[dataframe_tri['feret_diameter_max'] > 50]
mean_feret_tri = dataframe_tri['feret_diameter_max'].mean()
std_feret_tri = dataframe_tri['feret_diameter_max'].std()
dataframe_tri = dataframe_tri[dataframe_tri['feret_diameter_max'] > mean_feret_tri - 0.5*std_feret_tri]
dataframe_tri = dataframe_tri[dataframe_tri['feret_diameter_max'] < mean_feret_tri + 0.5*std_feret_tri]
mean_pixel_tri = dataframe_tri['feret_diameter_max'].mean()
#size of the triangle in um detected according to the 3D CAD file (skizze_triangle)
length_tri_um = 1790                                                                                    

#define pixel size for measurements/automated
pixels_to_um = length_tri_um/mean_pixel_tri

#start for loop
props_table = pd.DataFrame()
for file in glob.glob(path):
    gray = readpicture(file, lower_x, upper_x)
    #plt.imsave("gray.png", gray, cmap='gray')

    #subtracting the two pictures/to exclude the flow breaker
    subt = cv2.subtract(gray, gray_tri)
    labeled = thresholding (subt, window_size, additional_thresh)
    #plt.imsave("binary.png", labeled, cmap='gray')
    
    #segmentation verbessern!
    D = nd.distance_transform_edt(labeled)
    localMax = peak_local_max(D, indices=False, min_distance=7, labels=labeled)
    markers = nd.label(localMax, structure=np.ones((3, 3)))[0]                    #type: ignore
    labels = watershed(-D, markers, mask=labeled)                                 #type: ignore
    
    #colorize the thresholded picture to see the crystalls in different colors
    #img_col = color.label2rgb(labeled, bg_label=0)

    #overlay the thresholded picture over the original gray scale image to see if most of the particles can be detected and no areas are fals detected
    image_label_overlay = label2rgb(labels, image = gray)
    #plt.imsave("label_overlay.png", image_label_overlay)
    

#measurement/measure all detected crystalls and delete some of the noise
    props = pd.DataFrame(measure.regionprops_table(labels, gray,
                                      properties = ['label', 'area', 
                                                   'equivalent_diameter_area', 
                                                   'axis_minor_length']))
    props_table = pd.concat([props_table, props])

dataframe = pd.DataFrame(props_table)
    
#delete crystalls with area smaller than 3 pixels
dataframe = dataframe[dataframe['area'] > 3]
dataframe = dataframe[dataframe['axis_minor_length'] > 0]

#convert pixels to microns
dataframe['area_sq_microns'] = dataframe['area'] * (pixels_to_um**2)
dataframe['equivalent_diameter_area_microns'] = dataframe['equivalent_diameter_area'] * (pixels_to_um)
dataframe['axis_minor_length_microns'] = dataframe['axis_minor_length'] * (pixels_to_um)
dataframe = dataframe[dataframe['equivalent_diameter_area_microns'] < 700]
print(dataframe.head())

#calculations with the data
mean_feret = dataframe['equivalent_diameter_area_microns'].mean()
std_feret = dataframe['equivalent_diameter_area_microns'].std()

#calculation of the particle size distribution data (all operations are done for the whole array elementwise)
#bins are the size classes in which the particlepopulation is classified
count, division = np.histogram(dataframe['equivalent_diameter_area_microns'], bins = 25)   
#size width in µm         
delta_x = np.diff(division) 
#first entry is removed, just upper boundaries are used from the size classes                                                                        
division = np.delete(division, 0) 
#total amount of particles                                                                  
n_total = len(dataframe.index) 
#percentage of particles in each size class                                                                     
dQ0 = np.divide(count, n_total)   
#just for checking if sum of all dQ is 1 (has to be)                                                                  
#check = np.sum(dQ0) 
#calculates the cumulative sum of dQ0                                                                               
Q0 = np.cumsum(dQ0)      
#q0 distribution                                                                           
q0 = np.divide(dQ0, delta_x)                                                                        

#half of class width for calculation of x_mean
dx = np.divide(delta_x, 2)
#x_mean = x_o-dx/2 value in the middle of the size class boundaries                                                                          
x_mean = np.subtract(division, dx)                                                                  

#conversion of q0 distribution to q3 distribution
x_m_3_dQ0 = np.multiply(np.power(x_mean, 3), dQ0)
xm3_Q0 = np.sum(x_m_3_dQ0)
dQ3 = np.divide(x_m_3_dQ0, xm3_Q0)
Q3 = np.cumsum(dQ3)
q3 = np.divide(dQ3, delta_x)

#save the important results into an excel sheet
df = pd.DataFrame()
df['x_m'] = x_mean
df['q3'] = q3
df['Q3'] = Q3
df.to_excel(excel_writer = "data.xlsx")

#plot in python if necessary
#x = np.array(x_mean)
#y = np.array(Q3)

#plt.title("Q3 plot")
#plt.plot(x, y, color="red")

#plt.show()