#2.1
#Adding step to reverse threshold image, removing brightest spots
#done before colvolve step
#Doesnt work, strange convolves from the now removes small pixels

import numpy as np
import pandas as pd
import skimage
import os
import glob
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from shutil import copyfile
from os import listdir
from os.path import isfile, join
from scipy import ndimage
from scipy.misc import bytescale
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import erosion, dilation
from skimage import measure, exposure, morphology, img_as_uint, img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from skimage.filters import *
from skimage.io import imsave
from skimage.morphology import disk, watershed, remove_small_objects
from skimage.segmentation import *
from skimage.draw import circle
from scipy import ndimage

### Instructions
#Tweakable values - ensure they are all odd for reproduceability.
imagej_channel = 2      #which channel do you want to open?
v_close = 9     #default is 9, suggested  7-11.             -closing, removes most non-horizontal objects & those smaller than v_close across.  
v_open1 = 99    #default is 99, suggested 75-111           -opening, the larger the value the more likely strings will aglomerate together.
v_open2 = 85    # default is 75, ~24-12 less than v_open1. -opening, this value reduces the size of afformentioned agglomerates to real sizes. i.e. we normalise the strings back to real values 
imagej_exec = "C:\Code\FIJI\ImageJ-win64.exe" #put the location of your imagej programme here.
###-----end


### defining function 
def calculateFeret(coordinates):
    feret = np.nanmax(squareform(pdist(coordinates)))
    feret = feret + (((2*((0.5)**2))**(0.5))*2)
    return feret

def scale8bit(image):
    scale = float(256) / (image.max() - image.min())
    return np.clip(np.round(np.multiply(image, scale)), 0, 255).astype(np.uint8)

def stringSegmentation(String_image):   
    ###Closing -removes non string-like objects
    String_ero = skimage.morphology.erosion(String_image, np.ones((1,v_close)))
    String_dil = skimage.morphology.dilation(String_ero,  np.ones((1,v_close)))
    #opening - returns string-like objects to their actual values 75 better than 100
    String_dil = skimage.morphology.dilation(String_dil,  np.ones((3,v_open1)))
    String_ero = skimage.morphology.erosion(String_dil, np.ones((1,v_open2)))
    String_temp = String_ero
    ###segment and label 
    #protocol from http://scikit-image.org/docs/0.11.x/user_guide/tutorial_segmentation.html
    ###code to remove edge artifacts
    temp_height = String_temp.shape[0]
    temp_width = String_temp.shape[1]
    String_temp[0:45,:] = 0 #y plane
    String_temp[temp_height -46:temp_height-1,:] = 0 #y plane
    String_temp[:,0:10] = 0 #x plane
    String_temp[:,temp_width -11:temp_width-1] = 0 #x plane   
    ###deal with strings
    label_objects, nb_labels = ndimage.label(String_temp)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 108 #remove small objects. takes list of labelles size objects
    mask_sizes[np.where(sizes > 1000)] = False #remove artifact largest strings. guestimate from ~5 images [altered from 1500]
    String_cleaned = mask_sizes[label_objects]
    #segmented boolean image
    String_seg, _ = ndimage.label(String_cleaned)
    imsave(file_path+"/04_python_images/"+image_name.split('.',1)[0]+'.png', String_seg)
    cell_features = measureMorphometry(String_seg)
    if os.path.isfile(file_path+"/05_python_data/"+image_name.split('.',1)[0]+'_data.csv')  == True:
        os.remove(file_path+"/05_python_data/"+image_name.split('.',1)[0]+'_data.csv')
    cell_features.to_csv(file_path+"/05_python_data/"+image_name.split('.',1)[0]+'_data.csv', sep=',', mode='a', header=True, index=False)
    return (String_seg, cell_features)

#blob is a boolean, blob_label is labelles segmented objects, image name is in loop
def measureMorphometry(label_image):
    properties = measure.regionprops(label_image)
    properties_boundary = measure.regionprops(find_boundaries(label_image, mode='thick')*label_image)
    y_centroid = pd.Series([i[0] for i in [prop.centroid for prop in properties]]) * pixel_dimension
    x_centroid = pd.Series([i[1] for i in [prop.centroid for prop in properties]]) * pixel_dimension
    area = pd.Series([prop.area for prop in properties]) * pow(pixel_dimension, 2)
    perimeter = pd.Series([prop.perimeter for prop in properties]) * pixel_dimension
    feret = pd.Series([calculateFeret(prop.coords) for prop in properties_boundary]) * pixel_dimension
    equivalent_diameter = pd.Series([prop.equivalent_diameter for prop in properties]) * pixel_dimension
    convex_area = pd.Series([prop.convex_area for prop in properties]) * pow(pixel_dimension, 2)
    major_axis_length = pd.Series([prop.major_axis_length for prop in properties]) * pixel_dimension
    minor_axis_length = pd.Series([prop.minor_axis_length for prop in properties]) * pixel_dimension
    orientation = pd.Series([prop.orientation for prop in properties])
    solidity = pd.Series([prop.solidity for prop in properties])
    #max_intensity = pd.Series([prop.max_intensity for prop in properties])
    #min_intensity = pd.Series([prop.min_intensity for prop in properties])
    #mean_intensity = pd.Series([prop.mean_intensity for prop in properties])
    particle_id = pd.DataFrame(range(1,label_image.max(),1))
    particles_image = pd.concat([particle_id,x_centroid,y_centroid,area,perimeter,feret,equivalent_diameter,convex_area,major_axis_length,minor_axis_length,orientation,solidity],axis=1)
    particles_image.columns = ['particle_id', 'x_centroid', 'y_centroid','area','perimeter','feret','equivalent_diameter','convex_area','major_axis_length','minor_axis_length','orientation','solidity']
    return(particles_image)


def imagePreProcess(macro_file, image_list, imagej_exec, headless):
    #colvolve or clean+colvolve image list. updates the Fiji macro file with a new input and output files, creates a temporary macro file
    #for each image in image_list, generate a macro file, run it.
    channel = str(imagej_channel + 1) #one image will already be open, then channel1, channel2 etc..
    for pre_image in image_list:
        pre_image = pre_image
        pre_image_no_ext = os.path.splitext(os.path.basename(pre_image))[0]
        pre_image = pre_image.replace(".", dname,1) #imagej requires full path
        pre_image = pre_image.replace("/", "\\")
        pre_image = pre_image.replace("\\", "\\\\")
        pre_image = "\"" + pre_image + "\""
        #pre_image_no_ext = os.path.splitext(os.path.basename(dirlist[1]))[0]
        clean_imagename = dname + "\\02_TIF_tiled\\" + pre_image_no_ext + ".tif"
        clean_imagename = clean_imagename.replace("\\", "\\\\")
        clean_imagename = "\"" + clean_imagename + "\""
        convolved_imagename = dname + "\\03_TIF_convolved\\" + pre_image_no_ext + ".tif"
        convolved_imagename  = convolved_imagename .replace("\\", "\\\\")
        convolved_imagename = "\"" + convolved_imagename + "\""
        #create temporary custom macro file to run
        temp = "macro_temp.ijm" #create temp macro file
        sample1=''
        sample2=''
        sample3=''
        with open(macro_file, 'r')as f:
            sample1 = f.read().replace("[01]", pre_image, 1)
            sample2 = sample1.replace("[02]", clean_imagename, 1)
            sample3 = sample2.replace("[03]", convolved_imagename, 1)
            sample3 = sample3.replace("[channel]", channel, 1)
            sample3 = sample3.replace('@"', '', 2) #bodge
            sample3 = sample3.replace('"@', '', 2) #bodge
        with open(temp, 'w')as f:    
            f.write(sample3)
        #carry out imagej process  ---example - <ImageJ --headless -macro path-to-Macro.ijm>
        temp_full = dname + "\\" +temp
        temp_full  = temp_full.replace("\\", "\\\\")
        if(headless == True):
            command = imagej_exec + " --headless -macro " + temp_full
            os.system(command)
        else:
            command = imagej_exec + " -macro " + temp_full
            os.system(command)
            #os.system("TASKKILL /F /IM ImageJ-win64.exe /T") #close program after each process
        #os.remove(temp)
    return("")

#set the working directory to the script directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#other small things to declare
file_path = ".\\"
start_time = datetime.now()
pixel_dimension = 0.4807637 #micro meter

### Things happen
#decide macro and images to use with this logic [if # files in folder, make assumptions]
if len(glob.glob('./03_TIF_convolved/*')) > 0:  #no actions needed - move onto image analysis
    print("...no images identified for imageJ processing")
elif len(glob.glob('./02_TIF_tiled/*')) > 0: #images to convolve only
    image_list = glob.glob('./02_TIF_tiled/*')
    #2.1 added step
    #later - oscar create a t/f mask for original image like this and apply to the v2 initial string image
    for temp_file in image_list:
        temp_im = skimage.io.imread(temp_file, plugin='tifffile')
        temp_im = scale8bit(temp_im)
        temp_im[temp_im > 100] = 0
        skimage.io.imsave(temp_file, temp_im, plugin='tifffile')
    #2.1 done
    
    image_macro = "macros\\Fiji_macro_convolve.ijm"
    print("...cleaned images found ... sending to imagej")
    print(image_list)
    del_1 = imagePreProcess(image_macro, image_list, imagej_exec, False) #doesnt work headless :()
    print("...imagej convolve macro complete")
elif len(glob.glob('./01_image_original/*/*.mvd2')) > 0: #images to extract and convolve
    image_list = glob.glob('./01_image_original/*/*.mvd2')
    image_macro = macro_file = "macros\\Fiji_macro_extract-convolve.ijm"
    print("...original images found ... sending  to imagej")
    print(image_list)
    del_1 = imagePreProcess(image_macro, image_list, imagej_exec, False)
    print("...imagej extract-convolve macro complete")
else:
    print("...error no images detected!")

#main loop
print("...analysing images")
image_list = set([f for f in listdir(file_path + "\\03_TIF_convolved") if isfile(join(file_path + "\\03_TIF_convolved",f))]) #images are now all processed
#experiment_name  = os.path.basename(os.path.dirname(os.path.dirname(os.getcwd())))
number_of_images = len(image_list)
for image in range(0, number_of_images):
    image_name = list(image_list)[image]
    tiffStack = skimage.io.imread(file_path+"03_TIF_convolved\\"+image_name, plugin='tifffile')
    String_image = tiffStack
    String_output_image, String_output_data = stringSegmentation(String_image)
    
    print "Analysing {0}, image {1} of {2}, detected:".format(image_name, image+1, number_of_images)
    print "    {0} strings ".format(len(String_output_data))


print '\nAnalysis time: ', datetime.now() - start_time, ' seconds'