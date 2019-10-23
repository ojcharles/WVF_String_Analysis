#last edit, 19/07/2019
#most false posotivies are blobs, some are erronious from convolve
#   1. score matrix for string like - erode_dilate
#   2. score matrix of blob like
#   3. score matrix of combined, removes any leftover small objects

#todo
#update scoring system from erode dilate
#make variable library to clean up variable explorer

script_name = "2_2_2"

import numpy as np #oscar - clean up imports
import pandas as pd
import skimage
import os
import glob
import warnings
import scipy.misc
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
#   v_close = 9     #default is 9, suggested  7-11.             -closing, removes most non-horizontal objects & those smaller than v_close across.  
#   v_open1 = 99    #default is 99, suggested 75-111           -opening, the larger the value the more likely strings will aglomerate together.
#   v_open2 = 85    # default is 75, ~24-12 less than v_open1. -opening, this value reduces the size of afformentioned agglomerates to real sizes. i.e. we normalise the strings back to real values 
imagej_exec = "C:/Code/FIJI/ImageJ-win64.exe" #put the location of your imagej programme here.
###-----end

### defining function 
def calculateFeret(coordinates):
    feret = np.nanmax(squareform(pdist(coordinates)))
    feret = feret + (((2*((0.5)**2))**(0.5))*2)
    return feret

def scale8bit(image):
    #scales an image between 0-255
    scale = float(256) / (image.max() - image.min())
    return np.clip(np.round(np.multiply(image, scale)), 0, 255).astype(np.uint8)

def removeBorder(image):
    #removes edge pixels as convolving doesnt cope well, simple but effective
    im_height = image.shape[0]
    im_width = image.shape[1]
    image[0:50,:] = 0 #y plane
    image[im_height -51:im_height-1,:] = 0 #y plane
    image[:,0:50] = 0 #x plane
    image[:,im_width -51:im_width-1] = 0 #x plane 
    return(image)

def getScore(image, overlay_h, overlay_w, cutoff = 0.7, score = 1):
    #overlay a shape over input image, if cutoff propoertion are 1 then set score matrix pixels in mask ++ 1
    #returns numbered matrix, where higher number measn higher chance of string
    im_height = image.shape[0]
    im_width = image.shape[1]
    overlay_area = overlay_h * overlay_w
    score_matrix = np.zeros((im_height, im_width))
    for w in range(overlay_w, im_width - overlay_w): #oscar slow nested loop
        for h in range(overlay_h, im_height - overlay_h):
            overlay = image[h : h + overlay_h , w : w + overlay_w]
            overlay_sum = overlay.sum()
            if(overlay_sum >= overlay_area * cutoff ):
                #score_matrix[h : h + overlay_h , w : w + overlay_w] = score_matrix[h : h + overlay_h , w : w + overlay_w] + score
                score_matrix[h : h + overlay_h , w : w + overlay_w] = 1
    return(score_matrix)     

def stringSegmentation(convolve_image, blob_image, image_name, file_path):
    #takes convolved and original image data, identifies the strings, returns labelled images and metrics as csv
    im_conv = removeBorder(convolve_image)
    im_blob = removeBorder(blob_image)
    im_blob = skimage.morphology.dilation(im_blob,  np.ones((3,3)))
    score_conv = getScore(im_conv, 2,8,0.9, 1)      #Overlay masks, create scores, create combined overall score score
    score_blob = getScore(im_blob, 4, 4, 0.8, 1)    # needs to act on non-conv image
    score_blob = skimage.morphology.dilation(score_blob, np.ones((5,5)))
    score_overall = score_conv - score_blob
    score_overall[score_overall < 0] = 0
    score_overall[score_overall >= 1] = 1
    ##segment and label protocol from http://scikit-image.org/docs/0.11.x/user_guide/tutorial_segmentation.html
    label_objects, nb_labels = ndimage.label(score_overall)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 108 #remove small objects. takes list of labelles size objects
    mask_sizes[np.where(sizes > 1000)] = False #remove artifact largest strings. guestimate from ~5 images [altered from 1500]
    strings_out = mask_sizes[label_objects] #segmented boolean image
    strings_out, t1 = ndimage.label(strings_out)
    #imsave(file_path+"/04_python_images/"+image_name.split('.',1)[0]+'.png', strings_out)
    
    scipy.misc.imsave('C:/Oscar/OneDrive/UCL/1-Cutler/6-Patella_Strings/strings_tool_2/test/im_blob.png', im_blob)
    scipy.misc.imsave('C:/Oscar/OneDrive/UCL/1-Cutler/6-Patella_Strings/strings_tool_2/test/strings_out.png', strings_out)
    scipy.misc.imsave('C:/Oscar/OneDrive/UCL/1-Cutler/6-Patella_Strings/strings_tool_2/test/score_conv.png', score_conv)
    scipy.misc.imsave('C:/Oscar/OneDrive/UCL/1-Cutler/6-Patella_Strings/strings_tool_2/test/score_blob.png', score_blob)
    #scipy.misc.imsave('C:/Oscar/OneDrive/UCL/1-Cutler/6-Patella_Strings/strings_tool_2/test/score_String.png', score_string)
    scipy.misc.imsave('C:/Oscar/OneDrive/UCL/1-Cutler/6-Patella_Strings/strings_tool_2/test/score_overall.png', score_overall)
    
    cell_features = measureMorphometry(strings_out)
    if os.path.isfile(file_path+"/05_python_data/"+image_name.split('.',1)[0]+'_data.csv')  == True:
        os.remove(file_path+"/05_python_data/"+image_name.split('.',1)[0]+'_data.csv')
    cell_features.to_csv(file_path+"/05_python_data/"+image_name.split('.',1)[0]+'_data.csv', sep=',', mode='a', header=True, index=False)
    return (strings_out, cell_features)

#blob is a boolean, blob_label is labelles segmented objects, image name is in loop
def measureMorphometry(label_image):
    #takes a matrix where wach joined object has unique number, returns metrics on object morphology
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

def alterConvolvedImage(image_file):
    #helper fundtion, may not have a use for real-world-use
    #alterations for v2 of macro, plays with the convolved image and returns an altered file of same name/dir
    im = skimage.io.imread(image_file, plugin='tifffile')
    im = scale8bit(im)
    #im_labeled = measure.label(im)
    #remove_small_objects(im, 3)
    
    skimage.io.imsave(image_file, im, plugin='tifffile')
    
def checkImagesexist(image_name):
    #v2 of macro requires convolved and tiled image of strings, this checks if both images are present
    if(len(glob.glob(file_path+"03_TIF_convolved\\"+image_name)) > 0):
        if(len(glob.glob(file_path+"02_TIF_tiled\\"+image_name)) > 0):
            return(True)
        else:
            return(False)
    

#set the working directory to the script directory
#runvars = dict()
#runvars.fileloc = os.path.abspath(__file__)
#runvars.dir = os.path.dirname(abspath)
#runvars.file_path = ".\\"
#runvars.datetime.now()
#runvars.pixel_dimensions = 0.4807637

#os.chdir(runvars.dir)

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
        alterConvolvedImage(temp_file)
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
    # check iages exist if(checkImagesexist(image_name) = False): 
    im_conv = skimage.io.imread(file_path+"03_TIF_convolved\\"+image_name, plugin='tifffile')
    im_blob = skimage.io.imread(file_path+"02_TIF_tiled\\"+image_name, plugin='tifffile')

    #im_str = skimage.io.imread(file_path+"02_TIF_tiled\\"+image_name, plugin='tifffile')
    #im_blob = im_str.copy()
    im_conv[im_conv == 255] = 1 #normalise values
    im_blob[im_blob < 50] = 0 #normalise values
    im_blob[im_blob >=50 ] = 1 #normalise values
    im_blob = skimage.morphology.erosion(im_blob, np.ones((3,3)))
    im_blob = skimage.morphology.dilation(im_blob, np.ones((4,4)))
    #im_str[im_str < 10] = 0 #normalise values
    #im_str[im_str >=50 ] = 0 #normalise values
    #im_str[im_str < 0] = 1 #normalise values
    #im_str = skimage.morphology.erosion(im_str, np.ones((1,4)))
    #im_str = skimage.morphology.dilation(im_str, np.ones((1,4)))

    string_output_image, string_output_data = stringSegmentation(im_conv, im_blob, image_name, file_path)

    print "Analysing {0}, image {1} of {2}, detected:".format(image_name, image+1, number_of_images)
    print "    {0} strings ".format(len(string_output_data))


print '\nAnalysis time: ', datetime.now() - start_time, ' seconds'