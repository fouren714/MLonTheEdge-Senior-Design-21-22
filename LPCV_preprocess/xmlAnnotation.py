import elementpath
import pandas as pd
import numpy as np
from xml.dom.minidom import parseString

# =============================================================================
# from lxml import etree
# import xml.etree.ElementTree as ET
# =============================================================================
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree, tostring
import xml.etree.ElementTree as ET

from pascal_voc_writer import Writer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2


def create_labels(df,w,h,variety,frames,frame_num,path):
    annotation = Element('annotation')
    SubElement(annotation, 'folder').text = 'dataset/images/'
    SubElement(annotation, 'filename').text = '{}_frame{}.jpg'.format(variety, str(frame_num))
    SubElement(annotation, 'segmented').text = '0'
    size = SubElement(annotation, 'size')
    SubElement(size, 'width').text = str(w)
    SubElement(size, 'height').text = str(h)
    SubElement(size, 'depth').text = str(df['Depth'].iloc[0])

    #print('element created')
    
    more_boxes = True
    # set i to first index of frame

    index_list = df.index[df['Frame']==frame_num].tolist()    
    j = index_list[0]
    
    frame_df = df[(df["Frame"] == frame_num)]
    
    for j in range(len(frame_df)):
        ob = SubElement(annotation, 'object')
        SubElement(ob, 'name').text = str(frame_df['Class'].iloc[j])
        SubElement(ob, 'pose').text = 'Unspecified'
        SubElement(ob, 'truncated').text = '0'
        SubElement(ob, 'difficult').text = '0'
        bbox = SubElement(ob, 'bndbox')
        SubElement(bbox, 'xmin').text = str(frame_df['X_min'].iloc[j])
        SubElement(bbox, 'ymin').text = str(frame_df['Y_min'].iloc[j])
        SubElement(bbox, 'xmax').text = str(df['X_max'].iloc[j])
        SubElement(bbox, 'ymax').text = str(df['Y_max'].iloc[j])
        
# =============================================================================
#         #add last box .iloc[i+1]
#         if ( df['Frame'].iloc[j] != df['Frame'].iloc[j+1]):
#             more_boxes = False
#             j = j + 1
#         if more_boxes == True:
#             j = j + 1
# =============================================================================
   
    #format ElementTree object
    tree = ElementTree(annotation)
    xmlstr = tostring(tree.getroot(), encoding='utf8', method='xml')
    formatted_tree = parseString(xmlstr).toprettyxml()
    

    #setup file path
    fileName = '{}_frame{}'.format(variety,str(frame_num))
    save_path = path
    file_name = fileName + '.xml'
    complete_name = os.path.join(save_path, file_name)
    
    #write xml file to directory
    with open(complete_name, 'w') as file:
        file.write(formatted_tree)
    
   

def scale(df_col,dim):
    return (df_col * dim).astype(int)

def frames_list(df):
    frames_list = []
    #frames_list.append(df['Frame'].iloc[0])
    
    frame = df['Frame']
    for i in range(len(df)-1):
        if(frame.iloc[i+1] != frame.iloc[i] ):
            frames_list.append(frame.iloc[i])
    frames_list.append(frame.iloc[len(df)-1])        
    print('frames list:',frames_list)         
    return frames_list

def extract_frames(df,variety):
    # get list of frames from dataframe (CSV)
    frames = frames_list(df)
    
    path_base = 'C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/'
    
    #path should be relative path for openCV
    path = path_base + 'dataset/images/'#variety + '_frames/'
    #os.mkdir(path)
    
    video = variety +'/' + variety +'.m4v'
    #print(video)
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    index = 0
    while success:
        #check to see if you are at the end of frames list
        if index == len(frames)-1:
            cv2.imwrite(path+"{}_frame{}.jpg".format(variety,count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            break 
        # write image when count matches a frame number
        elif count == frames[index]:
            cv2.imwrite(path+"{}_frame{}.jpg".format(variety,count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            
            count = count +1
            index = index + 1
            #print(count,index)
        # if index is not at end and count isnt equal to a needed frame: increment count
        else:
            count = count + 1
    print("Finished Reading: " + video)
    
#extracts frames from m4v and creates/augments data frame used in all_labels
def create_df(variety):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/"
    path = path_base + variety +"/"+ variety +".csv"
      
    fields = ['Frame', 'Class', 'ID' , 'X' , 'Y',  'Width', 'Height']
    
    df = pd.read_csv(path, usecols=fields)
    
    #create directory with needed frames
    extract_frames(df, variety)
    
    img_path = path_base + variety + '_frames/{}_frame0.jpg'.format(variety)
    
    img = Image.open(img_path)
    #'C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/'+variety+'_frames'
    w = img.width
    h = img.height
    
   
    #print(frame_height)
    #assuming x is centerpoint make needed rows
    x_radius = df['Width']/2
    y_radius = df['Height']/2
    df['Depth'] = 3
    
    #create max and min for x and y and scale assuming percentages
    df['X_max'] = scale((df['X'] + x_radius),w)
    df['Y_max'] = scale((df['Y'] + y_radius),h)
    
    df['X_min'] = scale((df['X'] - x_radius),w)
    df['Y_min'] = scale((df['Y'] - y_radius),h)
    
    #create xml header
    #create_header(df,frame_width,frame_height,variety, '0') #put in for loop and 
    #iterate frame num
    
# =============================================================================
#     #add objects
#     i =0
#     while df['Frame'].iloc[i] == df['Frame'].iloc[i+1]:
#         object_and_bbox(df, xml, i)
#         i = i+1
#     
#     print(xml)
#     #write file
#     write_file(df,xml,i)
# =============================================================================
    
    return df





def all_labels(df,variety):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/"
    
    #make folder for labels 
    
    path = path_base + 'dataset/annotations/'
    #os.mkdir(path)
    
    frames = frames_list(df)
    
    # go through enitre df
   
    for frame in frames:
        
        img_path = path_base + variety + '_frames/{}_frame{}.jpg'.format(variety,str(frame))
        img = Image.open(img_path)
        
        frame_width = img.width
        frame_height = img.height
        #make xml annotation for each frame with multuple objects
        create_labels(df,frame_width,frame_height,variety, frames, frame,path)

videos = ['4p1b_01A2','5p2b_01A1','5p4b_01A2','5p5b_03A1','7p3b_02M']

# =============================================================================
# for video in videos:
#     df = create_df(video)
#     all_labels(df,video) #need frame num
# =============================================================================
# =============================================================================
# df= create_df(videos[0])
# all_labels(df,videos[0]) #need frame num
# =============================================================================
