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

#importlib.reload(ET)






def create_labels(df,w,h,variety,frames,frame_num,path):
    annotation = Element('annotation')
    SubElement(annotation, 'folder').text = variety+'_frames'
    SubElement(annotation, 'filename').text = 'frame' + str(frame_num) + '.jpg'
    SubElement(annotation, 'segmented').text = '0'
    size = SubElement(annotation, 'size')
    SubElement(size, 'width').text = str(w)
    SubElement(size, 'height').text = str(h)
    SubElement(size, 'depth').text = str(df['Depth'].iloc[0])

    
    #print('element created')
    
   
    
    more_boxes = True
    # set i to first index of frame

    index_list = df.index[df_5p4b['Frame']==frame_num].tolist()    
    j = index_list[0]
    while more_boxes == True:
        ob = SubElement(annotation, 'object')
        SubElement(ob, 'name').text = str(df['Class'].iloc[j])
        SubElement(ob, 'pose').text = 'Unspecified'
        SubElement(ob, 'truncated').text = '0'
        SubElement(ob, 'difficult').text = '0'
        bbox = SubElement(ob, 'bndbox')
        SubElement(bbox, 'xmin').text = str(df['X_min'].iloc[j])
        SubElement(bbox, 'ymin').text = str(df['Y_min'].iloc[j])
        SubElement(bbox, 'xmax').text = str(df['X_max'].iloc[j])
        SubElement(bbox, 'ymax').text = str(df['Y_max'].iloc[j])
        
        #add last box .iloc[i+1]
        if ( df['Frame'].iloc[j] != df['Frame'].iloc[j+1]):
            more_boxes = False
            j = j + 1
        if more_boxes == True:
            j = j + 1
   
    #format ElementTree object
    tree = ElementTree(annotation)
    xmlstr = tostring(tree.getroot(), encoding='utf8', method='xml')
    formatted_tree = parseString(xmlstr).toprettyxml()
    

    #setup file path
    fileName = 'frame' + str(frame_num)
    save_path = path
    file_name = fileName + '.xml'
    complete_name = os.path.join(save_path, file_name)
    
    #write xml file to directory
    with open(complete_name, 'w') as file:
        file.write(formatted_tree)
    
    #tree.write(fileName + ".xml") #, encoding='utf8'
    #print('file complete')



def scale(df_col,dim):
    return (df_col * dim).astype(int)

def frames_list(df):
    frames_list = []
    #frames_list.append(df['Frame'].iloc[0])
    
    frame = df['Frame']
    for i in range(len(df)-1):
        if(frame.iloc[i+1] != frame.iloc[i] ):
            frames_list.append(frame.iloc[i])
             
    return frames_list


def create_df(video_csv, variety):
    #'C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/5p4b_01A2'
    path = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/"+variety+"/"+variety+".csv"
      
    fields = ['Frame', 'Class', 'ID' , 'X' , 'Y',  'Width', 'Height']
    
    df = pd.read_csv(path, usecols=fields)
    
    img = Image.open('C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/'+variety+'_frames/frame0.jpg')
    'C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/5p4b_frames'
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

s_5p4b = '5p4b_01A2'
df_5p4b = create_df('5pb4_01A2.mv4', '5p4b_01A2')


def all_labels(df,variety):
    
    
    #make folder for labels 
    path_base = 'C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/'
    path = path_base + variety + '_labels'
    os.mkdir(path)
    
    frames = frames_list(df)
    
    # go through enitre df
   
    for frame in frames:
        
        img = Image.open(path_base+variety+'_frames/frame'+str(frame)+'.jpg')
        
        frame_width = img.width
        frame_height = img.height
        #make xml annotation for each frame with multuple objects
        create_labels(df_5p4b,frame_width,frame_height,s_5p4b, frames, frame,path)

all_labels(df_5p4b,s_5p4b)
