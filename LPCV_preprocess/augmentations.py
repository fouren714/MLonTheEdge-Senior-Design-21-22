# -*- coding: utf-8 -*-

import pandas as pd
from PIL import Image
import numpy as np
from xmlAnnotation import *
import tensorflow as tf
import time

videos = ['4p1b_01A2','5p2b_01A1','5p4b_01A2','5p5b_03A1','7p3b_02M']

def create_OG_df(variety):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/"
    path = path_base + variety +"/"+ variety +".csv"
      
    fields = ['Frame', 'Class', 'ID' , 'X' , 'Y',  'Width', 'Height']
    
    df = pd.read_csv(path, usecols=fields)
    
    img_path = path_base + 'dataset/images/{}_frame0.jpg'.format(variety) #variety + '_frames/{}_frame0.jpg'.format(variety)
    
    img = Image.open(img_path)
    #'C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/'+variety+'_frames'
    w = img.width
    h = img.height
    
    df['Image_height'] = h
    df['Image_width'] = w
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
    
    return df

def insert_row(row_num,df,row_val):
    
    df.iloc[row_num] = row_val
    df.index = df.index+1

    return df

#Label Augmentaitons
#def swap_col(col1, col2):
    
def copy_label(df,frame_num,frame_shift):
    df_new = df[(df["Frame"] == frame_num)]
    df_new['Frame'] = frame_num+frame_shift
    df_new = df_new.reset_index(level = 0, drop = True)
    dfs = [df,df_new]
    
    new_df = pd.concat(dfs)
    
    #print(new_df)

    return new_df

def flip_label_lr(df,frame_num,frame_shift):
    df_new = df[(df["Frame"] == frame_num)]
    df_new['Frame'] = frame_num+frame_shift
    df_new = df_new.reset_index(level = 0, drop = True)
    #Shift xmin and xmax
    for i in range(len(df_new)-1):
        df_new['X_min'].iloc[i] = df_new['Image_width'].iloc[i] - df_new['X_min'].iloc[i]-1
        df_new['X_max'].iloc[i] = df_new['Image_width'].iloc[i] - df_new['X_max'].iloc[i]-1
    
    # TODO: swap xmin and xmax, flipping the actual frame should not be relevant
    
    dfs = [df,df_new]
    new_df = pd.concat(dfs)
    
    return new_df


def flip_label_ud(df,frame_num,frame_shift):
    df_new = df[(df["Frame"] == frame_num)]
    df_new['Frame'] = frame_num+frame_shift
    df_new = df_new.reset_index(level = 0, drop = True)
    #Shift xmin and xmax
    for i in range(len(df_new)-1):
        df_new['Y_min'].iloc[i] = df_new['Image_height'].iloc[i] - df_new['Y_min'].iloc[i]-1
        df_new['Y_max'].iloc[i] = df_new['Image_height'].iloc[i] - df_new['Y_max'].iloc[i]-1
    
    # TODO: swap ymin and ymax, flipping the actual frame should not be relevant
    
    dfs = [df,df_new]
    new_df = pd.concat(dfs)
    
    return new_df

def rot_90_label(df,frame_num,frame_shift):
    df_new = df[(df["Frame"] == frame_num)]
    df_new['Frame'] = frame_num+frame_shift
    #df_new.drop(['index'],1)
    df_new = df_new.reset_index(level = 0, drop = True)
    #print(df_new.index)
    #Shift xmin and xmax
    for i in range(len(df_new)):
        x_max = df_new['X_max'].iloc[i]
        y_max = df_new['Y_max'].iloc[i]
        x_min = df_new['X_min'].iloc[i]
        y_min = df_new['Y_min'].iloc[i]
        
        #rotate 90 degrees CCW
        df_new['X_max'].iloc[i] = y_min
        df_new['Y_max'].iloc[i] = x_max
        df_new['X_min'].iloc[i] = y_max
        df_new['Y_min'].iloc[i] = x_min
        
        #swap width and height
        w = df_new['Image_width'].iloc[i]
        h = df_new['Image_height'].iloc[i]
        
        df_new['Image_width'].iloc[i] = h
        df_new['Image_height'].iloc[i] = w
        
    
    dfs = [df,df_new]
    new_df = pd.concat(dfs)
    
    return new_df
    

#Image Augmentations
def contrast_image(df, variety, frame_num):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/dataset/images/" #{}_frames/".format(variety)
    image = '{}_frame{}.jpg'.format(variety, frame_num)
    img = Image.open(path_base+image)
    img = np.array(img)
    img = tf.image.random_contrast(img, 2, 5, 42)
    img = img.numpy()
    im = Image.fromarray(img)
    
    new_frame = frame_num + 20
    new_image = '{}_frame{}.jpg'.format(variety, new_frame)
    
    im.save(path_base + new_image)
    df = copy_label(df,frame_num, 20)
    return df

def noise_image(df, variety, frame_num):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/dataset/images/"
    image = '{}_frame{}.jpg'.format(variety, frame_num)
    img = Image.open(path_base+image)
    img = np.array(img)
    img = tf.image.stateless_random_jpeg_quality(img, 2, 20, (42,42))
    img = img.numpy()
    im = Image.fromarray(img)
    
    new_frame = frame_num + 31
    new_image = '{}_frame{}.jpg'.format(variety, new_frame)
    
    im.save(path_base + new_image)
    df = copy_label(df,frame_num,31)
    return df

def grayscale_image(df, variety, frame_num):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/dataset/images/"
    image = '{}_frame{}.jpg'.format(variety, frame_num)
    img = Image.open(path_base+image)
    img = np.array(img)
    img = tf.image.rgb_to_grayscale(img)
    img = img.numpy()
    img = img.reshape((img.shape[0], img.shape[1]))
    im = Image.fromarray(img, 'L')
    
    new_frame = frame_num + 41
    new_image = '{}_frame{}.jpg'.format(variety, new_frame)
    
    im.save(path_base + new_image)
    df = copy_label(df,frame_num,41)
    return df

def saturate_image(df, variety, frame_num):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/dataset/images/"
    image = '{}_frame{}.jpg'.format(variety, frame_num)
    img = Image.open(path_base+image)
    img = np.array(img)
    img = tf.image.random_saturation(img, 2, 120, 42)
    img = img.numpy()
    im = Image.fromarray(img)
    
    new_frame = frame_num + 51
    
    new_image = '{}_frame{}.jpg'.format(variety, new_frame)
    
    im.save(path_base + new_image)
    df = copy_label(df,frame_num,51)
    return df

def hue_image(df, variety, frame_num):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/dataset/images/"
    image = '{}_frame{}.jpg'.format(variety, frame_num)
    img = Image.open(path_base+image)
    img = np.array(img)
    img = tf.image.random_hue(img, 0.1)
    img = img.numpy()
    im = Image.fromarray(img)
    
    new_frame = frame_num + 61
    
    new_image = '{}_frame{}.jpg'.format(variety, new_frame)
    
    im.save(path_base + new_image)
    df = copy_label(df,frame_num,61)
    return df

def flip_image_lr(df, variety, frame_num):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/dataset/images/"
    image = '{}_frame{}.jpg'.format(variety, frame_num)
    img = Image.open(path_base+image)
    img = np.array(img)
    img = tf.image.flip_left_right(img)
    img = img.numpy()
    im = Image.fromarray(img)
    
    new_frame = frame_num + 71
    
    new_image = '{}_frame{}.jpg'.format(variety, new_frame)
    
    im.save(path_base + new_image)
    df = flip_label_lr(df,frame_num,71)
    return df

def flip_image_ud(df, variety, frame_num):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/dataset/images/"
    image = '{}_frame{}.jpg'.format(variety, frame_num)
    img = Image.open(path_base+image)
    img = np.array(img)
    img = tf.image.flip_up_down(img)
    img = img.numpy()
    im = Image.fromarray(img)
    
    new_frame = frame_num + 81
    
    new_image = '{}_frame{}.jpg'.format(variety, new_frame)
    
    im.save(path_base + new_image)
    df = flip_label_ud(df,frame_num,81)
    return df

def rot_90_image(df, variety, frame_num):
    path_base = "C:/Users/ouren/Documents/Senior_Design/TFODCourse/LPCV/dataset/images/"
    image = '{}_frame{}.jpg'.format(variety, frame_num)
    img = Image.open(path_base+image)
    img = np.array(img)
    img = tf.image.rot90(img)
    img = img.numpy()
    im = Image.fromarray(img)
    
    new_frame = frame_num + 91
    
    new_image = '{}_frame{}.jpg'.format(variety, new_frame)
    
    im.save(path_base + new_image)
    df = rot_90_label(df,frame_num,91)
    return df


#extract all original frames (run once)
# =============================================================================
# for video in videos:
#     df1 = create_df(video)
# =============================================================================

def augment_and_label(videos):
    #go through all videos
    for i in range(len(videos)):
        #make dataframe for video
        df = create_OG_df(videos[i])
        #get original frames
        og_frames = df.Frame.unique()
        print('original frames: ', og_frames)
        #use orginal frame list to augment (og_frames)
        #augmetnations shift frame_num to create unique frame number
        for frame in og_frames:
            df = contrast_image(df,videos[i],frame)
            df.reset_index(inplace = False)
            df = noise_image(df,videos[i],frame)
            df.reset_index(inplace = False)
            df = grayscale_image(df,videos[i],frame)
            df.reset_index(inplace = False)
            df = saturate_image(df,videos[i],frame)
            df.reset_index(inplace = False)
            df = hue_image(df,videos[i],frame)
            df.reset_index(inplace = False)
            df = flip_image_lr(df,videos[i],frame)
            df.reset_index(inplace = False)
            df = flip_image_ud(df,videos[i],frame)
            df.reset_index(inplace = False)
            df = rot_90_image(df,videos[i],frame)
            df.reset_index(inplace = False)
     
        all_labels(df,videos[i])
        
        return df
    
# =============================================================================
# df = augment_and_label(videos)
# =============================================================================

def labels_list(df):
    frame = df
    










# =============================================================================
# df = create_OG_df(videos[0])
# og_frames = df.Frame.unique()
# for frame in og_frames:
#     df = rot_90_image(df,videos[0],frame)
#     df = df.reset_index(level = 0, drop = True)
# =============================================================================

#Valdiate boxes
# =============================================================================
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# 
# im = Image.open('stinkbug.png')
# 
# # Create figure and axes
# fig, ax = plt.subplots()
# 
# # Display the image
# ax.imshow(im)
# 
# # Create a Rectangle patch
# rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
# 
# # Add the patch to the Axes
# ax.add_patch(rect)
# 
# plt.show()
# =============================================================================
    
    
