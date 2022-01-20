# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:44:02 2021

@author: Pierre Vuillecard 
"""

import numpy as np
from matplotlib import pyplot as plt
from pandas.io.pickle import to_pickle
from scipy.spatial.distance import cdist  
from itertools import combinations

import os
from os import path as osp
from pathlib import Path
from skimage import io
from skimage.feature import corner_harris, corner_peaks
from skimage.draw import polygon2mask
from skimage.morphology import binary_erosion
from skimage.morphology import square
import pandas as pd

import cv2

"""
Algorithm for box detection in angiographic images 
"""

# Define the potential color used to label the boxes and dots in images
color_to_code = {  'magenta':[1,0,1],
                        'yellow':[1,1,0],
                        'green':[0.13,1,0],
                        'brown':[0.6,0.4,0.2],
                        'blue':[0,0,1],
                        'cyan':[0,1,1],
                        'red':[1,0,0],
                        'orange':[1,0.5,0],
                        'magenta_dark':[0.5,0,0.5],
                        'light_gray':[0.75294118 , 0.9254902 , 0.99607843]

                }

class Dot():
    """
    Class Dot that define a dot in the image with its coordinates and
    the color.
    """
    def __init__(self,coords,color):
        self.coords = coords
        self.color = color
        
        
        
class Box():
    """
    Class Box that define a box with the coordinate of the corner,
    verify if it is a rectangle and also svae the color of the box.
    """
    def __init__(self,coords,color):
        
        # Verify that the box have 4 coordinate and
        # the box is a rectangle 
        assert(coords.shape[0]==4)
        if not isRectangle(coords,0.05):
            print('Warning: Box created but not a rectangle ')
        self.coords = coords
        self.color = color
        self.color_code = color_to_code[color]
        self.dots = []
        
        # Find coordinate A, B and D of the rectangle ABCD
        dist = cdist(coords, coords)[0,:]
        ind = [ i for i in [1,2,3] if i !=np.argmax(dist)]
        self.A = coords[0]
        self.B = coords[ind[0]]
        self.D = coords[ind[1]]
        
    def set_dot(self,dot):
        self.dots.append(dot)
    
    def get_dot_coord_light_gray(self):
        for dot in self.dots:
            if dot.color == 'light_gray' :
                return dot.coords.tolist()
        return []
    
    def get_categories(self):
        out = []
        if not self.dots :
            out.append('No')
        else :
            for dot in self.dots: 
                out.append(dot.color)
        return out
        
    def is_point_inside(self,point):
        AM = point-self.A
        AB = self.B-self.A
        AD = self.D-self.A
        if 0 < np.dot(AM,AB) and np.dot(AM,AB) < np.dot(AB,AB) : 
            if 0 < np.dot(AM,AD) and np.dot(AM,AD) < np.dot(AD,AD) :
                return True
            else :
                return False
        else:
            return False
        
        
        
def isRectangle(X_rect, eps_rect = 0.05 ) :
    """
    Check if 4 coordinates form a rectangle based on the distance
    between the corner and the center of mass of the rectangle.
    
    Parameters
    ----------
    X_rect : numpy array
        Array that contains the coordinates of the points 
    eps_rect : float [0,1], optional
        The tolerance error of being a rectangle 0.05 means 5% tolerance 
        . The default is 0.05.

    Returns
    -------
    bool
        Return True is the 4 coordinate form a rectangle 

    """
    assert(X_rect.shape[0]==4)
    center_mass = X_rect.sum(axis=0)/4
    dist =  np.reshape(center_mass,(2,1)) -X_rect.T
    dist = np.linalg.norm(dist,axis =0)
    err = np.max(dist)*eps_rect 
    if (np.abs(dist[0] - dist[1])<err) and (np.abs(dist[0] - dist[2])<err) and (np.abs(dist[0] - dist[3])<err):
        return True
    else : 
        return False


 
def isRightTriangle(X,eps = 0.05):
    """
    Check if 3 coordinates form a right triangle based on pythagore thm.
    
    Parameters
    ----------
    X : numpy array
        Array that contains the coordinates of the points 
    eps_rect : float [0,1], optional
        The tolerance error of being a rectangle 0.05 means 5% tolerance 
        . The default is 0.05.

    Returns
    -------
    bool
        Return True is the 3 coordinate form a right triangle 

    """
    assert(X.shape[0]==3)
    # check if it corespond to a right triangle 
    d1 = np.linalg.norm(X[0]-X[1])**2
    d2 = np.linalg.norm(X[0]-X[2])**2
    d3 = np.linalg.norm(X[1]-X[2])**2
    d = [d1,d2,d3]
    ind_hypothenus = np.argmax(d)
    err = eps*d[ind_hypothenus]
    pythogare = 0
    for i in [0,1,2]:
        if i != ind_hypothenus:
            pythogare += d[i]
    if np.abs(pythogare - d[ind_hypothenus])<err:
        return True
    else:
        return False
    
def reconstruct_rectangle(X, eps = 0.05):
    """
    Reconstruct a rectangle from a right triangle.

    Parameters
    ----------
    X : numpy array
        Array that contains the coordinates of the points 
    eps_rect : float [0,1], optional
        The tolerance error of being a rectangle 0.05 means 5% tolerance 
        . The default is 0.05.

    Returns
    -------
    new_coord : array
        return the missing point that form a rectangle

    """
    if isRightTriangle(X,eps):
        d1 = np.linalg.norm(X[0]-X[1])
        d2 = np.linalg.norm(X[0]-X[2])
        d3 = np.linalg.norm(X[1]-X[2])
        d = [d1,d2,d3]
        ind_hypothenus = np.argmax(d)
        indice_right_angle = [2,1,0][ind_hypothenus]
        indice_other = [ i for i in [0,1,2] if i != indice_right_angle ]
        center_mass = X[indice_other].sum(axis=0)/2
        new_coord = 2*center_mass - X[indice_right_angle]
    else:
        new_coord = None
    return new_coord



def find_rectangle_small(X,eps = 0.05):
    """
    Find a potential rectangle. Based on three point that 
    form a right triangle it will reconstruct a rectangle.
    
    Parameters
    ----------
    X : numpy array
        Array that contains the coordinates of the points .
    eps_rect : float [0,1], optional
        The tolerance error of being a rectangle 0.05 means 5% tolerance.
        The default is 0.05.

    Returns
    -------
    TYPE
        return the coordinate that form a rectangle .

    """
    l = [i for i in range(X.shape[0])]
    combs_ = combinations(l, 3)
    hyp = []
    combs = []
    for comb in combs_:
        combs.append(list(comb))
        
    for comb in combs:
        if isRightTriangle(X[comb],eps):
            dist = cdist(X[comb], X[comb])
            hyp.append(np.max(dist))
        else:
            hyp.append(0)
            
    if np.sum(hyp)==0:
        return None
    else :
        new_X = X[combs[np.argmax(hyp)]]
        new_coord = reconstruct_rectangle(new_X, eps)
        print(new_X.shape)
        print(new_coord.shape)
        return np.c_[new_X.T,np.reshape(new_coord,(2,1))].T

def find_rectangle(X,eps = 0.05):
    """
    Find a potential rectangle. Based on 4 points
    
    Parameters
    ----------
    X : numpy array
        Array that contains the coordinates of the points .
    eps_rect : float [0,1], optional
        The tolerance error of being a rectangle 0.05 means 5% tolerance.
        The default is 0.05.

    Returns
    -------
    TYPE
        return the coordinate that form a rectangle .

    """
    dist = cdist(X, X)
    ind_max_1 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    dist[ind_max_1[0],ind_max_1[1]] = 0 
    dist[ind_max_1[1],ind_max_1[0]] = 0 
    ind_max_2 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    while ind_max_1[0]==ind_max_2[0] or\
          ind_max_1[1]==ind_max_2[0] or\
          ind_max_1[0]==ind_max_2[1] or\
          ind_max_1[1]==ind_max_2[1] :
        dist[ind_max_2[0],ind_max_2[1]] = 0 
        dist[ind_max_2[1],ind_max_2[0]] = 0       
        ind_max_2 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    
    indice_rect = [ind_max_1[0],ind_max_1[1],ind_max_2[0],ind_max_2[1]]
    indice_rect_out = [i for i in range(X.shape[0]) if i not in indice_rect]
    #print(indice_rect)
    assert(len(np.unique(indice_rect))==4)
    if isRectangle(X[indice_rect],eps) :
        return np.array(indice_rect) , indice_rect_out
    else :
        return None , None

def close_border(X):
    """
    Bool function that determine is a box is close the border of the image.
    
    Parameters
    ----------
    X : numpy array
        Binary images with 1 for the rectangle.
    Returns
    -------
    TYPE
        True if next the border

    """
    border_size = 4
    x_n = X.shape[0]
    y_n = X.shape[1]
    x = np.where(X==1)
    if ((x[0]<border_size).any()) or ((x[0]>(x_n-border_size)).any()) or\
       ((x[1]<border_size).any()) or ((x[1]>(y_n-border_size)).any()):
        return True
    else:
        return False

def compute_binary_fast(image):
    """
    Parameters
    ----------
    image : numpy array
        RGB images
    Returns
    -------
    TYPE
        Array of binary image

    """
    image_black = (((image[:,:,0] == image[:,:,1]) *(image[:,:,0] == image[:,:,2])).astype(int)-1)*(-1)     
    return image_black

def cluster_point(X):
    """
    Clean detected dot to have only one coordinate per dot.

    Parameters
    ----------
    X : Numpy array
        Array of the points detected from the dot.

    Returns
    -------
    TYPE
        Only one coordinate per point .

    """
    dist = cdist(X, X)
    a = set()
    for i in range(X.shape[0]):
        a.add(i)
    for i in range(X.shape[0]):
        if i in a:
            l = [ k for k in range(X.shape[0]) if k!=i]
            for j in l:
                if j in a:
                    if dist[i,j]<30:
                        a.remove(j)
    out = list(a)
    return X[out]
    
    
        
def image_preprocessing(path, visualize = False, visualize_simple= False):
    """
    Algorithm that find the corner of the box along with the color,
    it also detect the dots coordinate and color and inside which box is it.

    Parameters
    ----------
    path : string path
        Path of the image location.
    visualize : Bool, optional
        True if you want a details visualization of the algorithm.
        The default is False.
    visualize_simple : Bool, optional
        True if you want a simple visualization of algorithm results.
        The default is False.

    Returns
    -------
    image_info : Dict 
        'error' : if the algorithm encouter minor problem,
        'path_box' : path of images,
        'patient_box' : patient id ,
        'segment_box' : segment id,
        'view_box' : view id,
        'color_box' : color of the box,
        'coordinate_box' : coordinate of the corner ,
        'categories_box' : categories of the box (color of dots inside the box)

    """
    confidence = 0.05
    # First extract the segment information from the path
    path_split = path.split('\\')
    patient = path_split[-2]
    sgmt = path_split[-1][:3]
    view = path_split[-1][4]
    error = False
    print(path)
    # Load the images 
    image = io.imread(path)
    x_n = image.shape[0]
    y_n = image.shape[1]
    
    if visualize :
        fig, ax = plt.subplots(figsize=(15,15))
        ax.imshow(image, cmap=plt.cm.gray)
        ax.set_title('Original images')
        plt.show()
                    
    # rescale images and keep only the RGB chanel 
    image_rescale = image[:,:,:-1]/255
    
    # The first step is to find the box for each color
    box_colors = [ 'magenta','yellow','green','brown','blue','magenta_dark' ]
    boxes = []
    
    for box_color in box_colors:
        # Select the point in the original images that match the most the color
        # The goal is to have a binary images where 1 corespond to the box color
        color_matrix = np.broadcast_to(color_to_code[box_color],(x_n, y_n, 3))
        #print(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)))
        image_one_color = (np.linalg.norm(image_rescale-color_matrix,axis=2)<= \
                           min(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)),0.06)*1.05).astype(int)
        if close_border(image_one_color):
            print('Box is near the border')
            error = True
        #Pad the images in order to detect corner close to the border of the images
        padding = 15
        image_one_color_pad = np.pad(image_one_color, ((padding, padding),\
                                    (padding,padding)), 'constant', constant_values=0)
        #print(box_color,np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)))
        #print('nb point ', image_one_color.sum() )
        if image_one_color.sum() > 1: 
            ''' 
            # Try using polygon2mask but can take long if box is big
            x,y = np.where(image_one_color == 1)
            polygon = np.c_[x,y]
            image_shape = (x_n, y_n)
            box = polygon2mask(image_shape, polygon)
            '''
            # Detect the corner in the filtered images that contains only one box 
            coords = corner_peaks(corner_harris(image_one_color_pad), min_distance=5, threshold_rel=0.02)
            nb_corner = coords.shape[0]
            coords = coords - padding 
            if visualize :
                fig, ax1= plt.subplots(figsize=(15,15))
                ax1.imshow(image_one_color, cmap=plt.cm.gray)
                ax1.plot(coords[:, 1], coords[:, 0], color='red', marker='o',
                        linestyle='None', markersize=6,label='Detected corner')
                ax1.set_title(box_color+ ' box corner detection')
                ax1.legend()
                plt.show()    
                
            if visualize :
                old_coords = coords.copy()
            
            # Then check that the corner define a rectangle different case are here
            # When 4 coordinate corespond to a rectangle a Box object is created 
            # That contain the coordinate of the corner and the color of the box
            if nb_corner <=2:
                print('Not abble to find the rectangle of color', box_color,' in image path : ',path)
                error = True
            elif nb_corner ==3:
                new_coord = reconstruct_rectangle(coords, confidence)
                if not new_coord is None:
                    coords = np.c_[coords.T,np.reshape(new_coord,(2,1))].T
                    boxes.append(Box(coords,box_color))
                else:
                    print('Not abble to find the rectangle of color', box_color,' in image path : ',path)
                    error = True
            elif nb_corner ==4:
                if isRectangle(coords,confidence):
                    print('Is a rectangle')
                    boxes.append(Box(coords,box_color))
                else: 
                    coords = find_rectangle_small(coords,0.1)
                    if coords is None :
                        print('Not abble to find the rectangle of color', box_color,' in image path : ',path)
                        error = True
                    else :
                        boxes.append(Box(coords,box_color))
            else:
                indice_rect , indice_rect_out = find_rectangle(coords,eps = confidence)
                if indice_rect is None :
                    coords = find_rectangle_small(coords,confidence)
                    if coords is None :
                        print('Not abble to find the rectangle of color', box_color,' in image path : ',path)
                        error = True
                    else :
                        boxes.append(Box(coords,box_color))
                    
                else: 
                    coords = coords[indice_rect]
                    boxes.append(Box(coords,box_color))
            
            # PLot the original images with the corner detected
            if visualize :
                fig, (ax1 ,ax2)= plt.subplots(1,2,figsize=(15,15))
                ax1.imshow(image_one_color, cmap=plt.cm.gray)
                ax1.plot(old_coords[:, 1], old_coords[:, 0], color='red', marker='o',
                        linestyle='None', markersize=6,label='Detected corner')
                ax1.set_title(box_color+ ' box corner detection')
                ax2.imshow(image_one_color, cmap=plt.cm.gray)
                ax2.plot(coords[:, 1], coords[:, 0], color='red', marker='o',
                        linestyle='None', markersize=6,label='Cleaned corner')
                ax2.set_title(box_color+ ' box corner cleaning')
                ax1.legend()
                ax2.legend()
                plt.show()    
    
    # Finally, we need to include the dot that corespond a categories 
    # for each box. First we detect the dot and then find the box that
    # contains the dots
    dot_colors = ['orange', 'cyan', 'red']
    for dot_color in dot_colors:
        # Find the dot with the coresponding color in the original images
        # If it exist proceed with the detection
        color_matrix = np.broadcast_to(color_to_code[dot_color],(x_n, y_n, 3))
        image_one_color = (np.linalg.norm(image_rescale-color_matrix,axis=2)<= \
                           min(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2))*1.05,0.2)).astype(int)
        
        if image_one_color.sum()>1 :
            # Detect the dot in the images 
            #image_one_color = binary_erosion(image_one_color,square(8))
            coords = corner_peaks(corner_harris(image_one_color), min_distance=10, threshold_rel=0.02)
            #cluster the point 
            coords = cluster_point(coords)
            if visualize :
                fig, ax1 = plt.subplots(1,figsize=(15,15))
                ax1.imshow(image_one_color, cmap=plt.cm.gray)
                ax1.plot(coords[:, 1], coords[:, 0], color='red', marker='x',
                        linestyle='None', markersize=5,label='Detected corner')
                ax1.set_title(dot_color+ ' dot detection')
                plt.show()    
                
            # For each dot find the boxe that contains it 
            for coord_point in coords:
                is_in = []
                for box in boxes:
                    if box.is_point_inside(coord_point):
                        is_in.append(True)
                    else:
                        is_in.append(False)
                        
                # Verify that the point can be in only one box
                if np.sum(is_in)==0:
                    print('Dot found no box')
                    error = True
                elif np.sum(is_in)>1:
                    print('Dot found too much box')
                    error = True
                else :
                    ind_is_in = np.where(is_in)[0][0]
                    # Add the dot to the Box object 
                    boxes[ind_is_in].set_dot(Dot(coord_point,dot_color))
    # Plot the dot detection 
    if visualize :
        fig, ax = plt.subplots(figsize=(15,15))
        ax.imshow(image, cmap=plt.cm.gray)
        for i,box in enumerate(boxes):
            if not box.dots is None:
                for dot in box.dots:
                    tmp = list(color_to_code[dot.color].copy())
                    tmp.reverse()
                    ax.plot(dot.coords[ 1], dot.coords[ 0], color= tmp, marker='x',
                            linestyle='None', markersize=15,label = dot.color+' dot inside '+box.color+' box')
        ax.set_title('Dots detection')
        plt.legend()
        plt.show()
    # Plot the box corner coordinate with the dot along with the good boxes
    if visualize_simple :
        print('error is :',error)
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(image, cmap=plt.cm.gray)
        for i,box in enumerate(boxes):
            ax.plot(box.coords[:,1], box.coords[:,0], color= color_to_code[box.color], marker='x',
                            linestyle='None', markersize=15,label = box.color+' box')
            if not box.dots is None:
                for dot in box.dots:
                    tmp = list(color_to_code[dot.color].copy())
                    tmp.reverse()
                    ax.plot(dot.coords[1], dot.coords[0], color= tmp, marker='x',
                            linestyle='None', markersize=15,label = dot.color+' dot inside '+box.color+' box')
        ax.set_title('Dots detection')
        plt.legend()
        plt.show()
    
    # Save all the information into a dict for each boxes deteced in the images
    image_info = {
    'error' : [],
    'path_box' : [],
    'patient_box' : [],
    'segment_box' : [],
    'view_box' : [],
    'color_box' : [],
    'coordinate_box' : [],
    'categories_box' : []
    }
    for box in boxes :
        image_info['error'].append(error)
        image_info['path_box'].append(path)
        image_info['patient_box'].append(patient)
        image_info['segment_box'].append(sgmt)
        image_info['view_box'].append(view)
        image_info['color_box'].append(box.color)
        image_info['coordinate_box'].append(box.coords.tolist())
        image_info['categories_box'].append(box.get_categories())
        
    return  image_info



def image_preprocessing_test(path, visualize = False, visualize_simple= False):
    """
    Algorithm that find the corner of the box along with the color,
    it also detect the dots coordinate and color and inside which box is it.

    Parameters
    ----------
    path : string path
        Path of the image location.
    visualize : Bool, optional
        True if you want a details visualization of the algorithm.
        The default is False.
    visualize_simple : Bool, optional
        True if you want a simple visualization of algorithm results.
        The default is False.

    Returns
    -------
    image_info : Dict 
        'error' : if the algorithm encouter minor problem,
        'path_box' : path of images,
        'patient_box' : patient id ,
        'segment_box' : segment id,
        'view_box' : view id,
        'color_box' : color of the box,
        'coordinate_box' : coordinate of the corner ,
        'categories_box' : categories of the box (color of dots inside the box)

    """
    # First extract the segment information from the path
    print(path)
    path_split = path.split('/')
    patient = path_split[-2]
    sgmt = path_split[-1][:3]
    view = path_split[-1][4]
    error = False
    
    # Load the images 
    image = io.imread(path)
    x_n = image.shape[0]
    y_n = image.shape[1]
    
    if visualize :
        fig, ax = plt.subplots(figsize=(15,15))
        ax.imshow(image, cmap=plt.cm.gray)
        ax.set_title('Original images')
        plt.show()
                    
    # rescale images and keep only the RGB chanel
    if image.shape[2] == 4 : 
        image_rescale = image[:,:,:-1]/255
    else :
        image_rescale = image[:,:,:]/255
    
    # The first step is to find the box for each color
    box_colors = [ 'magenta','yellow','green','brown','blue','magenta_dark' ]
    boxes = []
    
    for box_color in box_colors:
        # Select the point in the original images that match the most the color
        # The goal is to have a binary images where 1 corespond to the box color
        color_matrix = np.broadcast_to(color_to_code[box_color],(x_n, y_n, 3))
        #print(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)))
        image_one_color = (np.linalg.norm(image_rescale-color_matrix,axis=2)<= \
                           min(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)),0.06)*1.05).astype(int)
        if close_border(image_one_color):
            print('Box is near the border')
            error = True
        #Pad the images in order to detect corner close to the border of the images
        padding = 200
        image_one_color_pad = np.pad(image_one_color, ((padding, padding),\
                                    (padding,padding)), 'constant', constant_values=0)
        #print(box_color,np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)))
        #print('nb point ', image_one_color.sum() )
        if image_one_color.sum() > 1: 
            ''' 
            # Try using polygon2mask but can take long if box is big
            x,y = np.where(image_one_color == 1)
            polygon = np.c_[x,y]
            image_shape = (x_n, y_n)
            box = polygon2mask(image_shape, polygon)
            '''
            # Detect the corner in the filtered images that contains only one box 
            ##coords = corner_peaks(corner_harris(image_one_color_pad), min_distance=5, threshold_rel=0.02)
            ##nb_corner = coords.shape[0]
            ##coords = coords - padding
            point_in = np.where(image_one_color_pad==1)
            point_in = np.c_[point_in[0],point_in[1]]
            rect = cv2.minAreaRect(point_in)
            box = cv2.boxPoints(rect)
            coords = np.int0(box) -padding 
            nb_corner = 4
            coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0].copy()
            boxes.append(Box(coords,box_color))
            
            if visualize :
                fig, ax1= plt.subplots(figsize=(15,15))
                ax1.imshow(image_one_color, cmap=plt.cm.gray)
                ax1.plot(coords[:, 0], coords[:, 1], color='red', marker='o',
                        linestyle='None', markersize=6,label='Detected corner')
                ax1.set_title(box_color+ ' box corner detection')
                ax1.legend()
                plt.show()    
            
    # Finally, we need to include the dot that corespond a categories 
    # for each box. First we detect the dot and then find the box that
    # contains the dots
    dot_colors = ['orange', 'cyan', 'red']
    for dot_color in dot_colors:
        # Find the dot with the coresponding color in the original images
        # If it exist proceed with the detection
        color_matrix = np.broadcast_to(color_to_code[dot_color],(x_n, y_n, 3))
        image_one_color = (np.linalg.norm(image_rescale-color_matrix,axis=2)<= \
                           min(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2))*1.05,0.2)).astype(int)
        
        if image_one_color.sum()>1 :
            # Detect the dot in the images 
            #image_one_color = binary_erosion(image_one_color,square(8))
            coords = corner_peaks(corner_harris(image_one_color), min_distance=10, threshold_rel=0.02)
            #cluster the point 
            coords = cluster_point(coords)
            coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0].copy()
            if visualize :
                fig, ax1 = plt.subplots(1,figsize=(15,15))
                ax1.imshow(image_one_color, cmap=plt.cm.gray)
                ax1.plot(coords[:, 1], coords[:, 0], color='red', marker='x',
                        linestyle='None', markersize=5,label='Detected corner')
                ax1.set_title(dot_color+ ' dot detection')
                plt.show()    
                
            # For each dot find the boxe that contains it 
            for coord_point in coords:
                is_in = []
                for box in boxes:
                    if box.is_point_inside(coord_point):
                        is_in.append(True)
                    else:
                        is_in.append(False)
                        
                # Verify that the point can be in only one box
                if np.sum(is_in)==0:
                    print('Dot found no box')
                    error = True
                elif np.sum(is_in)>1:
                    print('Dot found too much box')
                    error = True
                else :
                    ind_is_in = np.where(is_in)[0][0]
                    # Add the dot to the Box object 
                    boxes[ind_is_in].set_dot(Dot(coord_point,dot_color))
    # Plot the dot detection 
    if visualize :
        fig, ax = plt.subplots(figsize=(15,15))
        ax.imshow(image, cmap=plt.cm.gray)
        for i,box in enumerate(boxes):
            if not box.dots is None:
                for dot in box.dots:
                    tmp = list(color_to_code[dot.color].copy())
                    tmp.reverse()
                    ax.plot(dot.coords[ 0], dot.coords[ 1], color= tmp, marker='x',
                            linestyle='None', markersize=15,label = dot.color+' dot inside '+box.color+' box')
        ax.set_title('Dots detection')
        plt.legend()
        plt.show()
    # Plot the box corner coordinate with the dot along with the good boxes
    if visualize_simple :
        print('error is :',error)
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(image, cmap=plt.cm.gray)
        for i,box in enumerate(boxes):
            ax.plot(box.coords[:,0], box.coords[:,1], color= color_to_code[box.color], marker='x',
                            linestyle='None', markersize=15,label = box.color+' box')
            if not box.dots is None:
                for dot in box.dots:
                    tmp = list(color_to_code[dot.color].copy())
                    tmp.reverse()
                    ax.plot(dot.coords[0], dot.coords[1], color= tmp, marker='x',
                            linestyle='None', markersize=15,label = dot.color+' dot inside '+box.color+' box')
        ax.set_title('Dots detection')
        plt.legend()
        plt.show()
    
    # Save all the information into a dict for each boxes deteced in the images
    image_info = {
    'error' : [],
    'path_box' : [],
    'patient_box' : [],
    'segment_box' : [],
    'view_box' : [],
    'color_box' : [],
    'coordinate_box' : [],
    'categories_box' : []
    }
    for box in boxes :
        image_info['error'].append(error)
        image_info['path_box'].append(path)
        image_info['patient_box'].append(patient)
        image_info['segment_box'].append(sgmt)
        image_info['view_box'].append(view)
        image_info['color_box'].append(box.color)
        image_info['coordinate_box'].append(box.coords.tolist())
        image_info['categories_box'].append(box.get_categories())
        
    return  image_info


def image_preprocessing_dot_extract(path, visualize = False, visualize_simple= False):
    """
    Algorithm that find the corner of the box along with the color,
    it also detect the dots coordinate and color and inside which box is it.

    Parameters
    ----------
    path : string path
        Path of the image location.
    visualize : Bool, optional
        True if you want a details visualization of the algorithm.
        The default is False.
    visualize_simple : Bool, optional
        True if you want a simple visualization of algorithm results.
        The default is False.

    Returns
    -------
    image_info : Dict 
        'error' : if the algorithm encouter minor problem,
        'path_box' : path of images,
        'patient_box' : patient id ,
        'segment_box' : segment id,
        'view_box' : view id,
        'color_box' : color of the box,
        'coordinate_box' : coordinate of the corner ,
        'categories_box' : categories of the box (color of dots inside the box)

    """
    # First extract the segment information from the path
    print(path)
    path_split = path.split('/')
    patient = path_split[-2]
    sgmt = path_split[-1][:3]
    view = path_split[-1][4]
    error = False
    
    # Load the images 
    image = io.imread(path)
    x_n = image.shape[0]
    y_n = image.shape[1]
    
    if visualize :
        fig, ax = plt.subplots(figsize=(15,15))
        ax.imshow(image, cmap=plt.cm.gray)
        ax.set_title('Original images')
        plt.show()
                    
    # rescale images and keep only the RGB chanel
    if image.shape[2] == 4 : 
        image_rescale = image[:,:,:-1]/255
    else :
        image_rescale = image[:,:,:]/255
    
    # The first step is to find the box for each color
    box_colors = [ 'magenta','yellow','green','brown','blue','magenta_dark' ]
    boxes = []
    
    for box_color in box_colors:
        # Select the point in the original images that match the most the color
        # The goal is to have a binary images where 1 corespond to the box color
        color_matrix = np.broadcast_to(color_to_code[box_color],(x_n, y_n, 3))
        #print(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)))
        image_one_color = (np.linalg.norm(image_rescale-color_matrix,axis=2)<= \
                           min(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)),0.06)*1.05).astype(int)
        if close_border(image_one_color):
            print('Box is near the border')
            error = True
        #Pad the images in order to detect corner close to the border of the images
        padding = 200
        image_one_color_pad = np.pad(image_one_color, ((padding, padding),\
                                    (padding,padding)), 'constant', constant_values=0)
        #print(box_color,np.min(np.linalg.norm(image_rescale-color_matrix,axis=2)))
        #print('nb point ', image_one_color.sum() )
        if image_one_color.sum() > 1: 
            ''' 
            # Try using polygon2mask but can take long if box is big
            x,y = np.where(image_one_color == 1)
            polygon = np.c_[x,y]
            image_shape = (x_n, y_n)
            box = polygon2mask(image_shape, polygon)
            '''
            # Detect the corner in the filtered images that contains only one box 
            ##coords = corner_peaks(corner_harris(image_one_color_pad), min_distance=5, threshold_rel=0.02)
            ##nb_corner = coords.shape[0]
            ##coords = coords - padding
            point_in = np.where(image_one_color_pad==1)
            point_in = np.c_[point_in[0],point_in[1]]
            rect = cv2.minAreaRect(point_in)
            box = cv2.boxPoints(rect)
            coords = np.int0(box) -padding 
            nb_corner = 4
            coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0].copy()
            boxes.append(Box(coords,box_color))
            
            if visualize :
                fig, ax1= plt.subplots(figsize=(15,15))
                ax1.imshow(image_one_color, cmap=plt.cm.gray)
                ax1.plot(coords[:, 0], coords[:, 1], color='red', marker='o',
                        linestyle='None', markersize=6,label='Detected corner')
                ax1.set_title(box_color+ ' box corner detection')
                ax1.legend()
                plt.show()    
            
    # Finally, we need to include the dot that corespond a categories 
    # for each box. First we detect the dot and then find the box that
    # contains the dots
    dot_colors = ['light_gray']
    for dot_color in dot_colors:
        # Find the dot with the coresponding color in the original images
        # If it exist proceed with the detection
        color_matrix = np.broadcast_to(color_to_code[dot_color],(x_n, y_n, 3))
        image_one_color = (np.linalg.norm(image_rescale-color_matrix,axis=2)<= \
                           min(np.min(np.linalg.norm(image_rescale-color_matrix,axis=2))*1.05,0.2)).astype(int)
        
        if image_one_color.sum()>1 :
            # Detect the dot in the images 
            #image_one_color = binary_erosion(image_one_color,square(8))
            coords = corner_peaks(corner_harris(image_one_color), min_distance=10, threshold_rel=0.02)
            #cluster the point 
            coords = cluster_point(coords)
            coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0].copy()
            if visualize :
                fig, ax1 = plt.subplots(1,figsize=(15,15))
                ax1.imshow(image_one_color, cmap=plt.cm.gray)
                ax1.plot(coords[:, 0], coords[:, 1], color='red', marker='x',
                        linestyle='None', markersize=5,label='Detected corner')
                ax1.set_title(dot_color+ ' dot detection')
                plt.show()    
                
            # For each dot find the boxe that contains it 
            for coord_point in coords:
                is_in = []
                for box in boxes:
                    if box.is_point_inside(coord_point):
                        is_in.append(True)
                    else:
                        is_in.append(False)
                        
                # Verify that the point can be in only one box
                if np.sum(is_in)==0:
                    print('Dot found no box')
                    error = True
                else :
                    ind_is_in = np.where(is_in)[0][0]
                    # Add the dot to the Box object 
                    boxes[ind_is_in].set_dot(Dot(coord_point,dot_color))
    # Plot the dot detection 
    if visualize :
        fig, ax = plt.subplots(figsize=(15,15))
        ax.imshow(image, cmap=plt.cm.gray)
        for i,box in enumerate(boxes):
            if not box.dots is None:
                for dot in box.dots:
                    tmp = list(color_to_code[dot.color].copy())
                    tmp.reverse()
                    ax.plot(dot.coords[ 0], dot.coords[ 1], color= tmp, marker='x',
                            linestyle='None', markersize=15,label = dot.color+' dot inside '+box.color+' box')
        ax.set_title('Dots detection')
        plt.legend()
        plt.show()
    # Plot the box corner coordinate with the dot along with the good boxes
    if visualize_simple :
        print('error is :',error)
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(image, cmap=plt.cm.gray)
        for i,box in enumerate(boxes):
            ax.plot(box.coords[:,0], box.coords[:,1], color= color_to_code[box.color], marker='x',
                            linestyle='None', markersize=15,label = box.color+' box')
            if not box.dots is None:
                for dot in box.dots:
                    tmp = list(color_to_code[dot.color].copy())
                    tmp.reverse()
                    ax.plot(dot.coords[0], dot.coords[1], color= tmp, marker='x',
                            linestyle='None', markersize=15,label = dot.color+' dot inside '+box.color+' box')
        ax.set_title('Dots detection')
        plt.legend()
        plt.show()
    
    # Save all the information into a dict for each boxes deteced in the images
    image_info = {
    'error' : [],
    'path_box' : [],
    'patient_box' : [],
    'segment_box' : [],
    'view_box' : [],
    'color_box' : [],
    'coordinate_box' : [],
    'categories_box' : [],
    'coord_stenosis': []
    }
    for box in boxes :
        image_info['error'].append(error)
        image_info['path_box'].append(path)
        image_info['patient_box'].append(patient)
        image_info['segment_box'].append(sgmt)
        image_info['view_box'].append(view)
        image_info['color_box'].append(box.color)
        image_info['coordinate_box'].append(box.coords.tolist())
        image_info['categories_box'].append(box.get_categories())
        image_info['coord_stenosis'].append(box.get_dot_coord_light_gray())
        
    return  image_info

def crop_images(df_image_info):

    image_file_name = []
    image_file_label = []

    image_dir = './data/image_box'
    for index, row in df_image_info.iterrows():
        
        image = io.imread(row['path_box'][:-4]+' copie.tif')
        # Keep only the RGB
        if image.shape[2] == 4:
            image = image[:,:,:-1]
        #print(row['path_box'],row['color_box'])
        coor = np.array(row['coordinate_box']).astype(int)
        
        rect = cv2.minAreaRect(coor)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        file_to_save = row['patient_box']+'__'+row['segment_box']+'__'+row['color_box']+'__'+str(row['view_box'])+'.png'
        image_file_name.append(file_to_save)
        label = row['target_box']
        image_file_label.append(label)

        # Rotate the image to always have an horizontal rectangular shape 
        if width < height :
            warped = np.rot90(warped)
            
        cv2.imwrite(os.path.join(image_dir,file_to_save ), warped)
    df_annotated_box = pd.DataFrame(list(zip(image_file_name,image_file_label)),columns = ['file','label'])
    df_annotated_box.to_pickle(os.path.join(image_dir,'annotated_box' ))

def crop(path,coor):
    image = io.imread(path)
    # transform into grayscal images
    image = image[:,:,0]
    #coor[:, 0], coor[:, 1] = coor[:, 1], coor[:, 0].copy()
    
    # Take the rectangle to crop the images.
    rect = cv2.minAreaRect(coor)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    # Rotate the image to always have an horizontal rectangular shape 
    if width < height :
        warped = np.rot90(warped)
    return warped

def crop_images_save_info(image_dir,img_name, row, x1, x2, aug='no'):
    add_annotation = [img_name]
    add_annotation.append(row['path_view1'])
    add_annotation.append(row['path_view2'])
    add_annotation.append(row['target_box'])
    add_annotation.append(row['patient_name'])
    add_annotation.append(row['segment_box'])
    add_annotation.append(row['color_box'])
    add_annotation.append(aug)
    
    crop_image1 = crop(row['path_view1'][:-4]+' copie.tif',np.array(x1).astype(int))
    crop_image2 = crop(row['path_view2'][:-4]+' copie.tif',np.array(x2).astype(int))
    dim = (224, 224)

    # resize image
    crop_image1_resized = np.expand_dims(cv2.resize(crop_image1, dim, interpolation = cv2.INTER_LINEAR),axis=2)
    crop_image2_resized = np.expand_dims(cv2.resize(crop_image2, dim, interpolation = cv2.INTER_LINEAR),axis=2)

    image_input = np.concatenate((crop_image1_resized,crop_image2_resized),axis=2)
    
    label = row['target_box']
    
    save_path = os.path.join(image_dir,img_name )
    file_ = Path(save_path).with_suffix('.npz')
    file_.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(save_path),image= np.asarray(image_input),label =np.asarray(label))

    return add_annotation

def crop_images_save_info_stenosis(image_dir,img_name, row, ):
    add_annotation = [img_name]
    add_annotation.append(row['path_box'])
    add_annotation.append(row['target_box'])
    add_annotation.append(row['patient_name'])
    add_annotation.append(row['segment_box'])
    add_annotation.append(row['view_box'])
    add_annotation.append(row['color_box'])

    if 'copie' in row['path_box']:
        crop_image = crop(row['path_box'][:-12]+' copie.tif',np.array(row['stenosis_patches']).astype(int))
    else :
        crop_image = crop(row['path_box'][:-4]+' copie.tif',np.array(row['stenosis_patches']).astype(int))

    # resize image
    crop_image_resized = np.expand_dims(crop_image,axis=2)
    
    image_input = np.repeat(crop_image_resized, 3, axis=2)
    
    label = row['target_box']
    
    save_path = os.path.join(image_dir,img_name )
    file_ = Path(save_path).with_suffix('.npz')
    file_.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(save_path),image= np.asarray(image_input),label =np.asarray(label))

    return add_annotation

def crop_images2(image_dir, df_image_info_path, time_aug_MI = 12):
    """
    In this version of box extraction we select images with different view from a file that have regroup 
    box by view. The goals is to limit the impact of artefact in the images. 
    Box are then resize to a 224,224 images. 
    """
    df_image_info = pd.read_pickle(df_image_info_path)
    line = [] # img_id, path_img1, path_img2, target, patient, segment, color, aug 
    col_names = ['img_id', 'path_img1', 'path_img2', 'target', 'patient', 'segment', 'color', 'aug']

    patient_per_folder = {}
    list_dir = ['Farhang MI with or without revasc','Farhang MI without revasc','Farhang NCL','Farhang NCL 2',
                'Farhang NCL 3','Farhang NCL 4', 'Farhang NCL 5_low_quality']

    data_dir = os.path.join('/data','cardio/SPUM/Labelling_EPFL_CHUV')

    for folder in list_dir:
        patient_per_folder[folder] = os.listdir(os.path.join(data_dir,folder))

    img_idx = -1
    for index, row in df_image_info.iterrows():
        
        print(index)
        img_idx += 1
        img_name = 'img_%d.npz'%(img_idx)
        
        # following code fix issues with path name may be cause by accent in some patient name
        split_path = row['path_view1'].split('/')
        img_which = split_path[-1]
        img_patient = [ name for name in patient_per_folder[split_path[-3]] if name.startswith( split_path[-2].split('.')[0] ) ][0]
        row['path_view1'] = osp.join(data_dir,split_path[-3],img_patient,img_which)

        split_path = row['path_view2'].split('/')
        img_which = split_path[-1]
        img_patient = [ name for name in patient_per_folder[split_path[-3]] if name.startswith( split_path[-2].split('.')[0] ) ][0]
        row['path_view2'] = osp.join(data_dir,split_path[-3],img_patient,img_which)

        if not os.path.isfile(row['path_view1'][:-4]+' copie.tif'):
            continue
        add_annotation = crop_images_save_info(image_dir, img_name, row, row['coord_view1'], row['coord_view2'], aug='no')
        line.append(add_annotation)

        if row['target_box']  == 1 :
            # Moving the box slightly in order to move the box around
            X_1 = np.array(row['coord_view1'])
            X_2 = np.array(row['coord_view2'])

            for i in range(time_aug_MI):

                x_pert = np.random.randint(-25, 25)
                y_pert = np.random.randint(-25, 25)
                
                img_idx += 1
                img_name = 'img_%d.npz'%(img_idx)
                
                x1 = X_1.copy()
                x2 = X_2.copy()
                x1[:, 0] = x1[:, 0] + x_pert
                x1[:, 1] = x1[:, 1] + y_pert
                x2[:, 0] = x2[:, 0] + x_pert
                x2[:, 1] = x2[:, 1] + y_pert
                add_annotation = crop_images_save_info(image_dir, img_name, row, x1, x2, aug='translat_'+str(i+1))
                line.append(add_annotation)

    df_annotated_box = pd.DataFrame(line , columns = col_names)
    df_annotated_box.to_pickle(os.path.join(image_dir,'annotated_box' ),protocol = 4)

def crop_images3(image_dir, df_image_info):
    """
    In this version of box extraction, simply resize the box and 3 similar chanel according to the 
    input of ResNet for imagnet. 
    """
    image_file_name = []
    image_file_label = []

    for index, row in df_image_info.iterrows():
        
        crop_image = crop(row['path_box'][:-4]+' copie.tif',np.array(row['coordinate_box']).astype(int))
        dim = (224, 224)
 
        # resize image
        crop_image_resized = np.expand_dims(cv2.resize(crop_image, dim, interpolation = cv2.INTER_LINEAR),axis=2)
        
        image_input = np.repeat(crop_image_resized, 3, axis=2)
        
        file_to_save = row['patient_name']+'_'+row['segment_box']+'_'+row['color_box']+'_'+ str(row['view_box']) +'.npz'
        label = row['target_box']

        image_file_name.append(file_to_save) 
        image_file_label.append(label)
        
        save_path = os.path.join(image_dir,file_to_save)
        file_ = Path(save_path).with_suffix('.npz')
        file_.parent.mkdir(parents=True,exist_ok=True)
        np.savez(str(save_path),image= np.asarray(image_input),label =np.asarray(label))
    
    df_annotated_box = pd.DataFrame(list(zip(image_file_name,image_file_label)),columns = ['file','label'])
    df_annotated_box.to_pickle(os.path.join(image_dir,'annotated_box' ),protocol = 4)

def crop_images4(image_dir, df_image_info_path):
    """
    In this version of box extraction, we extract patches for the patches stenosis, we crop and add 3 similar channel for the model
    """
    df_image_info = pd.read_pickle(df_image_info_path)
   
    line = [ ] # img_name, path_box, target_box, patient_name, segment_box, view_box, color_box
    col_names = ['img_id', 'path', 'target', 'patient', 'segment','view', 'color']

    patient_per_folder = {}
    list_dir = ['Farhang MI with or without revasc','Farhang MI with revasc 2nd annotation','Farhang MI without revasc','Farhang NCL','Farhang NCL 2',
                'Farhang NCL 3','Farhang NCL 4', 'Farhang NCL 5_low_quality']

    data_dir = os.path.join('/data','cardio/SPUM/Labelling_EPFL_CHUV')

    for folder in list_dir:
        patient_per_folder[folder] = os.listdir(os.path.join(data_dir,folder))

    img_idx = -1 
    for index, row in df_image_info.iterrows():
        
        print(index)

        # following code fix issues with path name may be cause by accent in some patient name
        split_path = row['path_box'].split('/')
        img_which = split_path[-1]
        img_patient = [ name for name in patient_per_folder[split_path[-3]] if name.startswith( split_path[-2].split('.')[0] ) ][0]
        row['path_box'] = osp.join(data_dir,split_path[-3],img_patient,img_which)

    
        if 'copie' in row['path_box']:
            path = row['path_box'][:-12]+' copie.tif'
        else : 
            path = row['path_box'][:-4]+' copie.tif'
        
        img = io.imread(path)
        if len(img.shape) == 1: 
            continue

        img_idx += 1
        img_name = 'img_%d.npz'%(img_idx)
        add_annotation = crop_images_save_info_stenosis(image_dir, img_name, row)
        line.append(add_annotation)

    df_annotated_box = pd.DataFrame(line , columns = col_names)
    df_annotated_box.to_pickle(os.path.join(image_dir,'annotated_box' ),protocol = 4)

def aggregat_info(data_fram_list,image_info,col_names):
    """
    Aggregate the information of each images to create a pandas dataframe.
    """
    for i in range(len(image_info['path_box'])):
        line = []
        for col_name in col_names :
            line.append(image_info[col_name][i])
        data_fram_list.append(line)
