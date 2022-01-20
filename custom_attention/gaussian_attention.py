import numpy as np
import cv2
from skimage import io
import os 
from pathlib import Path

def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def create_gaussian_attention(img,poly):

    rect = cv2.minAreaRect(poly.astype(int))
    box_out = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box_out = np.int0(box_out)

    Y = np.linspace(0, img.shape[0]-1 , img.shape[0])
    X = np.linspace(0, img.shape[1]-1 , img.shape[1])

    X, Y = np.meshgrid(X, Y)
    theta = np.radians(rect[2])
    rot = np.array(( (np.cos(theta), -np.sin(theta)),
                    (np.sin(theta),  np.cos(theta)) ))
    # Mean vector and covariance matrix
    mu = np.array([rect[0][0], rect[0][1]])
    Sigma = rot@np.array([[ (rect[1][0]/2)**2 , 0], [0,  (rect[1][1]/2)**2]])@rot.T
    print(mu)
    print([[ (rect[1][0]/2)**2 , 0], [0,  (rect[1][1]/2)**2]])
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    print(img.shape)
    print(pos.shape)
    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    assert Z.shape[0] == img.shape[0]
    assert Z.shape[1] == img.shape[1]

    return Z 

def save_image_file(dir_image, name, img, label):
    save_path = os.path.join(dir_image,name )
    #print('img', img.shape)
    cv2.imwrite(save_path, img)

def create_gaussian_attention_aug(img_id ,img_path,row,annotated_line,dir_img):

    #print(img_path)
    img = io.imread(img_path[:-4]+' copie.tif')
    if img.shape[-1] == 4 :
        img = img[:,:,:-1]
    
    poly = np.asarray(row['coordinate_box'])
    rect = cv2.minAreaRect(poly.astype(int))
    box_out = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box_out = np.int0(box_out)
    target = row['target_box']
    line = [ 'img_%d.jpg'%(img_id) ,img_path, row['target_box'], row['patient_name'],
                 row['segment_box'], row['color_box'],row['view_box'], 'no' ]

    # Create the mesh for the gaussian attention 
    Y = np.linspace(0, img.shape[0]-1 , img.shape[0])
    X = np.linspace(0, img.shape[1]-1 , img.shape[1])
    X, Y = np.meshgrid(X, Y)

    if target == 0 :
        # Define the rotation matrix for the covariace matrix 
        theta_original = np.radians(rect[2])
        rot = np.array(( (np.cos(theta_original), -np.sin(theta_original)),
                                    (np.sin(theta_original),  np.cos(theta_original)) ))

        # Mean vector and covariance matrix
        mu_original = np.array([rect[0][0], rect[0][1]])
        Sigma = rot@np.array([[ (rect[1][0]/2)**2 , 0], [0,  (rect[1][1]/2)**2]])@rot.T

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # The distribution on the variables X, Y packed into pos.
        Z = multivariate_gaussian(pos, mu_original, Sigma)
        Z = Z/np.max(Z)*255
        img_att = img.copy()
        img_att[:,:,1] = Z
        img_att[:,:,2] = 0

        assert Z.shape[0] == img_att.shape[0]
        assert Z.shape[1] == img_att.shape[1]
        
        dim = (1500,1500)
        img_att = cv2.resize(img_att, dim, interpolation = cv2.INTER_LINEAR)
        img_id += 1
        line[0] = 'img_%d.jpg'%(img_id)
        line[-1] = 'no'
        save_image_file(dir_img, line[0], img_att, target)
        annotated_line.append(line.copy())

    else :
        theta_original = np.radians(rect[2])
        mu_original = np.array([rect[0][0], rect[0][1]])
        thetas = np.radians([rect[2]-15, rect[2], rect[2]+15])
        mu_shifts = [[-15,15] ,[0,15] ,[15,15] ,[-15,0] ,[0,0] ,[15,0] , [-15,-15] ,[0,-15] ,[15,-15]]
        scales = [ 0.7 , 1.0 , 1.30]
        id_aug = 0
        for theta in thetas: 
            for mu_shift in mu_shifts:
                for scale in scales :
                    mu = mu_original + np.array(mu_shift)
                    if (mu == mu_original).all() and (theta == theta_original) and (scale == 1.0) :
                        aug = 'no'
                    else :
                        id_aug += 1
                        aug = 'aug_'+str(id_aug)
                    
                    rot = np.array(( (np.cos(theta), -np.sin(theta)),
                                    (np.sin(theta),  np.cos(theta)) ))
                    # Mean vector and covariance matrix
                    mu = np.array([rect[0][0], rect[0][1]])
                    Sigma = rot@np.array([[ (scale*rect[1][0]/2)**2 , 0], [0,  (scale*rect[1][1]/2)**2]])@rot.T

                    # Pack X and Y into a single 3-dimensional array
                    pos = np.empty(X.shape + (2,))
                    pos[:, :, 0] = X
                    pos[:, :, 1] = Y
                    # The distribution on the variables X, Y packed into pos.
                    Z = multivariate_gaussian(pos, mu, Sigma)
                    Z = Z/np.max(Z)*255
                    img_att = img.copy()
                    img_att[:,:,1] = Z
                    img_att[:,:,2] = 0
                    
                    assert Z.shape[0] == img_att.shape[0]
                    assert Z.shape[1] == img_att.shape[1]
                    
                    dim = (1500,1500)
                    img_att = cv2.resize(img_att, dim, interpolation = cv2.INTER_LINEAR)
                    img_id += 1
                    line[0] = 'img_%d.jpg'%(img_id)
                    line[-1] = aug
                    save_image_file(dir_img, line[0], img_att, target)
                    annotated_line.append(line.copy())
    return img_id

