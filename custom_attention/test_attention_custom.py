import cv2
import numpy as np
import matplotlib.pyplot as plt

print(cv2.__version__)

def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos.

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

plots = True
#box = np.array([[1,0.5],[-1,0.5],[-1,-0.5],[1,-0.5]])
box = np.array([[0.5,1],[-0.5,1],[-0.5,-1],[0.5,-1]])
angles = 30*np.linspace(1,12,12)

for angle in angles :

    theta_ = np.radians(angle)

    r = np.array(( (np.cos(theta_), -np.sin(theta_)),
                (np.sin(theta_),  np.cos(theta_)) ))
    
   
    box_rot = (r@box.T*50+250).astype(int)
    print(box_rot.shape)
    rect1 = cv2.minAreaRect(box_rot.T)
    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], np.radians(rect1[2])

    box_out = cv2.boxPoints(rect1) # cv2.boxPoints(rect) for OpenCV 3.x
    box_out = np.int0(box_out)
    
    
    angle_c2v = theta
    
    if plots :
        im = np.ones((500,500,3))
        #plt.figure(figsize=(5,5))
        # Mean vector and covariance matrix
        X = np.linspace(0, 500 , 500)
        Y = np.linspace(0, 500 , 500)
        X, Y = np.meshgrid(X, Y)
        rot = np.array(( (np.cos(theta), -np.sin(theta)),
                (np.sin(theta),  np.cos(theta)) ))
        mu = np.array([x,y])
        Sigma = rot@np.array([[ (w/2)**2 , 0], [0,  (h/2)**2]])@rot.T
        print(Sigma)
        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        

        # The distribution on the variables X, Y packed into pos.
        Z = multivariate_gaussian(pos, mu, Sigma)
        print(np.expand_dims(Z, axis=2).shape)
        Z = np.repeat(np.expand_dims(Z, axis=2),3,2)
        Z = Z/np.max(Z)
        print(np.max(Z))
        print(Z.shape)
        print(im.shape)

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                im[i,j] = im[i,j]*Z[i,j]
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # org
        org = (50, 50)
        
        # fontScale
        fontScale = 1
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.putText() method
        image = cv2.putText(im, 'Angle '+str(rect1[2]), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        for point in box_rot.T:
            cv2.circle(im, (point[0],point[1]), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.drawContours(im,[box_out],0,(255),2)
        cv2.imshow('' , im)
        cv2.waitKey(1000)
        #plt.scatter(box_rot[0],box_rot[1])
        #plt.xlim(-2,2)
        #plt.ylim(-2,2)