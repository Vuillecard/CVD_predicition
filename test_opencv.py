import cv2
import numpy as np
import matplotlib.pyplot as plt

print(cv2.__version__)

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
    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
    print(theta)
    change_theta = False
    neg_teta=True
    if change_theta :
        if h >= w:
            h, w = w, h
            theta = theta - 90
        if theta < -90.0:
            theta = theta + 180
        elif theta > 90.0:
            theta = theta - 180
    if neg_teta :
        theta = -theta

    box_out = cv2.boxPoints(rect1) # cv2.boxPoints(rect) for OpenCV 3.x
    box_out = np.int0(box_out)
    
    
    angle_c2v = theta
    
    if plots :
        im = np.zeros((500,500,3))
        #plt.figure(figsize=(5,5))
          
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
        image = cv2.putText(im, 'Angle '+str(angle_c2v), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        for point in box_rot.T:
            cv2.circle(im, (point[0],point[1]), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.drawContours(im,[box_out],0,(255),2)
        cv2.imshow('' , im)
        cv2.waitKey(1000)
        #plt.scatter(box_rot[0],box_rot[1])
        #plt.xlim(-2,2)
        #plt.ylim(-2,2)
      