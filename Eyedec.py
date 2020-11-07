import cv2 
import math 
import numpy as np from scipy.spatial 
import distance as dist

            def midpoint(ptA, ptB):
                return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

            def calculateDistance(ex,ey,ew,eh):
                 dist = math.sqrt((ex - ew)**2 + (ey - eh)**2)
                 return dist
            # print calculateDistance(x1, y1, x2, y2)

            # using harcasde  for face
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

            # read both the images of the face and the glasses
            image = cv2.imread(r"C:\Users\Sundar Gopal\PycharmProjects\images11MPAYP5.jpg")

            gray    =   cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            centers=[]
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            #check for the face detected
            for (x,y,w,h) in faces:

                #create two Regions of Interest on face.
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                # Store the cordinates of eyes in the image to the 'center' array
                for (ex,ey,ew,eh) in eyes:
                    centers.append((x+int(ex+0.5*ew), y+int(ey+0.5*eh)))
                    # Point2f eye1, eye2;
                    # double res = cv::norm(eye1-eye2);

                    #creates rectangle with 'colour'
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


            overlay_img = np.ones(image.shape,np.uint8)*255

            #Create a mask and generate it's inverse.
            gray_glasses    =   cv2.cvtColor(overlay_img,   cv2.COLOR_BGR2GRAY)
            ret,    mask    =   cv2.threshold(gray_glasses, 110,    255,    cv2.THRESH_BINARY)
            mask_inv    =   cv2.bitwise_not(mask)
            temp    =   cv2.bitwise_and(image,  image,  mask=mask)
            temp2   =   cv2.bitwise_and(overlay_img,    overlay_img,    mask=mask_inv)
            final_img   =   cv2.add(temp,   temp2)

            # imS = cv2.resize(final_img, (1366, 768))
            # print calculateDistance(ex,ey,ew,eh)
            cv2.imshow('Final  Result', final_img)
            cv2.waitKey()
            cv2.destroyAllWindows()