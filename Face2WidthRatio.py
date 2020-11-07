import math
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import face_recognition
from cv2 import cv2
import matplotlib.pyplot as plt


def load_image(path):
    return face_recognition.load_image_file(path)


def get_FacePoints(points, method='average'):
    width_left, width_right = points[36], points[45]

    top_left = points[38]
    top_right = points[43]



    bottom_left, bottom_right = points[41], points[46]

    if method == 'left':
        coords = (width_left[0], width_right[0], top_left[1], bottom_left[1])

    elif method == 'right':
        coords = (width_left[0], width_right[0], top_right[1], bottom_right[1])

    else:
        top_average = int((top_left[1] + top_right[1]) / 2)
        bottom_average = int((bottom_left[1] + bottom_right[1]) / 2)
        coords = (width_left[0], width_right[0], top_average, bottom_average)

    

    return {'top_left': (coords[0], coords[2]),
            'bottom_left': (coords[0], coords[3]),
            'top_right': (coords[1], coords[2]),
            'bottom_right': (coords[1], coords[3])
            }


def good_PictureCheck(p, debug=False):
    ## To scale for picture size
    width_im = (p[16][0] - p[0][0]) / 100

    ## Difference in height between eyes
    eye_y_l = (p[37][1] + p[41][1]) / 2.0
    eye_y_r = (p[44][1] + p[46][1]) / 2.0
    eye_dif = (eye_y_r - eye_y_l) / width_im

    ## Difference top / bottom point nose
    nose_dif = (p[30][0] - p[27][0]) / width_im

    ## Space between face-edge to eye, left vs. right
    left_space = p[36][0] - p[0][0]
    right_space = p[16][0] - p[45][0]
    space_ratio = left_space / right_space

    if debug:
        print(eye_dif, nose_dif, space_ratio)

    ## These rules are not perfect, determined by trying a bunch of "bad" pictures
    if eye_dif > 5 or nose_dif > 3.5 or space_ratio > 3:
        return False
    else:
        return True


def FaceWidthHeightRatio_calc(corners):
    width = corners['top_right'][0] - corners['top_left'][0]
    height = corners['bottom_left'][1] - corners['top_left'][1]
    return float(width)


def show_box(image, corners):
    pil_image = Image.fromarray(image)

    ## Automatically determine width of the line depending on size of picture
   ## line_width = math.ceil(h/y)

    d = ImageDraw.Draw(pil_image)
    d.line([corners['bottom_left'], corners['top_left']], width=2)
    d.line([corners['bottom_left'], corners['bottom_right']], width=1)
    d.line([corners['top_left'], corners['top_right']], width=2)
    d.line([corners['top_right'], corners['bottom_right']], width=2)



    imshow(pil_image)
    plt.show(pil_image)



def get_FaceWidthHeightRatio(show=True, method='average'):
    image = face_recognition.load_image_file(r"C:\Users\Sundar Gopal\PycharmProjects\untitled.png")
    landmarks = face_recognition.api._raw_face_landmarks(image)
    landmarks_as_tuples = [(p.x, p.y) for p in landmarks[0].parts()]

    if good_PictureCheck(landmarks_as_tuples):
        corners = get_FacePoints(landmarks_as_tuples, method=method)
        fwh_ratio = FaceWidthHeightRatio_calc(corners)

        if show:
            print('The Facial-Width-Height ratio is: {}'.format(fwh_ratio))
            show_box(image, corners)
        else:
            return fwh_ratio
    else:
        if show:
            print("Picture is not suitable to calculate fwhr.")
            imshow(image)
        else:
            return None



get_FaceWidthHeightRatio()
