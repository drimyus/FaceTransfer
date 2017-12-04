import dlib
import os
import numpy as np
import cv2

from mishima_matte import mishima_matte


RIGHT_EYE_POINTS = list(range(36, 42))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_EYE_POINTS = list(range(42, 48))
LEFT_BROW_POINTS = list(range(22, 27))

LEFT_EYE_BROW = LEFT_EYE_POINTS + LEFT_BROW_POINTS
RIGHT_EYE_BROW = RIGHT_EYE_POINTS + RIGHT_BROW_POINTS

NUM_TOTAL_POINTS = 68


class Trans:

    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "model/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)

    def landmark_detect(self, img):

        rect = self.detector(img, 0)[0]

        shape = self.predictor(img, rect)

        coords = np.zeros((68, 2), dtype='int')
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
            cv2.circle(img, (coords[i][0], coords[i][1]), 2, (0, 0, 225), -1)

        return coords

    def left_eye_and_brow(self, img, landmarks):

        left_eye_brow = landmarks[LEFT_EYE_BROW]
        zip_pts = zip(*left_eye_brow)
        x0, y0, x1, y1 = (min(zip_pts[0]), min(zip_pts[1]), max(zip_pts[0]), max(zip_pts[1]))
        left_crop = img[y0:y1, x0:x1]

        avg_color_per_row = np.average(left_crop, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        color = np.array((int(avg_color[0]), int(avg_color[1]), int(avg_color[2])))

        # eye region fillout
        left_eye = landmarks[LEFT_EYE_POINTS]
        leftEyeHull = cv2.convexHull(left_eye)
        # cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.fillPoly(img, pts=[leftEyeHull], color=color)

        left_crop = img[y0:y1, x0:x1]
        cv2.imshow("l", left_crop)

        gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)

        equ = cv2.equalizeHist(gray)
        cv2.imshow("equ", equ)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
        cv2.imshow("thresh", thresh)

        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        cv2.imshow("dilate", dilate)

        trimap = cv2.addWeighted(thresh, 0.5, dilate, 0.5, 0)
        # trimap = cv2.cvtColor(trimap, cv2.COLOR_GRAY2BGR)
        cv2.imshow("trimap", trimap)

        return left_crop, trimap

    def right_eye_and_brow(self, img, landmarks):

        # right side eye and brow
        right_eye_brow = landmarks[RIGHT_EYE_BROW]
        zip_pts = zip(*right_eye_brow)
        x0, y0, x1, y1 = (min(zip_pts[0]), min(zip_pts[1]), max(zip_pts[0]), max(zip_pts[1]))
        right_crop = img[y0:y1, x0:x1]

        avg_color_per_row = np.average(right_crop, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        color = np.array((int(avg_color[0]), int(avg_color[1]), int(avg_color[2])))

        # eye region fillout
        right_eye = landmarks[RIGHT_EYE_POINTS]
        rightEyeHull = cv2.convexHull(right_eye)
        # cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.fillPoly(img, pts=[rightEyeHull], color=color)

        right_crop = img[y0:y1, x0:x1]
        return right_crop

    def run(self, img):

        landmarks = self.landmark_detect(img)

        l_crop, trimap = self.left_eye_and_brow(img, landmarks)
        alpha = mishima_matte(l_crop, trimap)
        cv2.imwrite("alpha.jpg", alpha)

        # r_crop = self.right_eye_and_brow(img, landmarks)

        # cv2.imshow("l", l_crop)
        # cv2.imshow("r", r_crop)
        cv2.waitKey(0)


if __name__ == '__main__':

    image = cv2.imread("data/beautiful-woman.jpg")
    trans = Trans().run(image)

