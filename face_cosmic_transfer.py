import dlib
import os
import cv2


class Trans:

    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "model/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)

    def landmark_detect(self, img):

        rect = self.detector(img, 0)[0]

        shape = self.predictor(img, rect)

        points = []
        for p in shape.parts():
            points.append([p.x, p.y])
            cv2.circle(img, (p.x, p.y), 2, (0, 0, 255), -1)

        cv2.imshow("img", img)
        cv2.waitKey()


if __name__ == '__main__':

    image = cv2.imread("data/beautiful-woman.jpg")
    trans = Trans().landmark_detect(image)
