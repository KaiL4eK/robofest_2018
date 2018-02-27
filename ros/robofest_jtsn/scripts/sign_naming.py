
from keras.models import Model, load_model, save_model

import sign_localize.rects as R
from sign_localize.net import *

import pickle
import sign_detect.ml_utils as mlu

try:
    xrange
except NameError:
    xrange = range

class SignDetector:
    def __init__(self, models):
        self.model_loc, self.model_det = models
        self.model_loc._make_predict_function()

    def process_naming(self, frame):
        input_img = preprocess_img(frame)
        mask = self.model_loc.predict(np.array([input_img]))[0] * 255.

        # mask = masks[:,:,0]
        mask = cv2.resize(mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
        im, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area    = 0
        pred    = None
        coords  = None

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)

            roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
            hog_features = mlu.get_hog_features(roi)
            y_pred = self.model_det.predict(hog_features.reshape(1, -1))[0]

            hull = cv2.convexHull(cnt)
            cv2.drawContours(frame, [hull], 0, (255, 255, 255), 4)

            if y_pred.title().lower() != 'negative':
                curr_area = cv2.contourArea(cnt)
                if curr_area > area:
                    pred = y_pred.title().lower()
                    area = curr_area
                    coords = (x,y,w,h)
        
        ### Circle estimation

        # for con in contours_area:
        #     perimeter = cv2.arcLength(con, True)
        #     area = cv2.contourArea(con)
        #     if perimeter == 0:
        #         break

        #     circularity = 4*math.pi*(area/(perimeter*perimeter))
        #     print(circularity)

        #     if 0.7 < circularity < 1.2:
        #         contours_cirles.append(con)

        ### ----------------------------------

        cv2.drawContours(frame, contours, -1, (255, 255, 0), 4)

        if pred is not None:
            x,y,w,h = coords
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
            cv2.putText(frame, pred, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)

        return pred

        # mask = masks[:,:,1]
        # mask = cv2.resize(mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
        # im, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

        # cv2.drawContours(frame, contours, -1, (0, 255, 255), 4)

