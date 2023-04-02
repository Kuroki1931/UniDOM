import cv2
from cv2 import aruco
import numpy as np
import time

class MarkSearch :

    ### --- aruco設定 --- ###
    dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    def __init__(self, cameraID):
        self.cap = cv2.VideoCapture(cameraID)

    def get_mark_coordinate(self, num_id):
        """
        静止画を取得し、所望のマークの座標を取得する
        """
        # ret, frame = self.cap.read()
        frame = cv2.imread('try1.jpg')
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        ### num_id のマーカーが検出された場合 ###
        if num_id in np.ravel(ids) :
            index = np.where(ids == num_id)[0][0] #num_id が格納されているindexを抽出
            cornerUL = corners[index][0][0]
            cornerUR = corners[index][0][1]
            cornerBR = corners[index][0][2]
            cornerBL = corners[index][0][3]

            center = [ (cornerUL[0]+cornerBR[0])/2 , (cornerUL[1]+cornerBR[1])/2 ]

            print('左上 : {}'.format(cornerUL))
            print('右上 : {}'.format(cornerUR))
            print('右下 : {}'.format(cornerBR))
            print('左下 : {}'.format(cornerBL))
            print('中心 : {}'.format(center))

            print(corners[index])

            return center

        return None


if __name__ == "__main__" :

    import cv2
    from cv2 import aruco
    import numpy as np
    import time

    ### --- aruco設定 --- ###
    dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    ### --- parameter --- ###
    cameraID = 0
    cam0_mark_search = MarkSearch(cameraID)

    markID = 6

    try:
        while True:
            print(' ----- get_mark_coordinate ----- ')
            print(cam0_mark_search.get_mark_coordinate(markID))
            time.sleep(0.5)
    except KeyboardInterrupt:
        cam0_mark_search.cap.release()