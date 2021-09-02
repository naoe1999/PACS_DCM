import cv2
import numpy as np

'''
362 260 318 305 316 307
183 276 167 323 166 325
'''


if __name__ == '__main__':

    img = np.zeros((512, 512, 3))

    p1 = (362, 260)
    p2 = (318, 305)
    p3 = (316, 307)
    bb1 = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    wh1 = (abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))

    cv2.circle(img, p1, 2, (0, 255, 255), -1)
    cv2.circle(img, p2, 2, (0, 255, 255), -1)
    cv2.circle(img, p3, 2, (0, 255, 255), -1)
    cv2.ellipse(img, (bb1, wh1, 0), (0, 255, 255), 1)


    p4 = (183, 276)
    p5 = (167, 323)
    p6 = (166, 325)
    bb2 = ((p4[0]+p5[0])//2, (p4[1]+p5[1])//2)
    wh2 = (abs(p4[0]-p5[0]), abs(p4[1]-p5[1]))

    cv2.circle(img, p4, 2, (255, 0, 255), -1)
    cv2.circle(img, p5, 2, (255, 0, 255), -1)
    cv2.circle(img, p6, 2, (255, 0, 255), -1)
    cv2.ellipse(img, (bb2, wh2, 0), (255, 0, 255), 1)

    # cv2.line(img, (362, 260), (318, 305), (255, 255, 255), 1)
    # cv2.line(img, (318, 305), (316, 307), (255, 255, 0), 1)

    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

