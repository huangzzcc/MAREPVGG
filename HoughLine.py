import cv2
import cv2 as cv
import numpy as np


def line_detection(image):  # 直线检测
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 120, 255, apertureSize=3)  # apertureSize是sobel算子窗口大小
    lines = cv.HoughLines(edges, 1, np.pi / 180, 300)  # 指定步长为1的半径和步长为π/180的角来搜索所有可能的直线
    # 将求得交点坐标反代换进行直线绘制
    for line in lines:
        #  print(type(lines))
        rho, theta = line[0]  # 获取极值ρ长度和θ角度
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho  # 获取x轴值
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  # 获取这条直线最大值点x1
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))  # 获取这条直线最小值点y2　　其中*1000是内部规则/
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 划线
    cv.imshow("image-lines", image)

def LSD_line_detection(image):  # LSD直线检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)
    # 将求得交点坐标反代换进行直线绘制
    for dline in lines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)

    cv2.imshow('image-lines', image)


src = cv.imread("huofu/qiangbi.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
line_detection(src.copy())
# LSD_line_detection(src.copy())
cv.waitKey(0)
cv.destroyAllWindows()
