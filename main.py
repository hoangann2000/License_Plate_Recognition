import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
import cv2,imutils
import numpy as np
import time
import pyshine as ps
from lib_detection import load_model, detect_lp, im2single
import li_layout
import datetime

img_path = None
tmp = None
Ivehicle = None
# binary = None
# tmp_binary = None

# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')


class MainWindow(QtWidgets.QFrame, li_layout.Ui_Frame):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btn_chonanh.clicked.connect(self.loadImage)
        self.btn_nhandang.clicked.connect(self.setThreshold)
        self.btn_info.clicked.connect(self.info)


    def showtime(self):
        while True:
            QApplication.processEvents()
            dt = datetime.datetime.now()
            self.let_ngay.setText('%s-%s-%s' %(dt.day, dt.month, dt.year))
            self.let_gio.setText('%s:%s:%s' % (dt.hour, dt.minute, dt.second))

    def loadImage(self):

        self.img_path = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.Ivehicle = cv2.imread(self.img_path)

        # nguyen goc
        self.cv2_path = cv2.imread(self.img_path)

        self.setPhoto()

    def setPhoto(self):
        # self.tmp = self.Ivehicle
        self.Ivehicle = imutils.resize(self.Ivehicle,width=301,height=291)
        frame = cv2.cvtColor(self.Ivehicle, cv2.COLOR_BGR2RGB)
        self.Ivehicle = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.original_img.setPixmap(QtGui.QPixmap.fromImage(self.Ivehicle))


    def setThreshold(self):
        # self.lbl_threshold.setText('cái này là hiện threshold')
        # self.lbl_contour.setText('cái này là hiện contour')
        # self.lbl_result.setText('cái này là hiện kết quả')

        # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
        ratio = float(max(self.cv2_path.shape[:2])) / min(self.cv2_path.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)

        _, LpImg, lp_type = detect_lp(wpod_net, im2single(self.cv2_path), bound_dim, lp_threshold=0.5)

        if (len(LpImg)):

            # Chuyen doi anh bien so
            LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

            self.roi = LpImg[0]

            # Chuyen anh bien so ve gray
            gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)

            # Ap dung threshold de phan tach so va nen
            self.binary = cv2.threshold(gray, 127, 255,
                                   cv2.THRESH_BINARY_INV)[1]
            # Segment kí tự
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(self.binary, cv2.MORPH_DILATE,kernel3)
            cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            plate_info = ""

            self.binary = imutils.resize(self.binary, width=381, height=161)
            frame = cv2.cvtColor(self.binary, cv2.COLOR_BGR2RGB)
            self.binary = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.lbl_threshold.setPixmap(QtGui.QPixmap.fromImage(self.binary))

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                if 1.5 <= ratio <= 3.5:  # Chon cac contour dam bao ve ratio w/h
                    if h / self.roi.shape[0] >= 0.6:  # Chon cac contour cao tu 60% bien so tro len

                        # Ve khung chu nhat quanh so
                        cv2.rectangle(self.roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Tach so va predict
                        curr_num = thre_mor[y:y + h, x:x + w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                        curr_num = np.array(curr_num, dtype=np.float32)
                        curr_num = curr_num.reshape(-1, digit_w * digit_h)

                        # Dua vao model SVM
                        result = model_svm.predict(curr_num)[1]
                        result = int(result[0, 0])

                        if result <= 9:  # Neu la so thi hien thi luon
                            result = str(result)
                        else:  # Neu la chu thi chuyen bang ASCII
                            result = chr(result)

                        plate_info += result

            self.roi = imutils.resize(self.roi, width=381, height=161)
            frame = cv2.cvtColor(self.roi, cv2.COLOR_BGR2RGB)
            self.roi = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.lbl_contour.setPixmap(QtGui.QPixmap.fromImage(self.roi))

            # Viet bien so len anh
            cv2.putText(self.cv2_path, fine_tune(plate_info), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255),
                        lineType=cv2.LINE_AA)

            # Hien thi anh
            print("Bien so=", plate_info)

            self.cv2_path = imutils.resize(self.cv2_path, width=711, height=481)
            frame = cv2.cvtColor(self.cv2_path, cv2.COLOR_BGR2RGB)
            self.cv2_path = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.lbl_result.setPixmap(QtGui.QPixmap.fromImage(self.cv2_path))

            self.let_bienso.setText(fine_tune(plate_info))

    def info(self):
        in4 = self.let_bienso.text()
        in5 = int(in4[0:2])
        self.let_ten.setText('Hoàng Lê Thiện An')
        lang = {
            11: 'Cao Bằng',12: 'Lạng Sơn',14: 'Quảng Ninh',15: 'Hải Phòng',17: 'Thái Bình',18: 'Nam Định',
            19: 'Phú Thọ',20: 'Thái Nguyên',21: 'Yên Bái',22: 'Tuyên Quang',23: 'Hà Giang',24: 'Lao Cai',
            25: 'Lai Châu',26: 'Sơn La',27: 'Điện Biên',28: 'Hoà Bình',29: 'Hà Nội',30: 'Hà Nội',31: 'Hà Nội',
            32: 'Hà Nội',33: 'Hà Nội',40: 'Hà Nội',34: 'Hải Dương',35: 'Ninh Bình',36: 'Thanh Hóa',37: 'Nghệ An',
            38: 'Hà Tĩnh',43: 'Đà Nẵng',47: 'Dak Lak',48: 'Đắc Nông',49: 'Lâm Đồng',50: 'HCM',51: 'HCM',52: 'HCM',
            53: 'HCM',54: 'HCM',55: 'HCM',56: 'HCM',57: 'HCM',58: 'HCM',59: 'HCM',60: 'Đồng Nai',61: 'Bình Dương',
            62: 'Long An',63: 'Tiền Giang',64: 'Vĩnh Long',65: 'Cần Thơ',66: 'Đồng Tháp',67: 'An Giang',68: 'Kiên Giang',
            69: 'Cà Mau',70: 'Tây Ninh',71: 'Bến Tre',72: 'Vũng Tàu',73: 'Quảng Bình',74: 'Quảng Trị',75: 'Huế',
            76: 'Quảng Ngãi',77: 'Bình Định',78: 'Phú Yên',79: 'Nha Trang',81: 'Gia Lai',82: 'Kon Tum',83: 'Sóc Trăng',
            84: 'Trà Vinh',85: 'Ninh Thuận',86: 'Bình Thuận',88: 'Vĩnh Phúc',89: 'Hưng Yên',90: 'Hà Nam',92: 'Quảng Nam',
            93: 'Bình Phước',94: 'Bạc Liêu',95: 'Hậu Giang',97: 'Bắc Cạn',98: 'Bắc Giang',99: 'Bắc Ninh',
        }

        for name, code in lang.items():
            if in5 == name:
                self.let_tinh.setText(code)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = MainWindow()
    widget.show()
    widget.showtime()
    try:
        sys.exit(app.exec_())
    except (SystemError, SystemExit):
        app.exit()

