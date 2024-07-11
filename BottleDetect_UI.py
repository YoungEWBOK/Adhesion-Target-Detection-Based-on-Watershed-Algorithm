import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

class ImageProcessor:
    @staticmethod
    def process_image(image):
        if image is None:
            print("Error: Input image is None.")
            return None, 0

        try:
            # 边缘保留滤波EPF 去噪
            blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)

            # 转成灰度图像
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            # 得到二值图像区间阈值
            ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 距离变换
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
            dist_output = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
            ret, surface = cv2.threshold(dist_output, 0.5*dist_output.max(), 255, cv2.THRESH_BINARY)

            # Marker labelling
            ret, markers = cv2.connectedComponents(np.uint8(surface))
            markers = markers + 1

            # 未知区域标记
            kernel = np.ones((3, 3), np.uint8)
            unknown = cv2.subtract(cv2.dilate(binary, kernel, iterations=1), np.uint8(surface))
            markers[unknown == 255] = 0

            # 分水岭算法分割
            markers = cv2.watershed(image, markers=markers)
            markers_8u = np.uint8(markers)

            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
                      (255,0,255), (0,255,255), (255,128,0), (255,0,128),
                      (128,255,0), (128,0,255), (255,128,128), (128,255,255)]

            areas = []
            for i in range(2, np.max(markers) + 1):
                mask = cv2.inRange(markers_8u, i, i)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    areas.append(cv2.contourArea(contours[0]))

            hist, bin_edges = np.histogram(areas, bins=20)
            most_common_bin = np.argmax(hist)
            standard_area = (bin_edges[most_common_bin] + bin_edges[most_common_bin + 1]) / 2
            area_threshold_low = standard_area * 0.7
            area_threshold_high = standard_area * 1.3

            object_count = 0
            for i in range(2, np.max(markers) + 1):
                mask = cv2.inRange(markers_8u, i, i)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    area = cv2.contourArea(contours[0])
                    if area_threshold_low <= area <= area_threshold_high:
                        object_count += 1
                    elif area > area_threshold_high:
                        num_objects = round(area / standard_area)
                        object_count += num_objects
                    
                    color = colors[(i-2)%len(colors)]
                    cv2.drawContours(image, contours, -1, color, -1)
                    
                    M = cv2.moments(contours[0])
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.drawMarker(image, (cx,cy), (0,0,255), 1, 10, 2)
                        # 显示每个检测对象的面积，可自行选择
                        # cv2.putText(image, f"{area:.0f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(image, f"count={object_count}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

            return image, object_count

        except Exception as e:
            print(f"An error occurred during image processing: {e}")
            return None, 0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Object Counter')
        self.setGeometry(100, 100, 1200, 700)

        main_layout = QHBoxLayout()

        # Left side - buttons and info
        left_layout = QVBoxLayout()
        self.select_button = QPushButton('Select Image')
        self.select_button.clicked.connect(self.select_image)
        left_layout.addWidget(self.select_button)

        # 添加较小的垂直间距
        left_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.process_button = QPushButton('Process Image')
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        left_layout.addWidget(self.process_button)

        # 再添加一个较小的垂直间距
        left_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.info_label = QLabel('Select an image to start')
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

        left_layout.addStretch(1)

        # Middle - original image display
        self.original_image_label = QLabel('Original Image')
        self.original_image_label.setAlignment(Qt.AlignCenter)

        # Right - processed image display
        self.processed_image_label = QLabel('Processed Image')
        self.processed_image_label.setAlignment(Qt.AlignCenter)

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.original_image_label, 3)
        main_layout.addWidget(self.processed_image_label, 3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                font-size: 18px;
                margin: 4px 2px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel {
                font-size: 16px;
                border: 1px solid #dcdcdc;
                background-color: white;
            }
        """)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path = file_name
            self.display_image(file_name, self.original_image_label)
            self.process_button.setEnabled(True)
            self.info_label.setText('Image selected. Click "Process Image" to analyze.')

    def process_image(self):
        if hasattr(self, 'image_path'):
            image = cv2.imread(self.image_path)
            processed_image, count = ImageProcessor.process_image(image)
            if processed_image is not None:
                self.display_image(processed_image, self.processed_image_label)
                self.info_label.setText(f'Objects detected: {count}')
            else:
                self.info_label.setText('Error processing image')

    def display_image(self, image, label):
        if isinstance(image, str):
            pixmap = QPixmap(image)
        else:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont('Arial', 12))  # 设置全局字体
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())