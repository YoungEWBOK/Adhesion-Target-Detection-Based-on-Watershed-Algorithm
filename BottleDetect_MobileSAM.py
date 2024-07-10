import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, 
                             QFileDialog, QHBoxLayout, QSplitter, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QSize
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from skimage.morphology import label

class BottleCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()

    def initUI(self):
        self.setWindowTitle('Bottle Counter')
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                font-size: 18px;
                color: #333;
            }
        """)
        
        main_layout = QHBoxLayout()
        
        # Left panel
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        
        self.selectButton = QPushButton('Select Image')
        self.selectButton.setIcon(QIcon('icon_folder.png'))  # Add an icon (you need to have this image file)
        self.selectButton.setIconSize(QSize(24, 24))
        self.selectButton.clicked.connect(self.selectImage)
        left_layout.addWidget(self.selectButton)
        
        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setStyleSheet("border: 2px solid #ccc; border-radius: 10px;")
        left_layout.addWidget(self.imageLabel)
        
        # Right panel
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        
        self.resultLabel = QLabel('Number of bottles: ')
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.resultLabel.setFont(QFont('Arial', 20, QFont.Bold))
        right_layout.addWidget(self.resultLabel)
        
        self.processedImageLabel = QLabel()
        self.processedImageLabel.setAlignment(Qt.AlignCenter)
        self.processedImageLabel.setStyleSheet("border: 2px solid #ccc; border-radius: 10px;")
        right_layout.addWidget(self.processedImageLabel)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)
        
        self.setLayout(main_layout)
        self.resize(1000, 600)

    def loadModel(self):
        model_type = "vit_t"
        sam_checkpoint = "./weights/mobile_sam.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.mobile_sam.to(device=self.device)
        self.mobile_sam.eval()
        
        self.mask_generator = SamAutomaticMaskGenerator(
            self.mobile_sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

    def selectImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp)")
        if fileName:
            pixmap = QPixmap(fileName)
            self.imageLabel.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.processImage(fileName)

    def processImage(self, imagePath):
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = 512, int(512 * w / h)
        else:
            new_h, new_w = int(512 * h / w), 512
        image = cv2.resize(image, (new_w, new_h))
        
        masks = self.mask_generator.generate(image)
        
        combined_mask = np.zeros((new_h, new_w), dtype=bool)
        total_area = new_h * new_w
        
        for mask in masks:
            mask_area = mask['area']
            mask_image = mask['segmentation']
            
            if 0.001 * total_area < mask_area < 0.1 * total_area:
                if np.mean(image[mask_image]) > 30:
                    combined_mask |= mask_image

        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        labeled_mask = label(combined_mask)
        
        bottle_count = np.max(labeled_mask)
        self.resultLabel.setText(f'Number of bottles: {bottle_count}')

        for i in range(1, bottle_count + 1):
            bottle_mask = (labeled_mask == i).astype(np.uint8)
            contours, _ = cv2.findContours(bottle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        h, w, c = image.shape
        bytesPerLine = 3 * w
        qImg = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.processedImageLabel.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BottleCounterApp()
    ex.show()
    sys.exit(app.exec_())