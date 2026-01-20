from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QPushButton, QFileDialog, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import numpy as np
from PIL import Image


class MosaicPreviewDialog(QDialog):
    def __init__(self, mosaic_img, context):
        """
        拼接預覽對話框（含對比度調整）。

        Args:
            mosaic_img: 拼接影像（H, W）
            context: AppContext
        """
        super().__init__()
        self.mosaic_img = mosaic_img  # shape: (H, W)
        self.info = context
        self.metadata = context.metadata or {}
        self.clip_lower = 0.0  # 下限裁切百分比
        self.clip_upper = 0.5  # 上限裁切百分比

        # 取得拼接尺寸。
        self.height, self.width = mosaic_img.shape
        self.rows = self.metadata.get('mosaic_row', '?')
        self.cols = self.metadata.get('mosaic_column', '?')

        # 設定視窗屬性。
        self.setMinimumSize(800, 600)
        self.resize(1000, 800)

        # 設定字體。
        font = QFont("Calibri", 10)
        self.setFont(font)

        # 以拼接資訊更新視窗標題。
        self.update_window_title()

        self.img_8bit = None
        self.qimg = None

        # 影像標籤。
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 資訊標籤。
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-family: Calibri; font-size: 10pt; color: #555; padding: 5px;")
        self.update_info_label()

        # 下限裁切滑桿。
        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_slider.setMinimum(0)
        self.lower_slider.setMaximum(100)  # 0-10%
        self.lower_slider.setValue(int(self.clip_lower * 10))
        self.lower_slider.valueChanged.connect(self.on_slider_changed)

        # 上限裁切滑桿。
        self.upper_slider = QSlider(Qt.Horizontal)
        self.upper_slider.setMinimum(0)
        self.upper_slider.setMaximum(100)  # 0-10%
        self.upper_slider.setValue(int(self.clip_upper * 10))
        self.upper_slider.valueChanged.connect(self.on_slider_changed)

        # 滑桿更新去彈跳計時器（降低延遲）。
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_contrast_change)

        # 儲存按鈕。
        self.save_btn = QPushButton("Save Image")
        self.save_btn.setStyleSheet("font-family: Calibri; font-size: 10pt;")
        self.save_btn.clicked.connect(self.save_image)

        # 版面配置。
        layout = QVBoxLayout()
        layout.addWidget(self.img_label, stretch=1)
        layout.addWidget(self.info_label)

        # 對比度控制項。
        lower_box = QHBoxLayout()
        lower_label = QLabel("Lower Clip:")
        lower_label.setStyleSheet("font-family: Calibri; font-size: 10pt;")
        self.lower_value_label = QLabel("0.0%")
        self.lower_value_label.setStyleSheet("font-family: Calibri; font-size: 10pt; min-width: 50px;")
        lower_box.addWidget(lower_label)
        lower_box.addWidget(self.lower_slider)
        lower_box.addWidget(self.lower_value_label)

        upper_box = QHBoxLayout()
        upper_label = QLabel("Upper Clip:")
        upper_label.setStyleSheet("font-family: Calibri; font-size: 10pt;")
        self.upper_value_label = QLabel("0.5%")
        self.upper_value_label.setStyleSheet("font-family: Calibri; font-size: 10pt; min-width: 50px;")
        upper_box.addWidget(upper_label)
        upper_box.addWidget(self.upper_slider)
        upper_box.addWidget(self.upper_value_label)

        layout.addLayout(lower_box)
        layout.addLayout(upper_box)

        button_box = QHBoxLayout()
        button_box.addStretch()
        button_box.addWidget(self.save_btn)
        layout.addLayout(button_box)

        self.setLayout(layout)

        # 初始影像處理。
        self.update_image_data()
        self.update_display()

    def update_window_title(self):
        """以拼接尺寸更新視窗標題。"""
        title = f"Mosaic Preview - {self.width}×{self.height}"
        if self.rows != '?' and self.cols != '?':
            title += f" ({self.rows}×{self.cols} tiles)"
        self.setWindowTitle(title)

    def update_info_label(self):
        """更新拼接資訊標籤。"""
        info_text = f"Resolution: {self.width}×{self.height}"
        if self.rows != '?' and self.cols != '?':
            info_text += f" | Grid: {self.rows}×{self.cols} tiles"
        self.info_label.setText(info_text)

    def on_slider_changed(self):
        """處理滑桿數值變更（去彈跳）。"""
        self.pending_lower = self.lower_slider.value()
        self.pending_upper = self.upper_slider.value()

        # 立即更新標籤以保持回饋。
        self.lower_value_label.setText(f"{self.pending_lower / 10.0:.1f}%")
        self.upper_value_label.setText(f"{self.pending_upper / 10.0:.1f}%")

        # 去彈跳處理影像（降低延遲）。
        self.update_timer.start(200)  # 200ms delay

    def apply_contrast_change(self):
        """在去彈跳延遲後套用對比度變更。"""
        self.clip_lower = self.pending_lower / 10.0
        self.clip_upper = self.pending_upper / 10.0
        self.update_image_data()
        self.update_display()

    def update_image_data(self):
        """依目前裁切值更新 8 位元影像資料。"""
        vmin = np.percentile(self.mosaic_img, self.clip_lower)
        vmax = np.percentile(self.mosaic_img, 100 - self.clip_upper)

        # 避免除以零。
        if vmax == vmin:
            vmax = vmin + 1e-7

        # 正規化至 8 位元。
        normalized = (self.mosaic_img - vmin) / (vmax - vmin)
        self.img_8bit = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        h, w = self.img_8bit.shape
        self.qimg = QImage(self.img_8bit.data, w, h, w, QImage.Format_Grayscale8)

    def update_display(self):
        if self.qimg is None:
            return

        label_w = self.img_label.width()
        label_h = self.img_label.height()

        if label_w > 0 and label_h > 0:
            pixmap = QPixmap.fromImage(self.qimg)
            scaled_pixmap = pixmap.scaled(
                label_w, label_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.img_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def save_image(self):
        """儲存處理後的拼接影像。"""
        sample_name = f"{self.info.sample_name}.tif"
        filename, _ = QFileDialog.getSaveFileName(self, "Save mosaic", sample_name, "TIFF files (*.tif)")
        if filename:
            Image.fromarray(self.img_8bit).save(filename)
            QMessageBox.information(self, "Save Complete", f"Mosaic saved to:\n{filename}")
