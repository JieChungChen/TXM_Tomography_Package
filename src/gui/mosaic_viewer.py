from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QPushButton, QFileDialog, QSizePolicy, QMessageBox, QFrame
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
        self.setMinimumSize(900, 650)
        self.resize(1100, 820)

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
        self.img_label.setStyleSheet("border-radius: 8px; background-color: #0d0d0d;")

        # 資訊標籤。
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.info_label.setStyleSheet("font-family: Calibri; font-size: 11pt; color: #222; padding: 4px 0;")
        self.update_info_label()

        self.clip_range_label = QLabel()
        self.clip_range_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.clip_range_label.setStyleSheet("font-family: Calibri; font-size: 10pt; color: #555;")

        self.pending_lower = int(self.clip_lower * 10)
        self.pending_upper = int(self.clip_upper * 10)

        # 對比度滑桿。
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #bfbfbf;
                height: 6px;
                border-radius: 3px;
                background: #dedede;
            }
            QSlider::handle:horizontal {
                background: #1f6feb;
                border: none;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """

        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_slider.setMinimum(0)
        self.lower_slider.setMaximum(100)
        self.lower_slider.setValue(self.pending_lower)
        self.lower_slider.setStyleSheet(slider_style)
        self.lower_slider.valueChanged.connect(self.on_slider_changed)

        self.upper_slider = QSlider(Qt.Horizontal)
        self.upper_slider.setMinimum(0)
        self.upper_slider.setMaximum(100)
        self.upper_slider.setValue(self.pending_upper)
        self.upper_slider.setStyleSheet(slider_style)
        self.upper_slider.valueChanged.connect(self.on_slider_changed)

        self.lower_value_label = QLabel()
        self.lower_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lower_value_label.setStyleSheet("font-family: Calibri; font-size: 11pt; min-width: 60px; color: #222;")

        self.upper_value_label = QLabel()
        self.upper_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.upper_value_label.setStyleSheet("font-family: Calibri; font-size: 11pt; min-width: 60px; color: #222;")

        # 滑桿更新去彈跳計時器（降低延遲）。
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_contrast_change)

        # 儲存按鈕。
        self.save_btn = QPushButton("Save Image")
        self.save_btn.setFixedWidth(140)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #1f6feb;
                border-radius: 6px;
                color: white;
                font-weight: 600;
                padding: 8px 0;
            }
            QPushButton:hover { background-color: #4b85ff; }
            QPushButton:pressed { background-color: #0b4ac6; }
        """)
        self.save_btn.clicked.connect(self.save_image)

        # 版面配置。
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        header_frame = QFrame()
        header_frame.setObjectName("header_frame")
        header_frame.setStyleSheet("QFrame#header_frame { border-bottom: 1px solid #d6d6d6; padding-bottom: 4px; }")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.info_label, stretch=2)
        header_layout.addWidget(self.clip_range_label, stretch=0)
        layout.addWidget(header_frame)

        layout.addWidget(self.img_label, stretch=1)

        contrast_frame = QFrame()
        contrast_frame.setObjectName("contrast_frame")
        contrast_frame.setStyleSheet("""
            QFrame#contrast_frame {
                border: 1px solid #e2e2e2;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fafafa, stop:1 #ffffff);
            }
        """)
        contrast_layout = QVBoxLayout(contrast_frame)
        contrast_layout.setContentsMargins(16, 12, 16, 14)
        contrast_layout.setSpacing(10)

        contrast_title = QLabel("Contrast Controls")
        contrast_title.setStyleSheet("font-family: Calibri; font-size: 13pt; font-weight: 600; color: #222;")
        contrast_layout.addWidget(contrast_title)

        contrast_desc = QLabel("Clip the darkest and brightest pixels to highlight subtle details.")
        contrast_desc.setStyleSheet("font-family: Calibri; font-size: 10pt; color: #555;")
        contrast_layout.addWidget(contrast_desc)

        lower_row = QHBoxLayout()
        lower_label = QLabel("Dark Clip")
        lower_label.setStyleSheet("font-family: Calibri; font-size: 11pt; color: #333;")
        lower_row.addWidget(lower_label)
        lower_row.addWidget(self.lower_slider, stretch=1)
        lower_row.addWidget(self.lower_value_label)
        contrast_layout.addLayout(lower_row)

        upper_row = QHBoxLayout()
        upper_label = QLabel("Bright Clip")
        upper_label.setStyleSheet("font-family: Calibri; font-size: 11pt; color: #333;")
        upper_row.addWidget(upper_label)
        upper_row.addWidget(self.upper_slider, stretch=1)
        upper_row.addWidget(self.upper_value_label)
        contrast_layout.addLayout(upper_row)

        layout.addWidget(contrast_frame)

        button_box = QHBoxLayout()
        button_box.addStretch()
        button_box.addWidget(self.save_btn)
        layout.addLayout(button_box)

        self.setLayout(layout)

        self.update_slider_value_labels(self.pending_lower / 10.0, self.pending_upper / 10.0)
        self.update_clip_summary()

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

    def update_clip_summary(self, lower=None, upper=None):
        """更新裁切範圍說明。"""
        lower_pct = lower if lower is not None else self.clip_lower
        upper_pct = upper if upper is not None else self.clip_upper
        self.clip_range_label.setText(
            f"Clipping {lower_pct:.1f}% darkest / {upper_pct:.1f}% brightest"
        )

    def update_slider_value_labels(self, lower_pct, upper_pct):
        self.lower_value_label.setText(f"{lower_pct:.1f}%")
        self.upper_value_label.setText(f"{upper_pct:.1f}%")

    def on_slider_changed(self):
        """處理滑桿數值變更（去彈跳）。"""
        self.pending_lower = self.lower_slider.value()
        self.pending_upper = self.upper_slider.value()

        lower_pct = self.pending_lower / 10.0
        upper_pct = self.pending_upper / 10.0
        self.update_slider_value_labels(lower_pct, upper_pct)
        self.update_clip_summary(lower_pct, upper_pct)

        # 去彈跳處理影像（降低延遲）。
        self.update_timer.start(200)  # 200ms delay

    def apply_contrast_change(self):
        """在去彈跳延遲後套用對比度變更。"""
        self.clip_lower = self.pending_lower / 10.0
        self.clip_upper = self.pending_upper / 10.0
        self.update_clip_summary()
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
