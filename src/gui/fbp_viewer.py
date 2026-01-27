from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QSlider, QSizePolicy,
                              QRadioButton, QDialogButtonBox, QGroupBox, QHBoxLayout,
                              QSpinBox, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import os


class FBPResolutionDialog(QDialog):
    """FBP 重建解析度選擇對話框。"""

    def __init__(self, original_size, parent=None):
        """
        Args:
            original_size: 原始影像尺寸（高度、寬度）
        """
        super().__init__(parent)
        self.setWindowTitle("FBP Reconstruction Settings")
        self.setFixedSize(450, 500)

        # 統一 Dialog 外觀
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)
        # 檢查astra套件是否可用
        self.astra_available = self.check_astra()

        # 設定字體。
        font = QFont("Calibri", 12)
        self.setFont(font)

        self.selected_size = 128  # 預設值
        self.angle_interval = 1.0  # 預設角度間隔（度）

        # 主版面配置。
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 資訊標籤。
        info_label = QLabel(f"<b>Original Image Size:</b> {original_size[0]}×{original_size[1]}")
        info_label.setStyleSheet("font-family: Calibri; font-size: 14pt; padding: 8px;")
        layout.addWidget(info_label)

        # astra支援標籤。
        astra_status = "astra GPU acceleration available" if self.astra_available else "astra GPU acceleration not available"
        astra_label = QLabel(f"<b>Astra Status:</b> {astra_status}")
        astra_label.setStyleSheet("font-family: Calibri; font-size: 14pt; padding: 8px; color: green;" if self.astra_available else "font-family: Calibri; font-size: 14pt; padding: 8px; color: red;")
        layout.addWidget(astra_label)

        # 角度間隔群組。
        angle_group = QGroupBox("Angle Interval")
        angle_group.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: bold;")
        angle_layout = QHBoxLayout()
        angle_layout.setSpacing(10)

        angle_label = QLabel("Projection angle interval:")
        angle_label.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: normal;")

        self.angle_spinbox = QSpinBox()
        self.angle_spinbox.setMinimum(1)
        self.angle_spinbox.setMaximum(90)
        self.angle_spinbox.setValue(1)
        self.angle_spinbox.setSuffix(" degree(s)")
        self.angle_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        self.angle_spinbox.valueChanged.connect(self.set_angle_interval)

        angle_layout.addWidget(angle_label)
        angle_layout.addWidget(self.angle_spinbox)
        angle_layout.addStretch()
        angle_group.setLayout(angle_layout)
        layout.addWidget(angle_group)

        # 解析度選擇群組。
        group_box = QGroupBox("Select Reconstruction Resolution")
        group_box.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: bold;")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)

        # 單選按鈕。
        self.radio_128 = QRadioButton("B8: 128×128 (~5-15 seconds for CPU)")
        self.radio_256 = QRadioButton("B4: 256×256 (~1 minutes for CPU)")
        self.radio_512 = QRadioButton("B2: 512×512 (>10 minutes for CPU)")

        # 設定預設值。
        self.radio_128.setChecked(True)

        # 設定單選按鈕樣式。
        radio_style = "font-family: Calibri; font-size: 14pt; font-weight: normal; padding: 5px;"
        self.radio_128.setStyleSheet(radio_style)
        self.radio_256.setStyleSheet(radio_style)
        self.radio_512.setStyleSheet(radio_style)

        # 連接事件。
        self.radio_128.toggled.connect(lambda checked: checked and self.set_size(128))
        self.radio_256.toggled.connect(lambda checked: checked and self.set_size(256))
        self.radio_512.toggled.connect(lambda checked: checked and self.set_size(512))

        group_layout.addWidget(self.radio_128)
        group_layout.addWidget(self.radio_256)
        group_layout.addWidget(self.radio_512)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        # 警示標籤。
        warning_label = QLabel(
            "<i>⚠ Higher resolutions require more computation time and memory.</i>"
        )
        warning_label.setStyleSheet("font-family: Calibri; font-size: 13pt; color: #d35400; padding: 8px;")
        warning_label.setWordWrap(True)
        layout.addWidget(warning_label)

        # 按鈕。
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def check_astra(self):
        """檢查astra-toolbox套件是否安裝。"""
        try:
            import astra
            return True
        except ImportError:
            return False
        
    def get_astra_available(self):
        """取得astra套件可用狀態。"""
        return self.astra_available
    
    def set_size(self, size):
        """設定選取的重建尺寸。"""
        self.selected_size = size

    def set_angle_interval(self, value):
        """設定角度間隔。"""
        self.angle_interval = float(value)

    def get_size(self):
        """取得選取的重建尺寸。"""
        return self.selected_size

    def get_angle_interval(self):
        """取得角度間隔。"""
        return self.angle_interval


class FBPViewer(QDialog):
    def __init__(self, recon_images, parent=None):
        super().__init__(parent)
        self.sample_name = parent.context.sample_name
        self.recon_images = recon_images
        self.current_index = 0

        self.n_slices, self.height, self.width = recon_images.shape

        # 設定視窗屬性。
        self.setFixedSize(800, 800)

        # 統一 Dialog 外觀
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)
        # 設定字體。
        font = QFont("Calibri", 10)
        self.setFont(font)

        # 影像標籤。
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)

        # 資訊標籤。
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-family: Calibri; font-size: 10pt; color: #555; padding: 5px;")

        # 滑桿。
        self.slider = QSlider(Qt.Horizontal)
        # 統一滑桿樣式
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
        self.slider.setStyleSheet(slider_style)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_slices - 1)
        self.slider.valueChanged.connect(self.update_image)

        # 儲存按鈕。
        self.save_button = QPushButton("Save Reconstruction as TIF Files")
        self.save_button.setStyleSheet("font-family: Calibri; font-size: 12pt; padding: 8px;")
        self.save_button.clicked.connect(self.save_reconstruction)

        # 版面配置。
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.info_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.save_button)

        self.update_image(0)

    def resizeEvent(self, event):
        """處理視窗大小變更事件。"""
        super().resizeEvent(event)
        self.update_image(self.current_index)

    def update_image(self, index):
        """更新顯示影像與視窗標題。"""
        self.current_index = index
        img = self.recon_images[index]
        h, w = img.shape

        # 建立 QImage（不對原始資料進行插值）。
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        # 縮放以符合標籤大小並維持長寬比。
        # 顯示時使用 SmoothTransformation 以提升視覺品質。
        label_w = self.image_label.width()
        label_h = self.image_label.height()

        if label_w > 0 and label_h > 0:
            scaled_pixmap = pixmap.scaled(
                label_w, label_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setPixmap(pixmap)

        # 更新視窗標題。
        self.setWindowTitle(
            f"FBP Reconstruction - {self.width}x{self.height} - Slice {index + 1}/{self.n_slices}"
        )

        # 更新資訊標籤。
        self.info_label.setText(
            f"Resolution: {self.width}x{self.height} | Slice: {index + 1}/{self.n_slices}"
        )

    def save_reconstruction(self):
        """將所有重建切片儲存為 TIF 檔。"""
        output_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Reconstruction", "", QFileDialog.ShowDirsOnly)

        if not output_dir:
            return

        try:
            for i in range(self.n_slices):
                filename = f"{self.sample_name}_{i+1:04d}.tif"
                filepath = os.path.join(output_dir, filename)

                img_data = self.recon_images[i]
                img_pil = Image.fromarray(img_data)
                img_pil.save(filepath)

            QMessageBox.information(
                self,
                "Save Complete",
                f"Successfully saved {self.n_slices} slices to:\n{output_dir}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save reconstruction:\n{str(e)}"
            )
