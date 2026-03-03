from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QSlider, QSizePolicy,
                              QRadioButton, QDialogButtonBox, QGroupBox, QHBoxLayout,
                              QSpinBox, QPushButton, QFileDialog, QMessageBox, QCheckBox)
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
        self.setFixedSize(450, 650)

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
        info_layout = QHBoxLayout()
        info_label = QLabel(f"<b>Original Image Size:</b> {original_size[0]}×{original_size[1]}")
        info_label.setStyleSheet("font-family: Calibri; font-size: 14pt; padding: 8px;")
        info_layout.addWidget(info_label)

        self.inverse_checkbox = QCheckBox("Inverse")
        self.inverse_checkbox.setStyleSheet("font-family: Calibri; font-size: 14pt; padding: 8px;")
        info_layout.addWidget(self.inverse_checkbox)
        info_layout.addStretch()  # 讓checkbox靠右

        layout.addLayout(info_layout)

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
        # self.angle_spinbox.valueChanged.connect(self.set_angle_interval)

        angle_layout.addWidget(angle_label)
        angle_layout.addWidget(self.angle_spinbox)
        angle_layout.addStretch()
        angle_group.setLayout(angle_layout)
        layout.addWidget(angle_group)

        # 旋轉中心校正群組（可選）。
        self.correction_group = QGroupBox("Center Correction (optional)")
        self.correction_group.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: bold;")
        self.correction_group.setCheckable(True)
        self.correction_group.setChecked(False)
        correction_layout = QVBoxLayout()
        correction_layout.setSpacing(8)

        range_layout = QHBoxLayout()
        range_label = QLabel("Search Range (\u00b1):")
        range_label.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: normal;")
        self.correction_range_spinbox = QSpinBox()
        self.correction_range_spinbox.setMinimum(1)
        self.correction_range_spinbox.setMaximum(500)
        self.correction_range_spinbox.setValue(20)
        self.correction_range_spinbox.setSuffix(" pixels")
        self.correction_range_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        range_layout.addWidget(range_label)
        range_layout.addWidget(self.correction_range_spinbox)
        range_layout.addStretch()

        layer_layout = QHBoxLayout()
        layer_label = QLabel("Target Layer:")
        layer_label.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: normal;")
        self.correction_layer_spinbox = QSpinBox()
        self.correction_layer_spinbox.setMinimum(0)
        self.correction_layer_spinbox.setMaximum(original_size[0] - 1)
        self.correction_layer_spinbox.setValue(original_size[0] // 2)
        self.correction_layer_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        layer_layout.addWidget(layer_label)
        layer_layout.addWidget(self.correction_layer_spinbox)
        layer_layout.addStretch()

        correction_layout.addLayout(range_layout)
        correction_layout.addLayout(layer_layout)
        self.correction_group.setLayout(correction_layout)
        layout.addWidget(self.correction_group)

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
        
    def set_size(self, size):
        self.selected_size = size

    def get_settings(self):
        """取得設定值。"""
        return {
            "astra_available": self.astra_available,
            "target_size": self.selected_size,
            "angle_interval": self.angle_spinbox.value(),
            "inverse": self.inverse_checkbox.isChecked(),
            "center_shift": 0,
            "center_correction_enabled": self.correction_group.isChecked(),
            "correction_range": self.correction_range_spinbox.value(),
            "correction_layer": self.correction_layer_spinbox.value()
        }
    

class CenterCorrectionPreviewDialog(QDialog):
    """Center Correction 預覽對話框：預先計算所有 shift 值的重建結果，讓使用者以滑桿瀏覽並選定最佳值。"""

    def __init__(self, images, angles, target_size, correction_layer, correction_range,
                 angle_interval, astra_available, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Center Correction Preview")
        self.setFixedSize(650, 720)
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)
        font = QFont("Calibri", 11)
        self.setFont(font)

        import numpy as np

        # Resize images to target_size（與 FBPWorker 相同邏輯）。
        resized = []
        for img in images:
            pil = Image.fromarray(img)
            pil = pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
            resized.append(np.array(pil))
        resized = np.array(resized)  # (N, target_size, target_size)

        n, h, w = resized.shape
        orig_h = images.shape[1]
        scaled_layer = int(correction_layer * target_size / orig_h)
        scaled_layer = max(0, min(scaled_layer, h - 1))

        # 建立角度陣列（與 FBPWorker 相同邏輯）。
        n_images = len(images)
        if angles is None or len(angles) == 0:
            ang = np.arange(n_images) * angle_interval - 90.0
        else:
            ang = np.arange(n_images) * angle_interval + angles[0]

        # 預先計算所有 shift 值的重建結果。
        self.shifts = list(range(-correction_range, correction_range + 1))
        base_sino = resized[:, scaled_layer, :]
        self.recon_images = []

        _astra = astra_available
        if _astra:
            try:
                from recon_algorithms import recon_fbp_astra
                for shift in self.shifts:
                    sino = np.roll(base_sino, shift, axis=1)
                    r = recon_fbp_astra(sino, angle_interval=angle_interval, norm=False).astype(np.float32)
                    r -= r.min()
                    if r.max() > 0:
                        r /= r.max()
                    self.recon_images.append((r * 255).astype(np.uint8))
            except ImportError:
                _astra = False

        if not _astra:
            from src.logic.fbp import filter_back_projection_fast, prepare_fbp_geometry, get_hann_filter
            img_size_padded = max(64, 2 ** int(np.ceil(np.log2(2 * w))))
            hann = get_hann_filter(img_size_padded)
            center, x, y, cos_vals, sin_vals = prepare_fbp_geometry(w, ang)
            empty_sino = np.ones((n, w), dtype=np.float32)
            recon_0 = filter_back_projection_fast(empty_sino, cos_vals, sin_vals, center, x, y, hann)
            for shift in self.shifts:
                sino = np.roll(base_sino, shift, axis=1)
                r = filter_back_projection_fast(sino, cos_vals, sin_vals, center, x, y, hann, filtered=True, circle=False)
                r = r / recon_0
                r -= r.min()
                if r.max() > 0:
                    r /= r.max()
                self.recon_images.append((r * 255).astype(np.uint8))

        self.selected_shift = 0

        # ---- UI ----
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 影像顯示區域。
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background: #222; border-radius: 6px;")
        layout.addWidget(self.image_label, stretch=1)

        # Shift 數值標籤。
        self.shift_label = QLabel()
        self.shift_label.setAlignment(Qt.AlignCenter)
        self.shift_label.setStyleSheet("font-family: Calibri; font-size: 13pt; color: #333; padding: 4px;")
        layout.addWidget(self.shift_label)

        # 滑桿。
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet("""
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
        """)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.shifts) - 1)
        self.slider.setValue(correction_range)  # 預設指向 shift=0
        self.slider.valueChanged.connect(self.update_preview)
        layout.addWidget(self.slider)

        # 說明文字。
        hint_label = QLabel("<i>Drag the slider to browse different center shifts. Click OK to apply the selected shift for full 3D reconstruction.</i>")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("font-family: Calibri; font-size: 11pt; color: #888; padding: 4px;")
        layout.addWidget(hint_label)

        # 按鈕。
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.update_preview(correction_range)

    def update_preview(self, index):
        """更新預覽影像與 shift 資訊。"""
        self.selected_shift = self.shifts[index]
        img = self.recon_images[index]
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        if label_w > 0 and label_h > 0:
            scaled = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setPixmap(pixmap)
        sign = "+" if self.selected_shift > 0 else ""
        self.shift_label.setText(
            f"Center Shift: <b>{sign}{self.selected_shift}</b> pixels"
            f"  |  Index: {index + 1} / {len(self.shifts)}"
        )

    def get_selected_shift(self):
        """回傳使用者選定的 shift 值。"""
        return self.selected_shift


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
            # 取得 3D volume 的維度 (Z, Y, X)
            z_dim, y_dim, x_dim = self.recon_images.shape

            # 定義子資料夾路徑
            paths = {
                "XY": os.path.join(output_dir, "XY"),
                "YZ": os.path.join(output_dir, "YZ"),
                "XZ": os.path.join(output_dir, "XZ")
            }

            # 自動建立資料夾 (如果不存在的話)
            for folder_path in paths.values():
                os.makedirs(folder_path, exist_ok=True)

            # 1. 儲存 XY 切面 (橫切面，原本的邏輯)
            for i in range(z_dim):
                filename = f"{self.sample_name}_{i+1:04d}_XY.tif"
                filepath = os.path.join(paths["XY"], filename)

                img_data = self.recon_images[i, :, :]
                img_pil = Image.fromarray(img_data)
                img_pil.save(filepath)
            
            QMessageBox.information(
                self,
                "Save Complete",
                f"Successfully saved {z_dim} slices to:\n{paths['XY']}"
            )

            # 2. 儲存 YZ 切面 (矢狀面)
            # 取第 i 個 X 座標的所有 Y, Z
            for i in range(x_dim):
                filename = f"{self.sample_name}_{i+1:04d}_YZ.tif"
                filepath = os.path.join(paths["YZ"], filename)

                # 切片 [:, :, i] 取得 YZ 平面
                img_data = self.recon_images[:, :, i]
                Image.fromarray(img_data).save(filepath)

            QMessageBox.information(
                self,
                "Save Complete",
                f"Successfully saved {x_dim} slices to:\n{paths['YZ']}"
            )

            # 3. 儲存 XZ 切面 (冠狀面)
            # 取第 i 個 Y 座標的所有 X, Z
            for i in range(y_dim):
                filename = f"{self.sample_name}_{i+1:04d}_XZ.tif"
                filepath = os.path.join(paths["XZ"], filename)

                # 切片 [:, i, :] 取得 XZ 平面
                img_data = self.recon_images[:, i, :]
                Image.fromarray(img_data).save(filepath)
       
            QMessageBox.information(
                self,
                "Save Complete",
                f"Successfully saved {y_dim} slices to:\n{paths['XZ']}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save reconstruction:\n{str(e)}"
            )
