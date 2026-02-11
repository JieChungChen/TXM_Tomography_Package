from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
                              QDialogButtonBox, QGroupBox, QHBoxLayout, QCheckBox)
from PyQt5.QtGui import QFont


class MLEMSettingsDialog(QDialog):
    """ML-EM 重建設定對話框。"""

    def __init__(self, parent=None, size=512):
        super().__init__(parent)
        self.setWindowTitle("ML-EM Reconstruction Settings")
        self.setFixedSize(450, 500)

        # 統一 Dialog 外觀
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)

        # 設定字體
        font = QFont("Calibri", 12)
        self.setFont(font)

        # 預設值
        self.iter_count = 100
        self.mask_ratio = 0.95
        self.image_size = size
        self.start_layer = 0
        self.end_layer = 100

        # 檢查 torch 和 astra 是否可用
        self.torch_available = self.check_torch()
        self.astra_available = self.check_astra()

        # 主版面配置
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 顯示 Image Size
        info_layout = QHBoxLayout()
        size_label = QLabel(f"<b>Original Image Size:</b> {self.image_size}×{self.image_size}")
        size_label.setStyleSheet("font-family: Calibri; font-size: 14pt; padding: 8px;")
        info_layout.addWidget(size_label)

        self.inverse_checkbox = QCheckBox("Inverse")
        self.inverse_checkbox.setStyleSheet("font-family: Calibri; font-size: 14pt; padding: 8px;")
        self.inverse_checkbox.setChecked(False) 
        info_layout.addWidget(self.inverse_checkbox)
        info_layout.addStretch() 

        layout.addLayout(info_layout)

        # 顯示 torch 和 astra 狀態
        status_group = QGroupBox("Library Status")
        status_group.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: bold;")
        status_layout = QHBoxLayout()
        status_layout.setSpacing(10)

        torch_status = "available" if self.torch_available else "not available"
        torch_label = QLabel(f"<b>Torch:</b> {torch_status}")
        torch_label.setStyleSheet("font-family: Calibri; font-size: 14pt; color: green;" if self.torch_available else "font-family: Calibri; font-size: 14pt; color: red;")
        
        astra_status = "available" if self.astra_available else "not available"
        astra_label = QLabel(f"<b>Astra:</b> {astra_status}")
        astra_label.setStyleSheet("font-family: Calibri; font-size: 14pt; color: green;" if self.astra_available else "font-family: Calibri; font-size: 14pt; color: red;")

        status_layout.addWidget(torch_label)
        status_layout.addWidget(astra_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 迭代次數
        iter_group = QGroupBox("Iteration Count")
        iter_group.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: bold;")
        iter_layout = QHBoxLayout()
        iter_layout.setSpacing(10)
        iter_label = QLabel("Number of iterations:")
        iter_label.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: normal;")
        self.iter_spinbox = QSpinBox()
        self.iter_spinbox.setMinimum(1)
        self.iter_spinbox.setMaximum(500)
        self.iter_spinbox.setValue(self.iter_count)
        self.iter_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self.iter_spinbox)
        iter_group.setLayout(iter_layout)
        layout.addWidget(iter_group)

        # 遮罩比例
        mask_group = QGroupBox("Mask Ratio")
        mask_group.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: bold;")
        mask_layout = QHBoxLayout()
        mask_layout.setSpacing(10)
        mask_label = QLabel("Mask ratio (0-1):")
        mask_label.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: normal;")
        self.mask_spinbox = QDoubleSpinBox()
        self.mask_spinbox.setMinimum(0.1)
        self.mask_spinbox.setMaximum(1.0)
        self.mask_spinbox.setSingleStep(0.01)
        self.mask_spinbox.setValue(self.mask_ratio)
        self.mask_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        mask_layout.addWidget(mask_label)
        mask_layout.addWidget(self.mask_spinbox)
        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)

        # 角度間隔
        angle_group = QGroupBox("Angle Interval")
        angle_group.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: bold;")
        angle_layout = QHBoxLayout()
        angle_layout.setSpacing(10)
        angle_label = QLabel("Projection angle interval:")
        angle_label.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: normal;")
        self.angle_spinbox = QSpinBox()
        self.angle_spinbox.setMinimum(1)
        self.angle_spinbox.setMaximum(5)
        self.angle_spinbox.setValue(1)  # 預設值
        self.angle_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        angle_layout.addWidget(angle_label)
        angle_layout.addWidget(self.angle_spinbox)
        angle_group.setLayout(angle_layout)
        layout.addWidget(angle_group)

        # 層範圍
        layer_group = QGroupBox("Layer Range")
        layer_group.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: bold;")
        layer_layout = QHBoxLayout()
        layer_layout.setSpacing(10)
        start_label = QLabel("Start layer:")
        start_label.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: normal;")
        self.start_spinbox = QSpinBox()
        self.start_spinbox.setMinimum(0)
        self.start_spinbox.setMaximum(size-1)
        self.start_spinbox.setValue(self.start_layer)
        self.start_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        end_label = QLabel("End layer:")
        end_label.setStyleSheet("font-family: Calibri; font-size: 14pt; font-weight: normal;")
        self.end_spinbox = QSpinBox()
        self.end_spinbox.setMinimum(0)
        self.end_spinbox.setMaximum(size-1)
        self.end_spinbox.setValue(self.end_layer)
        self.end_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        layer_layout.addWidget(start_label)
        layer_layout.addWidget(self.start_spinbox)
        layer_layout.addWidget(end_label)
        layer_layout.addWidget(self.end_spinbox)
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)

        # 按鈕
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def check_torch(self):
        """檢查 torch 套件是否安裝。"""
        try:
            import torch
            return True
        except ImportError:
            return False

    def check_astra(self):
        """檢查 astra-toolbox 套件是否安裝。"""
        try:
            import astra
            return True
        except ImportError:
            return False

    def get_settings(self):
        """取得設定值。"""
        return {
            "iter_count": self.iter_spinbox.value(),
            "mask_ratio": self.mask_spinbox.value(),
            "start_layer": self.start_spinbox.value(),
            "end_layer": self.end_spinbox.value(),
            "angle_interval": self.angle_spinbox.value(),
            "inverse": self.inverse_checkbox.isChecked()
        }