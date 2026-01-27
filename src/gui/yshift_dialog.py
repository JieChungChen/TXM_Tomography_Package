from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox, QPushButton
from PyQt5.QtCore import pyqtSignal


class ShiftDialog(QDialog):
    apply_shift = pyqtSignal(int)

    def __init__(self, image_height, parent=None):
        """
        Args:
            image_height: 影像高度 (pixels)
        """
        super().__init__(parent)
        self.setWindowTitle("Y-axis Shift")
        self.setFixedSize(400, 300)
        self.shift_amount = 50  # 預設值

        # 統一 Dialog 外觀
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)
        # 主版面配置。
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 資訊標籤。
        info_label = QLabel(
            "<b>Shift images vertically</b><br>"
            f"Image height: {image_height} pixels"
        )
        info_label.setStyleSheet("font-family: Calibri; font-size: 14pt; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 位移量輸入。
        shift_layout = QHBoxLayout()
        shift_layout.setSpacing(10)

        shift_label = QLabel("Shift amount (pixels):")
        shift_label.setStyleSheet("font-family: Calibri; font-size: 14pt;")

        self.shift_spinbox = QSpinBox()
        self.shift_spinbox.setMinimum(-image_height)
        self.shift_spinbox.setMaximum(image_height)
        self.shift_spinbox.setValue(50)
        self.shift_spinbox.setSuffix(" px")
        self.shift_spinbox.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        self.shift_spinbox.valueChanged.connect(self.set_shift_amount)

        shift_layout.addStretch()
        shift_layout.addWidget(shift_label)
        shift_layout.addWidget(self.shift_spinbox)
        shift_layout.addStretch()
        layout.addLayout(shift_layout)

        # 說明文字。
        desc_label = QLabel(
            "<i>Positive values shift down, negative values shift up.<br>"
            "The shift wraps around (np.roll behavior).</i>"
        )
        desc_label.setStyleSheet("font-family: Calibri; font-size: 12pt; color: #555; padding: 5px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Apply按鈕
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(lambda: self.apply_shift.emit(self.get_shift_amount()))
        layout.addWidget(apply_button)


    def set_shift_amount(self, value):
        self.shift_amount = value

    def get_shift_amount(self):
        return self.shift_amount