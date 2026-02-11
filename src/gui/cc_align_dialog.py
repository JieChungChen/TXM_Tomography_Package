import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QDialogButtonBox, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from src.logic.utils import common_line_method


class CCAlignDialog(QDialog):
    FONT_CTRL = QFont('Calibri', 14)
    FONT_TITLE = QFont('Calibri', 16, QFont.Bold)
    LABEL_STYLE = """
        QLabel {
            border: 2px solid #bfc7d5;
            border-radius: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #f5f7fa, stop:1 #c3cfe2);
            padding: 4px;
        }
    """
    def __init__(self, features, parent=None):
        """
        Parameters
        ----------
        features: np.ndarray
            Horizontal sum array of shape (N_projections, Height)
            Note: This should be passed as (N, H), not (H, N).
        """
        super().__init__(parent)
        self.setWindowTitle("Auto Alignment (Cross Correlation)")
        self.setFixedSize(800, 600)
        
        # Data
        self.features = features
        self.calculated_shifts = np.zeros(features.shape[1], dtype=int)
        
        # UI Elements
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Title
        title_lbl = QLabel("Cross-Correlation Alignment Preview")
        title_lbl.setFont(self.FONT_TITLE)
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        # Controls Layout
        ctrl_layout = QHBoxLayout()
        
        lbl_mode = QLabel("Method:")
        lbl_mode.setFont(self.FONT_CTRL)
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["center", "average"])
        self.combo_mode.setFont(self.FONT_CTRL)
        self.combo_mode.setFixedWidth(150)
        
        self.btn_calc = QPushButton("Calculate & Preview")
        self.btn_calc.setFont(self.FONT_CTRL)
        self.btn_calc.setMinimumHeight(40)
        self.btn_calc.clicked.connect(self.calculate_alignment)

        ctrl_layout.addWidget(lbl_mode)
        ctrl_layout.addWidget(self.combo_mode)
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(self.btn_calc)
        ctrl_layout.addStretch()

        # Preview Image Label
        self.lbl_preview = QLabel("Preview Area")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_preview.setStyleSheet(self.LABEL_STYLE)

        # Dialog Buttons (OK/Cancel)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.setFont(self.FONT_CTRL)
        for btn in self.button_box.buttons():
            btn.setMinimumSize(100, 40)
            
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Add to main layout
        layout.addLayout(ctrl_layout)
        layout.addWidget(self.lbl_preview)
        layout.addWidget(self.button_box)
        
        self.setLayout(layout)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_preview(self.features)

    def calculate_alignment(self):
        mode = self.combo_mode.currentText()
        try:
            # Calculate shifts using the util function
            self.calculated_shifts = common_line_method(self.features.T, c_line=mode)
            
            # Create aligned features for preview
            aligned_features = np.zeros_like(self.features)
            for i, shift in enumerate(self.calculated_shifts):
                aligned_features[:, i] = np.roll(self.features[:, i], shift)
            
            self.update_preview(aligned_features)
            
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Alignment failed:\n{str(e)}")

    def update_preview(self, features_data):
        # Convert to QPixmap
        h, w = features_data.shape
        qimg = QImage(features_data.data, w, h, w, QImage.Format_Grayscale8)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qimg)
        rect = self.lbl_preview.contentsRect()
        scaled_pixmap = pixmap.scaled(rect.width(), rect.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.lbl_preview.setPixmap(scaled_pixmap)

    def get_shifts(self):
        return self.calculated_shifts