from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSlider, QPushButton, QLineEdit, QHBoxLayout, QFileDialog, QMessageBox, QDialogButtonBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# 參考模式選擇Dialog
class ReferenceModeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Reference Mode")
        # 統一 Dialog 外觀
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)
        font = QFont("Calibri", 14)
        label = QLabel("Please choose to apply one or two references?")
        label.setFont(font)
        self.single_btn = QPushButton("Single Reference")
        self.single_btn.setFont(font)
        self.dual_btn = QPushButton("Dual Reference")
        self.dual_btn.setFont(font)
        self.single_btn.clicked.connect(self.accept_single)
        self.dual_btn.clicked.connect(self.accept_dual)
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.single_btn)
        layout.addWidget(self.dual_btn)
        self.setLayout(layout)
        self.mode = None
    def accept_single(self):
        self.mode = 'single'
        self.accept()
    def accept_dual(self):
        self.mode = 'dual'
        self.accept()


class SplitSliderDialog(QDialog):
    def __init__(self, max_value, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dual Reference and Split Point")
        # 統一 Dialog 外觀
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)
        from PyQt5.QtGui import QFont
        font = QFont("Calibri", 14)
        # 檔案欄位1
        self.ref1_label = QLabel("Reference file for first half:")
        self.ref1_label.setFont(font)
        self.ref1_line = QLineEdit()
        self.ref1_line.setFont(font)
        self.ref1_line.setReadOnly(True)
        self.ref1_btn = QPushButton("Browse")
        self.ref1_btn.setFont(font)
        self.ref1_btn.clicked.connect(self.browse_ref1)
        # 檔案欄位2
        self.ref2_label = QLabel("Reference file for second half:")
        self.ref2_label.setFont(font)
        self.ref2_line = QLineEdit()
        self.ref2_line.setFont(font)
        self.ref2_line.setReadOnly(True)
        self.ref2_btn = QPushButton("Browse")
        self.ref2_btn.setFont(font)
        self.ref2_btn.clicked.connect(self.browse_ref2)
        # 分割點滑桿
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
        self.slider.setMinimum(1)
        self.slider.setMaximum(max_value-1)
        self.slider.setValue(max_value//2)
        self.split_label = QLabel(self._split_text())
        self.split_label.setFont(font)
        self.slider.valueChanged.connect(self.update_label)
        # 確定按鈕
        self.ok_btn = QPushButton("Apply Reference")
        self.ok_btn.setFont(font)
        self.ok_btn.clicked.connect(self.try_accept)
        # 佈局
        layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.ref1_label)
        row1.addWidget(self.ref1_line)
        row1.addWidget(self.ref1_btn)
        layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(self.ref2_label)
        row2.addWidget(self.ref2_line)
        row2.addWidget(self.ref2_btn)
        layout.addLayout(row2)
        layout.addWidget(self.split_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.ok_btn)
        self.setLayout(layout)
        self.ref1_path = ''
        self.ref2_path = ''
    def browse_ref1(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select reference file for first half", "", "(*.xrm *.tif)")
        if fname:
            self.ref1_path = fname
            self.ref1_line.setText(fname)
    def browse_ref2(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select reference file for second half", "", "(*.xrm *.tif)")
        if fname:
            self.ref2_path = fname
            self.ref2_line.setText(fname)
    def update_label(self):
        self.split_label.setText(self._split_text())
    def _split_text(self):
        v = self.slider.value()
        maxv = self.slider.maximum()+1
        return f"Split index: {v}    First half: 0~{v-1}    Second half: {v}~{maxv-1}"
    def get_split(self):
        return self.slider.value()
    def get_refs(self):
        return self.ref1_path, self.ref2_path
    def try_accept(self):
        if not self.ref1_path or not self.ref2_path:
            QMessageBox.warning(self, "Missing Reference File", "Please select both reference files!")
            return
        self.accept()
