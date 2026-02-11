import os
import sys
import yaml
import time
import torch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QMessageBox,
    QHBoxLayout, QLineEdit, QPushButton, QFileDialog,
    QSpinBox, QCheckBox, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from sinogram_alignment.model import sino_align_transformer_builder
from src.logic.utils import norm_to_8bit


class SinoAlignerThread(QThread):
    progress_updated = pyqtSignal(int, int, float)
    image_ready = pyqtSignal(object)
    finished_signal = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, txm_images, model_config, model_path, max_shift, n_iter, seed, target_slice, device="cuda:0"):
        super().__init__()
        self.txm_images = txm_images
        self.model_config = model_config
        self.model_path = model_path
        self.max_shift = max_shift
        self.n_iter = n_iter
        self.seed = seed
        self.target_slice = target_slice
        self.device = device

    def apply_shift(self, sinogram, shifts):
        """Apply shifts to sinogram projections"""
        n_projs, size = sinogram.shape
        for i in range(n_projs):
            sinogram[i] = torch.roll(sinogram[i], shifts[i], dims=0)
        return sinogram

    def run(self):
        try:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)

            with open(self.model_config, 'r') as f:
                configs = yaml.safe_load(f)

            # Get images from TXM_Images object
            raw_images = self.txm_images # (N, H, W)

            # Load model
            model = sino_align_transformer_builder(configs['model_settings']).to(self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=True)
            model.eval()

            start_time = time.time()

            with torch.no_grad():
                slice_idx = self.target_slice
                raw_sino = raw_images[:, slice_idx, :].astype(np.float32)  # (N_proj, Width)
                scale = 512 // raw_sino.shape[1]
                
                if scale != 1:
                    raw_sino_resized = Image.fromarray(raw_sino).resize((512, raw_sino.shape[0]), resample=Image.BILINEAR)
                    raw_sino = np.array(raw_sino_resized)
                    pmin, pmax = np.percentile(raw_sino, [0.0, 99.9])
                    raw_sino = np.clip((raw_sino - pmin) / (pmax - pmin), 0, 1)

                raw_sino = 1 - raw_sino  # Invert for model

                # Prepare tensor (pad to 181 projections if needed)
                n_projs, detector_pixs = raw_sino.shape
                mask = torch.zeros((1, 181), dtype=torch.bool, device=self.device)
                mask[0, n_projs:] = True  # mask invalid projections
                
                sino_temp = torch.zeros((181, detector_pixs), dtype=torch.float32)
                sino_temp[:n_projs, :] = torch.from_numpy(raw_sino)
                sino_temp = sino_temp.float().to(self.device)

                # Iterative alignment
                total_shift = np.zeros(n_projs, dtype=int)
                for iter_i in range(self.n_iter):
                    sino_temp_input = sino_temp.unsqueeze(0)  # (1, 181, detector_pixs)
                    pred_shift, _ = model(sino_temp_input, mask)
                    pred_shift = pred_shift.cpu().detach().squeeze().numpy()
                    pred_shift = (pred_shift * self.max_shift).astype(int)
                    print(pred_shift)
                    total_shift = total_shift - pred_shift[:n_projs]
                    sino_temp = self.apply_shift(sino_temp, -pred_shift)

                aligned_sino = sino_temp.cpu().detach().squeeze().numpy()
                aligned_sino = aligned_sino[:n_projs, :]  
                self.image_ready.emit(aligned_sino)  

                elapsed = time.time() - start_time
                self.progress_updated.emit(1, self.n_iter, elapsed)

            if not self.isInterruptionRequested():
                self.finished_signal.emit([aligned_sino, total_shift/scale])
            else:
                self.failed.emit("Alignment cancelled by user")

        except Exception as e:
            import traceback
            self.failed.emit(f"{str(e)}\n{traceback.format_exc()}")


class SinoAlignerDialog(QDialog):
    def __init__(self, txm_images, parent=None):
        super().__init__(parent)
        self.txm_images = txm_images.get_norm_images()  
        self.aligned_images = None
        self.thread = None
        self.current_slice = txm_images.original.shape[1] // 2  # Default to middle slice

        self.setWindowTitle("Sinogram Alignment (Transformer)")
        self.setFixedSize(800, 850)
        self.setModal(True)

        # 統一 Dialog 外觀
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)
        font = QFont("Calibri", 14)
        self.setFont(font)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # GPU 狀態顯示
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("GPU 狀態:")
        gpu_label.setFont(QFont("Calibri", 14))
        self.gpu_status_label = QLabel("檢測中…")
        self.gpu_status_label.setFont(QFont("Calibri", 14))
        self.gpu_status_label.setFixedHeight(20)
        gpu_layout.addWidget(gpu_label)
        gpu_layout.addWidget(self.gpu_status_label)
        gpu_layout.addStretch(1)
        layout.addLayout(gpu_layout)
        self.check_cuda_availability("cuda:0")

        # Slice Selector
        slice_layout = QHBoxLayout()
        slice_label = QLabel("選擇 Sinogram Slice:")
        slice_label.setFont(QFont("Calibri", 14))
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(txm_images.original.shape[1] - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.setTickInterval(10)
        self.slice_slider.valueChanged.connect(self.update_slice_preview)
        
        self.slice_spinbox = QSpinBox()
        self.slice_spinbox.setFont(QFont("Calibri", 14))
        self.slice_spinbox.setMinimum(0)
        self.slice_spinbox.setMaximum(txm_images.original.shape[1] - 1)
        self.slice_spinbox.setValue(self.current_slice)
        self.slice_spinbox.valueChanged.connect(self.slice_slider.setValue)
        self.slice_slider.valueChanged.connect(self.slice_spinbox.setValue)
        
        slice_layout.addWidget(slice_label)
        slice_layout.addWidget(self.slice_slider)
        slice_layout.addWidget(self.slice_spinbox)
        layout.addLayout(slice_layout)

        # Image preview (原始 Sinogram)
        # Original Sinogram label
        original_title = QLabel("Raw Sinogram")
        original_title.setFont(QFont("Calibri", 12, QFont.Bold))
        original_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(original_title)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(180)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #bfc7d5;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #f5f7fa, stop:1 #c3cfe2);
                padding: 4px;
            }
        """)
        layout.addWidget(self.image_label)

        # Aligned Sinogram label
        aligned_title = QLabel("Aligned Sinogram")
        aligned_title.setFont(QFont("Calibri", 12, QFont.Bold))
        aligned_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(aligned_title)
        
        self.aligned_image_label = QLabel("Not Aligned Yet")
        self.aligned_image_label.setAlignment(Qt.AlignCenter)
        self.aligned_image_label.setFixedHeight(180)
        self.aligned_image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #bfc7d5;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #f5f7fa, stop:1 #c3cfe2);
                padding: 4px;
            }
        """)
        layout.addWidget(self.aligned_image_label)

        # Update initial preview
        self.update_slice_preview(self.current_slice)

        # Path inputs
        self.config_input = self.create_path_input(
            "模型設定檔:", 
            "sinogram_alignment/configs/alignment_self_att_v3.yml", 
            filter="YAML Files (*.yml)"
        )
        self.ckpt_input = self.create_path_input(
            "模型權重:", 
            "sinogram_alignment/checkpoints/alignment_ep500.pt", 
            filter="PyTorch Model (*.pt)"
        )

        # Parameters layout - all in one row
        params_layout = QHBoxLayout()
        
        # Max shift
        shift_label = QLabel("Max Shift:")
        shift_label.setFont(QFont("Calibri", 12))
        self.max_shift_spinbox = QSpinBox()
        self.max_shift_spinbox.setFont(QFont("Calibri", 12))
        self.max_shift_spinbox.setMinimum(10)
        self.max_shift_spinbox.setMaximum(200)
        self.max_shift_spinbox.setValue(50)
        self.max_shift_spinbox.setFixedWidth(70)
        
        # Iterations
        iter_label = QLabel("Iter:")
        iter_label.setFont(QFont("Calibri", 12))
        self.n_iter_spinbox = QSpinBox()
        self.n_iter_spinbox.setFont(QFont("Calibri", 12))
        self.n_iter_spinbox.setMinimum(1)
        self.n_iter_spinbox.setMaximum(20)
        self.n_iter_spinbox.setValue(5)
        self.n_iter_spinbox.setFixedWidth(60)
        
        # Random seed
        self.seed_checkbox = QCheckBox("Seed:")
        self.seed_checkbox.setFont(QFont("Calibri", 12))
        self.seed_checkbox.setChecked(False)
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setFont(QFont("Calibri", 12))
        self.seed_spinbox.setMinimum(0)
        self.seed_spinbox.setMaximum(99999)
        self.seed_spinbox.setValue(42)
        self.seed_spinbox.setFixedWidth(80)
        self.seed_spinbox.setEnabled(False)
        self.seed_checkbox.toggled.connect(self.seed_spinbox.setEnabled)
        
        params_layout.addWidget(shift_label)
        params_layout.addWidget(self.max_shift_spinbox)
        params_layout.addSpacing(15)
        params_layout.addWidget(iter_label)
        params_layout.addWidget(self.n_iter_spinbox)
        params_layout.addSpacing(15)
        params_layout.addWidget(self.seed_checkbox)
        params_layout.addWidget(self.seed_spinbox)
        params_layout.addStretch(1)
        layout.addLayout(params_layout)

        # Run button
        self.run_button = QPushButton("Run Alignment")
        self.run_button.setFont(QFont("Calibri", 14, QFont.Bold))
        self.run_button.setMinimumHeight(45)
        self.run_button.clicked.connect(self.run_alignment)
        layout.addWidget(self.run_button)

        # Apply and Discard buttons (for after alignment)
        result_buttons_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.setFont(QFont("Calibri", 14, QFont.Bold))
        self.apply_button.setMinimumHeight(45)
        self.apply_button.setEnabled(False)
        self.apply_button.setStyleSheet("QPushButton:enabled { background-color: #28a745; color: white; }")
        self.apply_button.clicked.connect(self.apply_changes)
        
        self.discard_button = QPushButton("Discard")
        self.discard_button.setFont(QFont("Calibri", 14))
        self.discard_button.setMinimumHeight(45)
        self.discard_button.setEnabled(False)
        self.discard_button.clicked.connect(self.discard_changes)
        
        result_buttons_layout.addWidget(self.apply_button)
        result_buttons_layout.addWidget(self.discard_button)
        layout.addLayout(result_buttons_layout)

        # Progress bar + Cancel
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(QFont("Calibri", 14))
        self.progress_bar.setAlignment(Qt.AlignCenter)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFont(QFont("Calibri", 14))
        self.cancel_button.clicked.connect(self.cancel_alignment)
        self.cancel_button.setEnabled(False)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.cancel_button)
        layout.addLayout(progress_layout)

        # Status label
        self.status_label = QLabel("選擇 Slice 並按 Run Alignment 開始")
        self.status_label.setFont(QFont("Calibri", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(30)
        layout.addWidget(self.status_label)

    def update_slice_preview(self, slice_idx):
        self.current_slice = slice_idx
        sino = self.txm_images[:, slice_idx, :]  # (N_proj, Width)
        
        # Normalize and display
        img_8bit = norm_to_8bit(sino)  # Keep as (N_proj, Width) for horizontal display
        h, w = img_8bit.shape
        
        img_8bit = np.ascontiguousarray(img_8bit)
        qimg = QImage(img_8bit.data, w, h, w, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.image_label.width() - 10, 
            self.image_label.height() - 10, 
            Qt.IgnoreAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def create_path_input(self, label_text, default="", filter=None):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFont(QFont("Calibri", 14))
        line_edit = QLineEdit(default)
        line_edit.setFont(QFont("Calibri", 14))
        browse_btn = QPushButton("Browse")
        browse_btn.setFont(QFont("Calibri", 14))
        browse_btn.clicked.connect(lambda: self.browse_file(line_edit, filter))
        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)
        self.layout().addLayout(layout)
        return line_edit

    def browse_file(self, line_edit, filter):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter)
        if path:
            line_edit.setText(path)

    def check_cuda_availability(self, device_str):
        available = torch.cuda.is_available()
        if available:
            self.gpu_status_label.setText(f"{device_str} ✅ 可用")
            self.gpu_status_label.setStyleSheet("color: green;")
        else:
            self.gpu_status_label.setText(f"{device_str} ❌ 無法使用 (將使用 CPU)")
            self.gpu_status_label.setStyleSheet("color: orange;")

    def run_alignment(self):
        if self.txm_images is None:
            QMessageBox.warning(self, "No Images", "No images to process.")
            return

        model_config = self.config_input.text()
        model_path = self.ckpt_input.text()

        if not os.path.exists(model_config):
            QMessageBox.warning(self, "Config File", "Config file not found.")
            return
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Model File", "Model checkpoint not found.")
            return

        max_shift = self.max_shift_spinbox.value()
        n_iter = self.n_iter_spinbox.value()
        seed = self.seed_spinbox.value() if self.seed_checkbox.isChecked() else None

        self.progress_bar.setValue(0)
        self.status_label.setText("正在對齊...")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.slice_slider.setEnabled(False)
        self.slice_spinbox.setEnabled(False)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.thread = SinoAlignerThread(
            self.txm_images, model_config, model_path, 
            max_shift, n_iter, seed, self.current_slice, device
        )
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.failed.connect(self.on_failed)
        self.thread.image_ready.connect(self.show_aligned_preview)
        self.thread.start()

    def cancel_alignment(self):
        if self.thread and self.thread.isRunning():
            self.thread.requestInterruption()
            self.status_label.setText("正在取消...")
            self.cancel_button.setEnabled(False)

    def update_progress(self, current, total, elapsed):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_finished(self, result_list):
        aligned_sino, shifts = result_list
        self.calculated_shifts = shifts
        
        # Prepare preview of aligned images
        aligned_images = self.txm_images
        n_proj = aligned_sino.shape[0]
        
        # Apply shifts along Width axis (axis=1) since sinogram alignment is horizontal
        for i in range(n_proj):
            aligned_images[i, :, :] = np.roll(aligned_images[i, :, :], int(shifts[i]), axis=1)
        
        self.aligned_images = aligned_images
        
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText("✅ 對齊完成！請檢查預覽並選擇 Apply 或 Discard。")
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.slice_slider.setEnabled(True)
        self.slice_spinbox.setEnabled(True)
        
        # Enable Apply and Discard buttons
        self.apply_button.setEnabled(True)
        self.discard_button.setEnabled(True)

    def apply_changes(self):
        """Apply the alignment and close dialog"""
        if self.aligned_images is not None:
            self.accept()
        else:
            QMessageBox.warning(self, "No Results", "No alignment results to apply.")
    
    def discard_changes(self):
        """Discard alignment results and reset UI"""
        self.aligned_images = None
        self.calculated_shifts = None
        self.aligned_image_label.setText("尚未對齊")
        self.aligned_image_label.setPixmap(QPixmap())
        self.apply_button.setEnabled(False)
        self.discard_button.setEnabled(False)
        self.status_label.setText("已捨棄對齊結果，可重新選擇參數執行對齊")
    
    def on_failed(self, error_msg):
        QMessageBox.critical(self, "Error", f"Sinogram Alignment failed:\n{error_msg}")
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.slice_slider.setEnabled(True)
        self.slice_spinbox.setEnabled(True)
        self.reject()

    def show_aligned_preview(self, aligned_sino):
        """Display the aligned sinogram in the second label"""
        img_8bit = norm_to_8bit(aligned_sino)  # Keep as (N_proj, Width) for horizontal display
        h, w = img_8bit.shape
        
        img_8bit = np.ascontiguousarray(img_8bit)
        qimg = QImage(img_8bit.data, w, h, w, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.aligned_image_label.width() - 10, 
            self.aligned_image_label.height() - 10, 
            Qt.IgnoreAspectRatio, 
            Qt.SmoothTransformation
        )
        self.aligned_image_label.setPixmap(pixmap)
        self.status_label.setText("✅ 對齊完成（預覽）")