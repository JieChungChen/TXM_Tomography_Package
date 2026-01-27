import os
import sys
import yaml
import time
import torch
import torch.nn.functional as F
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QMessageBox,
    QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QScrollBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage

# Import from ref_remover_ddpm
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ref_remover_ddpm'))
from ref_remover import Diffusion_UNet, DDIM_Sampler
from src.logic.utils import norm_to_8bit


class AIRefRemoverThread(QThread):
    progress_updated = pyqtSignal(int, int, float)
    image_ready = pyqtSignal(object)
    finished_signal = pyqtSignal(list)  # list of processed images
    failed = pyqtSignal(str)

    def __init__(self, txm_images, configs, model_ckpt, device="cuda:0"):
        super().__init__()
        self.txm_images = txm_images
        self.configs = configs
        self.model_ckpt = model_ckpt
        self.device = device

    def run(self):
        try:
            # Get images from TXM_Images object
            raw_images = self.txm_images.original

            # Load config
            with open(self.configs, 'r') as f:
                configs = yaml.safe_load(f)

            # Load model
            model_configs = configs['model_settings']
            model = Diffusion_UNet(model_configs).to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(self.model_ckpt, map_location=self.device), strict=True)
            model.eval()
            sampler = DDIM_Sampler(model, configs['ddpm_settings'], ddim_sampling_steps=50).to(self.device)

            glob_max = np.max(raw_images)
            processed_images = []

            start_time = time.time()
            n_steps = len(raw_images) - 1
            self.progress_updated.emit(0, n_steps, 0.0)

            with torch.no_grad():
                for i in range(n_steps):
                    if self.isInterruptionRequested():
                        break

                    max_g = np.max(raw_images[i:i + 2])
                    input_1 = torch.tensor(raw_images[i] / max_g).unsqueeze(0).float().to(self.device)
                    input_2 = torch.tensor(raw_images[i + 1] / max_g).unsqueeze(0).float().to(self.device)

                    if input_1.shape[1] != 256:
                        input_1 = F.interpolate(input_1.unsqueeze(0), size=(256, 256), mode='bicubic').squeeze(0)
                        input_2 = F.interpolate(input_2.unsqueeze(0), size=(256, 256), mode='bicubic').squeeze(0)

                    input_imgs = torch.cat([input_1, input_2], dim=0)

                    torch.manual_seed(0)
                    noise = torch.randn(size=[1, 1, 256, 256], device=self.device)
                    pred = sampler(input_imgs.unsqueeze(0), noise).squeeze().cpu().numpy()
                    pred = pred / pred.mean()

                    obj_pred_1 = input_1.squeeze().cpu().numpy() / pred * max_g
                    obj_pred_2 = input_2.squeeze().cpu().numpy() / pred * max_g

                    # Process first image
                    processed_images.append(obj_pred_1)
                    self.image_ready.emit(obj_pred_1)

                    # Process second image if last pair
                    if i == (len(raw_images) - 2):
                        processed_images.append(obj_pred_2)
                        self.image_ready.emit(obj_pred_2)

                    elapsed = time.time() - start_time
                    self.progress_updated.emit(i+1, n_steps, elapsed)

            if not self.isInterruptionRequested():
                self.finished_signal.emit(processed_images)
            else:
                self.failed.emit("Inference cancelled by user")

        except Exception as e:
            self.failed.emit(str(e))

class AIRefRemoverDialog(QDialog):
    def __init__(self, txm_images, parent=None):
        super().__init__(parent)
        self.txm_images = txm_images
        self.processed_images = None
        self.thread = None
        self.preview_images = []
        self.current_preview_index = 0

        self.setWindowTitle("AI Background Removal")
        self.setFixedSize(600, 700)
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

        # Path inputs
        self.config_input = self.create_path_input("模型設定檔:", "ref_remover/configs/BGC_v1_inference.yml", filter="*.yml")
        self.ckpt_input = self.create_path_input("模型權重:", "ref_remover/checkpoints/ddpm_pair_ft_10K.pt", filter="*.pt")

        # Run button
        self.run_button = QPushButton("Run Inference")
        self.run_button.setFont(QFont("Calibri", 14))
        self.run_button.clicked.connect(self.run_inference)
        layout.addWidget(self.run_button)

        # Progress bar + ETA + Cancel
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(QFont("Calibri", 14))
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.eta_label = QLabel("剩餘時間: -")
        self.eta_label.setFont(QFont("Calibri", 14))
        self.eta_label.setAlignment(Qt.AlignCenter)
        self.eta_label.setFixedHeight(20)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFont(QFont("Calibri", 14))
        self.cancel_button.clicked.connect(self.cancel_inference)
        self.cancel_button.setEnabled(False)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.eta_label)
        progress_layout.addWidget(self.cancel_button)
        layout.addLayout(progress_layout)

        # Status label
        self.status_label = QLabel("準備開始推論...")
        self.status_label.setFont(QFont("Calibri", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(30)
        layout.addWidget(self.status_label)

        # Image preview
        self.image_label = QLabel("Preview.")
        self.image_label.setFont(QFont("Calibri", 14))
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Slider for browsing stack
        scroll_layout = QVBoxLayout()
        self.index_label = QLabel("0 / 0")
        self.index_label.setAlignment(Qt.AlignCenter)
        self.index_label.setFixedHeight(20)
        scroll_layout.addWidget(self.index_label)

        self.scrollbar = QScrollBar(Qt.Horizontal)
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
        self.scrollbar.setStyleSheet(slider_style)
        self.scrollbar.setEnabled(False)
        self.scrollbar.valueChanged.connect(self.scrollbar_changed)
        scroll_layout.addWidget(self.scrollbar)
        layout.addLayout(scroll_layout)

    def create_path_input(self, label_text, default="", filter=None, is_dir=False):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFont(QFont("Calibri", 14))
        line_edit = QLineEdit(default)
        line_edit.setFont(QFont("Calibri", 14))
        browse_btn = QPushButton("Browse")
        browse_btn.setFont(QFont("Calibri", 14))
        if is_dir:
            browse_btn.clicked.connect(lambda: self.browse_dir(line_edit))
        else:
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
            self.gpu_status_label.setText(f"{device_str} ❌ 無法使用")
            self.gpu_status_label.setStyleSheet("color: red;")

    def run_inference(self):
        if self.txm_images is None:
            QMessageBox.warning(self, "No Images", "No images to process.")
            return

        configs = self.config_input.text()
        model_ckpt = self.ckpt_input.text()

        if not os.path.exists(configs):
            QMessageBox.warning(self, "Config File", "Config file not found.")
            return
        if not os.path.exists(model_ckpt):
            QMessageBox.warning(self, "Model File", "Model checkpoint not found.")
            return

        self.progress_bar.setValue(0)
        self.eta_label.setText("剩餘時間: -")
        self.status_label.setText("開始推論...")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self.thread = AIRefRemoverThread(self.txm_images, configs, model_ckpt)
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.failed.connect(self.on_failed)
        self.thread.image_ready.connect(self.add_preview_image)
        self.thread.start()

    def cancel_inference(self):
        if self.thread and self.thread.isRunning():
            self.thread.requestInterruption()
            self.status_label.setText("正在取消...")
            self.cancel_button.setEnabled(False)

    def update_progress(self, current, total, elapsed):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"處理第 {current} 張 / 共 {total} 張")

        if current > 0 and elapsed:
            avg_time = elapsed / current
            eta = avg_time * (total - current)
            self.eta_label.setText(f"剩餘時間: {self.format_eta(eta)}")

    def on_finished(self, processed_images):
        self.processed_images = np.array(processed_images)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText("✅ 推論完成")
        self.eta_label.setText("完成 ✅")
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.accept()

    def on_failed(self, error_msg):
        QMessageBox.critical(self, "Error", f"AI Background Removal failed: {error_msg}")
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.reject()

    def add_preview_image(self, image):
        self.preview_images.append(image)
        self.scrollbar.setEnabled(True)
        self.scrollbar.setMaximum(len(self.preview_images) - 1)
        self.scrollbar.setValue(len(self.preview_images) - 1)  # 自動跳到最新一張
        self.show_preview_image(len(self.preview_images) - 1)

    def show_preview_image(self, index):
        if not self.preview_images:
            return
        self.current_preview_index = index
        img = self.preview_images[index]
        img = norm_to_8bit(img)

        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.index_label.setText(f"{index+1} / {len(self.preview_images)}")

    def scrollbar_changed(self, value):
        self.show_preview_image(value)

    def format_eta(self, seconds):
        if seconds < 60:
            return f"{int(seconds)} 秒"
        elif seconds < 3600:
            return f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒"
        else:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hrs} 小時 {mins} 分"
