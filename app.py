import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.')) 
from PyQt5.QtWidgets import QProgressDialog, QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from src.gui import (AlignViewer, ContrastDialog, FBPViewer, MLEMSettingsDialog,
                     FBPResolutionDialog, MosaicPreviewDialog, ShiftDialog, 
                     ReferenceModeDialog, SplitSliderDialog, resolve_duplicates)
from src.gui.main_window import Ui_TXM_ToolBox
from src.logic import (AppContext, TXM_Images, FBPWorker, MLEMWorker, data_io, 
                       norm_to_8bit, find_duplicate_angles, angle_sort, handle_errors)


class TXM_ToolBox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_TXM_ToolBox()
        self.ui.setupUi(self)
        self.setMinimumSize(600, 600)

        self.context = AppContext()

        self.current_id = 0
        self.clip_lower = 0.1
        self.clip_upper = 0.1
        
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.adjust_aspect_ratio)

        self.ui.imageSlider.valueChanged.connect(self.update_image)
        self.ui.action_tomo_txrm.triggered.connect(self.load_tomo_txrm)
        self.ui.action_multi_txrm.triggered.connect(self.load_multiple_txrm)
        self.ui.action_tomo_tifs.triggered.connect(lambda: self.load_tifs('tomo'))
        self.ui.action_mosaic_txrm.triggered.connect(self.load_mosaic)
        self.ui.action_mosaic_tifs.triggered.connect(lambda: self.load_tifs('mosaic'))
        self.ui.action_single_xrm.triggered.connect(self.load_single)
        self.ui.action_save_raw.triggered.connect(lambda: self.save_image_as_tif('global'))
        self.ui.action_save_norm.triggered.connect(lambda: self.save_image_as_tif('each'))
        
        self.ui.action_vertical_flip.triggered.connect(self.vertical_flip)
        self.ui.action_reference.triggered.connect(self.load_reference)
        self.ui.action_y_shift.triggered.connect(self.apply_y_shift)
        self.ui.action_adjust_contrast.triggered.connect(self.open_contrast_dialog)
        self.ui.action_alignment.triggered.connect(self.open_align_viewer)
        self.ui.action_reconstruction.triggered.connect(self.get_fbp_result)
        self.ui.action_ML_EM.triggered.connect(self.get_mlem_result)
        self.ui.action_full_view.triggered.connect(self.mosaic_stitching)

        self.ui.actionAI_Reference.triggered.connect(self.ref_ai_remover)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_timer.start(100)
        if self.context.images is not None:
            self.update_image(self.current_id)

    def adjust_aspect_ratio(self):
        aspect_ratio = 1 / 1  
        current_size = self.size()
        width = current_size.width()
        height = int(width / aspect_ratio)
        if current_size.height() != height:
            self.resize(width, height)

    @handle_errors(title="Load TXRM Error")
    def load_tomo_txrm(self, *args):
        """載入單一斷層 TXRM 原始檔。"""
        filename, _ = QFileDialog.getOpenFileName(self, "Open .txrm file", self.context.last_load_dir, "*.txrm")
        if not filename:
            return

        images, metadata, angles, ref = data_io.read_txm_raw(filename, mode='tomo')
        self.context.set_from_file(filename, 'tomo')
        self.context.images = TXM_Images(images, 'tomo', metadata, angles)
        self.context.images.apply_ref(ref)
        self.update_env()
        self.show_info_message("TXM Metadata", metadata)

    @handle_errors(title="Load Multiple TXRM Error")
    def load_multiple_txrm(self, *args):
        """載入多個斷層 TXRM 原始檔，並人工篩選重複投影。"""
        file_list, _ = QFileDialog.getOpenFileNames(self, "Select multiple .txrm file", self.context.last_load_dir, "*.txrm")
        if not file_list:
            return

        images, angles, ref, file_names = data_io.read_multiple_txrm(file_list)
        self.context.set_from_file(file_list[0], 'tomo')
        duplicates = find_duplicate_angles(angles)

        if duplicates:
            images, angles = resolve_duplicates(images, angles, duplicates, ref, file_names)
        else:
            images, angles = angle_sort(images, angles)

        self.context.images = TXM_Images(images, 'tomo', angles=angles)
        self.update_env()

    @handle_errors(title="Load XRM Error")
    def load_mosaic(self, *args):
        """載入拼接 XRM 原始檔。"""
        filename, _ = QFileDialog.getOpenFileName(self, "Open .xrm file", self.context.last_load_dir, "*.xrm")
        if not filename:
            return

        images, metadata, ref = data_io.read_txm_raw(filename, mode='mosaic')
        self.context.set_from_file(filename, 'mosaic')
        self.context.images = TXM_Images(images, 'mosaic', metadata)
        self.context.images.apply_ref(ref)
        self.update_env()
        self.show_info_message("TXM Metadata", metadata)

    @handle_errors(title="Load XRM Error")
    def load_single(self, *args):
        """載入單張 XRM 原始檔。"""
        filename, _ = QFileDialog.getOpenFileName(self, "Open .xrm file", self.context.last_load_dir, "*.xrm")
        if not filename:
            return

        images, metadata, ref = data_io.read_txm_raw(filename, mode='single')
        self.context.set_from_file(filename, 'single')
        self.context.images = TXM_Images(images, 'single', metadata)
        self.context.images.apply_ref(ref)
        self.update_env()
        self.show_info_message("TXM Metadata", metadata)

    @handle_errors(title="Load TIFs Error")
    def load_tifs(self, mode):
        """載入斷層或拼接 TIF 影像。"""
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", self.context.last_load_dir)
        if not folder:
            return

        images = data_io.load_tif_folder(folder)
        if images is None:
            raise FileNotFoundError("No TIF images found in the selected folder")
        
        self.context.set_from_folder(folder, mode)
        self.context.images = TXM_Images(images, 'tomo')
        self.update_env()

    @handle_errors(title="Load Reference Error")
    def load_reference(self, *args):
        mode_box = ReferenceModeDialog(self)
        if mode_box.exec_() != QDialog.Accepted:
            return
        mode = mode_box.mode
        if mode == 'single':
            filename, _ = QFileDialog.getOpenFileName(self, "Select Reference File", "", "(*.xrm *.tif)")
            if not filename:
                return
            ref = data_io.load_ref(filename)
            self.context.images.apply_ref(ref)
        elif mode == 'dual':
            num_imgs = len(self.context.images)
            if num_imgs < 2:
                QMessageBox.warning(self, "Insufficient Images", "Dual reference mode requires at least 2 images!")
                return
            dlg = SplitSliderDialog(num_imgs, self)
            if dlg.exec_() != QDialog.Accepted:
                return
            filename1, filename2 = dlg.get_refs()
            split_idx = dlg.get_split()
            if not filename1 or not filename2:
                QMessageBox.warning(self, "Missing Reference Files", "Please select two reference files!")
                return
            ref1 = data_io.load_ref(filename1)
            ref2 = data_io.load_ref(filename2)
            self.context.images.apply_ref(ref1, ref2, split_idx)
        self.update_image(self.ui.imageSlider.value())

    def update_image(self, index=0):
        self.current_id = index
        img = self.context.images.get_image(index)
        img = norm_to_8bit(img, clip_lower=self.clip_lower, clip_upper=self.clip_upper)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        self.ui.imageLabel.setPixmap(pixmap.scaled(
            self.ui.imageLabel.width(),
            self.ui.imageLabel.height(),
            Qt.KeepAspectRatio))
        self.ui.imageIndexLabel.setText(f"{index+1} / {len(self.context.images)}")
        if self.context.mode == 'tomo':
            theta = self.context.images.get_theta(index)
            if theta is not None:
                self.setWindowTitle(f"{self.context.sample_name}, theta = {theta}")
        else:
            self.setWindowTitle(f"{self.context.sample_name}")

    def update_env(self):
        """update UI and environment after loading images."""
        self.ui.action_reference.setEnabled(True)
        self.ui.action_save_raw.setEnabled(True)
        self.ui.action_save_norm.setEnabled(True)
        self.ui.action_vertical_flip.setEnabled(True)
        self.ui.action_y_shift.setEnabled(True)
        self.ui.action_adjust_contrast.setEnabled(True)
        self.ui.action_alignment.setEnabled(self.context.mode == 'tomo')
        self.ui.action_reconstruction.setEnabled(self.context.mode == 'tomo')
        self.ui.action_ML_EM.setEnabled(self.context.mode == 'tomo')
        self.ui.action_full_view.setEnabled(self.context.mode == 'mosaic')
        self.ui.actionAI_Reference.setEnabled(self.context.ai_available)

        self.ui.imageSlider.setMinimum(0)
        self.ui.imageSlider.setMaximum(len(self.context.images) - 1)
        self.update_image()

    def vertical_flip(self, *args):
        self.context.images.flip_vertical()
        self.update_image(self.current_id)

    def apply_y_shift(self, *args):
        h, w = self.context.get_image_size()

        dialog = ShiftDialog(h, self)
        dialog.apply_shift.connect(lambda amount: (
            self.context.images.apply_y_shift(amount),
            self.update_image(self.current_id),
        ) if amount != 0 else None) 
        dialog.exec_()

    def on_contrast_live_update(self, clip_lower, clip_upper):
        """對比度調整後即時更新影像。"""
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.update_image(self.current_id)

    def open_contrast_dialog(self):
        ContrastDialog(
            init_clip_lower=self.clip_lower,
            init_clip_upper=self.clip_upper,
            live_update_callback=self.on_contrast_live_update,
            parent=self
        ).show()

    @handle_errors(title="Alignment Error")
    def open_align_viewer(self, *args):
        dialog = AlignViewer(self.context.images, self.context.last_load_dir)
        if dialog.exec_() == QDialog.Accepted:
            self.update_image(self.current_id)

    @handle_errors(title="Reconstruction Error")
    def get_fbp_result(self, *args):
        """在背景執行緒啟動 FBP 重建，並顯示進度對話框"""
        img_array = self.context.get_images()
        original_size = self.context.get_image_size()  # （高度、寬度）

        # 顯示解析度選擇對話框。
        resolution_dialog = FBPResolutionDialog(original_size, self)
        if resolution_dialog.exec_() != QDialog.Accepted:
            return 

        target_size = resolution_dialog.get_size()
        angle_interval = resolution_dialog.get_angle_interval()
        astra_available = resolution_dialog.get_astra_available()
        self.worker = FBPWorker(img_array, self.context.images.angles, target_size, angle_interval, astra_available)

        # 顯示進度對話框。
        self.progress_dialog = QProgressDialog(
            "Reconstructing...", None, 0, 100, self
        )
        self.progress_dialog.setWindowTitle("FBP Reconstruction")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setFixedSize(350, 100)
        self.progress_dialog.canceled.connect(self.worker.cancel) 
        self.progress_dialog.show()

        # 以選定解析度與角度間隔開始重建。
        self.worker.progress.connect(lambda p, r: (
            self.progress_dialog.setValue(p), 
            self.progress_dialog.setLabelText(f"<span style='font-family: Calibri; font-size:15px; font-weight:bold;'>{r}</span>" )
        ) if not self.progress_dialog.wasCanceled() else None)
        self.worker.finished.connect(lambda recon: (self.progress_dialog.close(), FBPViewer(recon, self).exec_()))
        self.worker.start()

    @handle_errors(title="ML-EM Reconstruction Error")
    def get_mlem_result(self, *args):
        """在背景執行緒啟動 ML-EM 重建，並顯示進度對話框"""
        img_array = self.context.get_images()
        size = self.context.get_image_size()[0]

        mlem_dialog = MLEMSettingsDialog(self, size)
        if mlem_dialog.exec_() != QDialog.Accepted:
            return 
        
        settings = mlem_dialog.get_settings()
        iter_count = settings["iter_count"]
        mask_ratio = settings["mask_ratio"]
        start_layer = settings["start_layer"]
        end_layer = settings["end_layer"]
        angle_interval = settings["angle_interval"]

        self.worker = MLEMWorker(img_array, iter_count, mask_ratio, start_layer, end_layer, angle_interval)
        # 顯示進度對話框
        self.progress_dialog = QProgressDialog(
            "Reconstructing...", None, 0, 100, self
        )
        self.progress_dialog.setWindowTitle("MLEM Reconstruction")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setFixedSize(350, 100)
        self.progress_dialog.canceled.connect(self.worker.cancel) 
        self.progress_dialog.show()

        self.worker.progress.connect(lambda p, r: (
            self.progress_dialog.setValue(p), 
            self.progress_dialog.setLabelText(f"<span style='font-family: Calibri; font-size:15px; font-weight:bold;'>{r}</span>" )
        ) if not self.progress_dialog.wasCanceled() else None)
        self.worker.finished.connect(lambda recon: (self.progress_dialog.close(), FBPViewer(recon, self).exec_()))
        self.worker.start()

    @handle_errors(title="Mosaic Stitching Error")
    def mosaic_stitching(self, *args):
        mosaic = self.context.images.get_mosaic()
        if mosaic is not None:
            dialog = MosaicPreviewDialog(mosaic, self.context)
            dialog.exec_()

    @handle_errors(title="AI Reference Remover Error")
    def ref_ai_remover(self, *args):
        """使用去背景 Diffusion Model 處理原始影像"""
        if not self.context.ai_available:
            QMessageBox.warning(self, "AI Not Available", "PyTorch is not installed.")
            return
        else:
            from src.gui import AIRefRemoverDialog
            dialog = AIRefRemoverDialog(self.context.images, self)
            if dialog.exec_() == QDialog.Accepted:
                self.context.images.set_full_images(dialog.processed_images)
                self.update_env()
                self.show_info_message("AI Background Removal", f"Completed!")

    @handle_errors(title="Save Image Error")
    def save_image_as_tif(self, save_mode):
        default_path = os.path.join(self.context.last_save_dir, f"{self.context.sample_name}.tif")
        filename, _ = QFileDialog.getSaveFileName(self, "Save images", default_path, "TIFF files (*.tif)")
        if not filename:
            return

        self.context.last_save_dir = os.path.dirname(filename)
        sample_name = os.path.splitext(os.path.basename(filename))[0]
        data_io.save_tif(self.context.last_save_dir, sample_name, self.context.get_images(), save_mode)

        self.show_info_message("Save image", f"Success! TIF images saved to '{self.context.last_save_dir}'.")
    
    def show_info_message(self, title, info):
        if isinstance(info, dict):
            text = "\n".join(f"{key}: {value}" for key, value in info.items())
        else:
            text = info
        msg = QMessageBox(self)
        msg.setFont(QFont("Calibri", 14))
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TXM_ToolBox()
    window.show()
    sys.exit(app.exec_())
