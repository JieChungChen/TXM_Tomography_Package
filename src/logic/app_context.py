import os


class AppContext:
    def __init__(self):
        self.sample_name = None
        self.last_load_dir = "."
        self.last_save_dir = "."

        self.images = None
        self.mode = None

        try:
            import torch
            self.ai_available = True
        except ImportError:
            self.ai_available = False

    def set_from_file(self, filepath, mode):
        self.sample_name = os.path.splitext(os.path.basename(filepath))[0]
        self.last_load_dir = os.path.dirname(filepath)
        self.mode = mode

    def set_from_folder(self, folderpath, mode):
        self.sample_name = os.path.basename(folderpath)
        self.last_load_dir = os.path.dirname(folderpath)
        self.mode = mode

    def get_images(self):
        return self.images.get_full_images()

    def get_image_size(self):
        if self.images is not None:
            return self.images.get_image(0).shape
        return None

    @property
    def metadata(self):
        return self.images.metadata if self.images else {}

    @property
    def has_data(self):
        return self.images is not None