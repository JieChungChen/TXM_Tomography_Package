"""錯誤處理與記錄用的裝飾器。"""
import functools
import traceback
from PyQt5.QtWidgets import QMessageBox


def handle_errors(title="Error"):
    """處理例外並顯示錯誤訊息視窗的裝飾器。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                QMessageBox.critical(self, title, error_msg)
                print(f"Error in {func.__name__}: {e}")
                traceback.print_exc()
        return wrapper
    return decorator