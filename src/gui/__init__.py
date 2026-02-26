try:
    import torch
    _torch_available = True
    from src.gui.ai_ref_remover_dialog import AIRefRemoverDialog
    from src.gui.ai_sino_aligner_dialog import SinoAlignerDialog
except ImportError:
    _torch_available = False
    AIRefRemoverDialog = None
    SinoAlignerDialog = None


from src.gui.contrast_dialog import ContrastDialog
from src.gui.duplicates_selector import resolve_duplicates
from src.gui.fbp_viewer import FBPResolutionDialog, FBPViewer
from src.gui.mlem_dialog import MLEMSettingsDialog
from src.gui.manual_alignment import AlignViewer
from src.gui.mosaic_viewer import MosaicPreviewDialog
from src.gui.shift_dialog import ShiftDialog
from src.gui.reference_dialog import ReferenceModeDialog, SplitSliderDialog


__all__ = [
    "ContrastDialog", 
    "resolve_duplicates", 
    "FBPResolutionDialog", 
    "FBPViewer", 
    "AlignViewer", 
    "MLEMSettingsDialog",
    "MosaicPreviewDialog", 
    "ShiftDialog",
    "ReferenceModeDialog",
    "SplitSliderDialog"
    ]

if _torch_available:
    __all__.extend(["AIRefRemoverDialog", "SinoAlignerDialog"])