import sys
sys.path.insert(0, r"C:\Users\evyne\Documents\Shared_brainflow-cerelog\python_package")

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter
print("BrainFlow imported successfully")
print(f"Board description: {BoardShim.get_board_descr(0)}")