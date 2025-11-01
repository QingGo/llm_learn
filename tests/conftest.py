import sys
from pathlib import Path

# 将项目根目录和src目录加入 sys.path，确保模块可被测试导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))