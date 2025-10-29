import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保 `import torch_self_attention` 等同级模块可被测试导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))