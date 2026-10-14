"""从代码仓库或其父目录运行 pytest 时，使用同一个本地包。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
