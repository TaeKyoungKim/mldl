"""
================================================================================
UCI HAR Dataset - Random Forest 모델링 및 3개 개별 트리 시각화 (루트 진입점)
================================================================================
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.har_random_forest import main

if __name__ == "__main__":
    main()
