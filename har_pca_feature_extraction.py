"""
================================================================================
UCI HAR Dataset - PCA 피처 추출 실행 진입점 (루트)
================================================================================
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.har_pca_feature_extraction import main

if __name__ == "__main__":
    main()
