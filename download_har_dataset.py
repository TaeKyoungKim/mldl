"""
================================================================================
UCI ML Repository: Human Activity Recognition Using Smartphones (ID: 240)
데이터셋 다운로드 및 피처명 중복 해결/분석 스크립트 (루트 진입점)
================================================================================
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.download_har_dataset import download_har

if __name__ == "__main__":
    download_har()
