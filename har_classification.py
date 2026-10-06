"""
================================================================================
UCI ML Repository: Human Activity Recognition Using Smartphones (ID: 240)
루트 경로 실행 진입점 (src/har_classification.py 실행)
================================================================================
"""

import os
import sys

# src 디렉터리 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.har_classification import main

if __name__ == "__main__":
    main()
