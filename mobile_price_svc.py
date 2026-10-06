"""
================================================================================
스마트폰 가격 등급 SVC 모델링 및 평가 실행 진입점 (루트)
================================================================================
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.mobile_price_svc import main

if __name__ == "__main__":
    main()
