# 피처 집단 스트레스 테스트 (Diffusion + Monte Carlo)

데이터셋: `C:\apps\mldl\data\bank\bank-full.csv` (45,211행)

## 1. 기준 성능 (클린 테스트셋)
- F1: 0.3750
- AP: 0.3837
- Recall: 0.2586
- Precision: 0.6818

## 2. 100% 극단 대체 시 피처 그룹별 성능 하락 순위
```
       group  actual_changed_row_pct  f1_mean  average_precision_mean  average_precision_change_from_clean
     history                100.0000   0.1611                  0.2614                              -0.1223
    campaign                 99.9333   0.2269                  0.3055                              -0.0782
   financial                100.0000   0.3030                  0.3056                              -0.0782
demographics                 99.8000   0.3519                  0.3510                              -0.0327
```
