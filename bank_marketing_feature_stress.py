"""Single RF split + TabDDPM donor proposals + grouped Monte Carlo feature stress.

python bank_marketing_feature_stress.py
No cross-validation, oversampling, or retraining during stress evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent
OFFICIAL_COMMIT = "internal-tabddpm-v1.0"

GROUPS = {
    "demographics": ["age", "job", "marital", "education"],
    "financial": ["balance", "default", "housing", "loan"],
    "campaign": ["contact", "day", "month", "campaign"],
    "history": ["pdays", "previous", "poutcome"],
}
LABELS = {"demographics": "고객 특성", "financial": "금융 상태", "campaign": "현재 캠페인", "history": "과거 접촉 이력"}
METRICS = ["precision", "recall", "f1", "average_precision", "roc_auc", "balanced_accuracy"]


def evaluate(y, probability, threshold=0.5):
    """불균형 분류 평가지표 산출"""
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(
        average_precision=average_precision_score(y, probability),
        roc_auc=roc_auc_score(y, probability),
        precision=precision_score(y, predicted, zero_division=0),
        recall=recall_score(y, predicted, zero_division=0),
        f1=f1_score(y, predicted, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y, predicted),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )


# -----------------------------------------------------------------------------
# 내장 TabDDPM Donor Generator (외부 의존성 없이 자립 구동)
# -----------------------------------------------------------------------------

class TabularDiffusionMLP(nn.Module):
    def __init__(self, data_dim, time_dim=32, hidden_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, data_dim)
        )

    def sinusoidal_embedding(self, timesteps):
        half_dim = self.time_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def forward(self, x, t):
        t_emb = self.sinusoidal_embedding(t)
        t_feat = self.time_mlp(t_emb)
        return self.net(torch.cat([x, t_feat], dim=1))


class TabularDDPMTrainer:
    def __init__(self, data_dim, timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.device = torch.device(device)
        self.timesteps = timesteps
        self.data_dim = data_dim
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.model = TabularDiffusionMLP(data_dim=data_dim).to(self.device)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        return self.sqrt_alphas_cumprod[t].unsqueeze(1) * x_0 + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1) * noise

    def fit(self, dataloader, epochs=10, lr=1e-3):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                b_size = batch_x.size(0)
                t = torch.randint(0, self.timesteps, (b_size,), device=self.device).long()
                noise = torch.randn_like(batch_x)
                x_t = self.q_sample(batch_x, t, noise)

                optimizer.zero_grad()
                pred_noise = self.model(x_t, t)
                loss = criterion(pred_noise, noise)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * b_size
            if epoch % 5 == 0 or epoch == 1:
                print(f"    [DDPM Epoch {epoch:02d}/{epochs}] Loss: {total_loss / len(dataloader.dataset):.5f}", flush=True)

    @torch.no_grad()
    def sample(self, num_samples):
        self.model.eval()
        x = torch.randn(num_samples, self.data_dim, device=self.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=self.device)
            pred_noise = self.model(x, t)
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alphas_cumprod[step]
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
            if step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return x.cpu().numpy()


def generate_minority(train_df, num_donors, output_dir, seed=42, steps=3000, timesteps=100, threads=2,
                      device="cpu", training_scope="all", output_name="diffusion_donors.csv"):
    """
    Train 데이터로부터 TabDDPM을 통해 Realistic Donors(대체 후보 샘플) 생성 및 원본 스키마 복원
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    donor_cache_path = output_dir / output_name

    train_feat = features(train_df)
    numeric = train_feat.select_dtypes(include="number").columns.tolist()
    categorical = train_feat.columns.difference(numeric).tolist()

    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical)
    ])
    xt = pre.fit_transform(train_feat)

    # DDPM 학습
    dataset = TensorDataset(torch.tensor(xt, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # epochs 추정 (steps와 dataset 크기 비례)
    epochs = max(5, min(15, steps // (len(train_df) // 256)))
    ddpm = TabularDDPMTrainer(data_dim=xt.shape[1], timesteps=min(timesteps, 100), device=device)
    ddpm.fit(loader, epochs=epochs, lr=1e-3)

    # Donors 샘플링
    print(f"  → Generating {num_donors:,} diffusion donor proposals...", flush=True)
    synthetic_arr = ddpm.sample(num_samples=num_donors)

    # 원본 도메인(수치형 역변환 및 범주형 argmax 복원)
    num_trans = pre.named_transformers_["num"]
    scaler = num_trans.named_steps["standardscaler"]

    num_part = synthetic_arr[:, :len(numeric)]
    num_restored = scaler.inverse_transform(num_part)

    donors_df = pd.DataFrame(num_restored, columns=numeric)

    # 원본 데이터프레임의 정수형 타입 컬럼 복원 (타입 캐스팅 에러 방지)
    for col in numeric:
        if col in train_df.columns and np.issubdtype(train_df[col].dtype, np.integer):
            donors_df[col] = donors_df[col].round().astype(train_df[col].dtype)
        elif np.issubdtype(train_feat[col].dtype, np.integer):
            donors_df[col] = donors_df[col].round().astype(train_feat[col].dtype)

    # 범주형 복원
    cat_trans = pre.named_transformers_["cat"]
    encoder = cat_trans.named_steps["onehotencoder"]
    cat_part = synthetic_arr[:, len(numeric):]

    offset = 0
    for col, categories in zip(categorical, encoder.categories_):
        n_cats = len(categories)
        probs = cat_part[:, offset:offset + n_cats]
        chosen_idx = np.argmax(probs, axis=1)
        donors_df[col] = categories[chosen_idx]
        offset += n_cats

    # 원래 열 순서 보존
    donors_df = donors_df[train_df.columns]
    donors_df.to_csv(donor_cache_path, index=False, encoding="utf-8-sig")
    return donors_df


def features(frame):
    frame = frame.copy()
    frame["previously_contacted"] = (frame.pdays.notna() & frame.pdays.ne(-1)).astype(int)
    frame["pdays"] = frame.pdays.replace(-1, np.nan)
    return frame


def replace_group(base, donors, donor_rows, affected_rows, columns):
    """Replace an entire feature block in selected rows, preserving the rest."""
    result = base.copy()
    rows = np.flatnonzero(affected_rows)
    result[np.ix_(rows, columns)] = donors[np.ix_(donor_rows[rows], columns)]
    return result


def prediction_flip_statistics(clean_probability, stressed_probability, threshold=.5):
    """Count BEFORE -> AFTER predictions, independent of the true target."""
    before = np.asarray(clean_probability) >= threshold
    after = np.asarray(stressed_probability) >= threshold
    n00 = int((~before & ~after).sum())
    n01 = int((~before & after).sum())
    n10 = int((before & ~after).sum())
    n11 = int((before & after).sum())
    n = len(before)
    return dict(n_0_to_0=n00, n_0_to_1=n01, n_1_to_0=n10, n_1_to_1=n11,
                flipped_count=n01+n10, flip_pct=100*(n01+n10)/n,
                zero_to_one_pct_all=100*n01/n, one_to_zero_pct_all=100*n10/n,
                zero_to_one_pct_of_original_zero=100*n01/(n00+n01) if n00+n01 else np.nan,
                one_to_zero_pct_of_original_one=100*n10/(n10+n11) if n10+n11 else np.nan)


def encoded_feature_blocks(preprocessor):
    """Map raw feature groups to their fitted numerical/one-hot columns."""
    numeric = list(preprocessor.transformers_[0][2])
    categorical = list(preprocessor.transformers_[1][2])
    blocks = {col: [i] for i, col in enumerate(numeric)}
    if "previously_contacted" in blocks and "pdays" in blocks:
        blocks["pdays"] += blocks["previously_contacted"]
    offset = len(numeric)
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehotencoder"]
    for col, cats in zip(categorical, encoder.categories_):
        blocks[col] = list(range(offset, offset+len(cats)))
        offset += len(cats)
    return blocks


def run(args):
    if args.repeats < 2 or args.donors < args.neighbors:
        raise ValueError("repeats>=2, donors>=neighbors가 필요합니다.")
    if not all(0 < s <= 1 for s in args.levels):
        raise ValueError("스트레스 비율은 (0,1]이어야 합니다.")
    
    # C:/apps/mldl/data/bank/bank-full.csv 우선 탐색 및 검증
    data_path = Path(args.data)
    if not data_path.exists():
        fallback_path = ROOT / "data/bank/bank-full.csv"
        if fallback_path.exists():
            data_path = fallback_path
        else:
            fallback_path2 = ROOT / "data/bank+marketing/bank-full.csv"
            if fallback_path2.exists():
                data_path = fallback_path2

    print(f"▶ Loading data: {data_path.resolve()}", flush=True)
    data = pd.read_csv(data_path, sep=";")
    if data.empty or not data.y.isin(["yes", "no"]).all():
        raise ValueError("y=yes/no 데이터가 필요합니다.")
    
    x = data.drop(columns=["y", "duration"], errors="ignore")
    y = data.y.eq("yes").astype(int)

    ti, vi = train_test_split(np.arange(len(data)), test_size=.2, stratify=y, random_state=args.seed)
    # One fixed, stratified subset is reused in EVERY Monte Carlo condition.
    si = (train_test_split(vi, train_size=args.eval_rows, stratify=y.iloc[vi], random_state=args.seed+1)[0]
          if args.eval_rows < len(vi) else vi)
    assert not np.intersect1d(ti, vi).size and set(si) <= set(vi)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    def save(frame, name):
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    save(pd.DataFrame(dict(source_row_index=ti)), "train_indices")
    save(pd.DataFrame(dict(source_row_index=vi)), "test_indices")
    save(pd.DataFrame(dict(source_row_index=si)), "mc_test_indices")

    train = x.iloc[ti]
    train_features = features(train)
    numeric = train_features.select_dtypes(include="number").columns.tolist()
    categorical = train_features.columns.difference(numeric).tolist()
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical)])
    
    xt = pre.fit_transform(train_features)
    xs = pre.transform(features(x.iloc[si]))

    print(f"Training one Random Forest: {len(ti):,} train / {len(vi):,} test; MC subset={len(si):,}", flush=True)
    model = RandomForestClassifier(n_estimators=args.trees, class_weight=None, random_state=args.seed, n_jobs=args.jobs)
    model.fit(xt, y.iloc[ti])
    joblib.dump(dict(preprocessor=pre, model=model, raw_features=x.columns.tolist()), out / "random_forest.joblib")

    clean_probability = model.predict_proba(xs)[:, 1]
    baseline = evaluate(y.iloc[si], clean_probability, .5)
    full = evaluate(y.iloc[vi], model.predict_proba(pre.transform(features(x.iloc[vi])))[:, 1], .5)
    save(pd.DataFrame([dict(scope="full_test", rows=len(vi), **full),
                       dict(scope="fixed_mc_subset", rows=len(si), **baseline)]), "clean_baselines")
    save(pd.DataFrame(dict(source_row_index=si, y=y.iloc[si].to_numpy(), probability=clean_probability)), "clean_mc_predictions")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    print("Training diffusion on ALL training features; no y, no test rows", flush=True)
    start = perf_counter()
    donors = generate_minority(train, args.donors, out / "diffusion", seed=args.seed+2,
                               steps=args.steps, timesteps=args.timesteps, threads=args.threads,
                               device=device, training_scope="all training features; no labels",
                               output_name="diffusion_donors.csv")
    diffusion_seconds = perf_counter()-start
    xd = pre.transform(features(donors))
    assert np.isfinite(xd).all()

    quality = []
    for col in x.columns:
        if col in categorical:
            a, b = train[col].value_counts(normalize=True), donors[col].value_counts(normalize=True)
            index = a.index.union(b.index)
            value = .5*np.abs(a.reindex(index, fill_value=0)-b.reindex(index, fill_value=0)).sum()
            metric = "total_variation_distance"
            assert set(donors[col]) <= set(train[col])
        else:
            value, metric = ks_2samp(train[col], donors[col]).statistic, "KS_distance"
        quality.append(dict(feature=col, metric=metric, value=value))
    save(pd.DataFrame(quality), "diffusion_donor_quality")

    # Column blocks encode valid full categorical values, not fractional one-hot mixtures.
    blocks = encoded_feature_blocks(pre)
    all_columns = np.arange(xs.shape[1])
    ys = y.iloc[si].to_numpy()
    records = []
    rng = np.random.default_rng(args.seed+100)
    # Paired Monte Carlo draws: same affected rows at each level across groups.
    row_uniforms = rng.random((args.repeats, len(si)))
    candidate_draws = rng.integers(args.neighbors, size=(args.repeats, len(si)))

    context_distances = []
    examples = []
    for group, raw_columns in GROUPS.items():
        columns = np.unique(np.concatenate([blocks[c] for c in raw_columns]))
        context = np.setdiff1d(all_columns, columns)
        print(f"Group {group}: match diffusion proposals on UNCHANGED features", flush=True)
        # This is approximate donor matching, NOT exact conditional diffusion.
        nearest = NearestNeighbors(n_neighbors=args.neighbors, algorithm="brute", n_jobs=args.jobs)
        nearest.fit(xd[:, context])
        distances, candidates = nearest.kneighbors(xs[:, context])
        context_distances.append(dict(group=group, mean_nearest_distance=float(distances[:, 0].mean()),
                                      mean_candidate_distance=float(distances.mean())))
        donor_raw = donors[raw_columns].to_numpy()
        original_raw = x.iloc[si][raw_columns].to_numpy()
        for repeat in range(args.repeats):
            chosen = candidates[np.arange(len(si)), candidate_draws[repeat]]
            raw_changed = donor_raw[chosen] != original_raw
            for level in args.levels:
                mask = row_uniforms[repeat] < level
                corrupted = replace_group(xs, xd, chosen, mask, columns)
                probability = model.predict_proba(corrupted)[:, 1]
                result = evaluate(ys, probability, .5)
                records.append(dict(group=group, repeat=repeat, level=level,
                                    selected_row_pct=100*mask.mean(),
                                    actual_changed_row_pct=100*np.mean(mask & raw_changed.any(axis=1)),
                                    changed_group_cells_pct=100*np.mean(raw_changed & mask[:, None]),
                                    mean_absolute_probability_change=float(np.abs(probability-clean_probability).mean()),
                                    **prediction_flip_statistics(clean_probability, probability),
                                    **result))
                if repeat == 0 and level == max(args.levels):
                    example = x.iloc[si].copy().reset_index(drop=True)
                    example.loc[mask, raw_columns] = donors.iloc[chosen[mask]][raw_columns].to_numpy()
                    example["y_original"] = np.where(ys, "yes", "no")
                    example["source_row_index"] = si
                    example.to_csv(out / f"example_stress_{group}.csv", index=False, encoding="utf-8-sig")
                    examples.append(dict(group=group, level=level, precision=result["precision"], recall=result["recall"], f1=result["f1"], average_precision=result["average_precision"]))
            if (repeat+1) % 10 == 0:
                print(f"  {group}: Monte Carlo {repeat+1}/{args.repeats}", flush=True)
        save(pd.DataFrame(records), "mc_trials")

    trials = pd.DataFrame(records)
    summaries = []
    for (group, level), sub in trials.groupby(["group", "level"]):
        row = dict(group=group, level=level, trials=len(sub),
                   actual_changed_row_pct=sub.actual_changed_row_pct.mean(),
                   changed_group_cells_pct=sub.changed_group_cells_pct.mean())
        for metric in METRICS:
            values = sub[metric]
            row.update({f"{metric}_mean": values.mean(), f"{metric}_std": values.std(),
                        f"{metric}_p025": values.quantile(.025), f"{metric}_p975": values.quantile(.975),
                        f"{metric}_change_from_clean": values.mean()-baseline[metric]})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    save(summary, "stress_summary")
    save(pd.DataFrame(context_distances), "proposal_context_distances")
    save(pd.DataFrame(examples), "example_metrics")

    if "Malgun Gothic" in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, metric, name in zip(axes, ["f1", "average_precision", "recall"], ["F1", "AP", "Recall"]):
        for group in GROUPS:
            sub = summary[summary.group.eq(group)].sort_values("level")
            levels = np.r_[0, sub.level.to_numpy()]*100
            means = np.r_[baseline[metric], sub[f"{metric}_mean"]]
            lo = np.r_[baseline[metric], sub[f"{metric}_p025"]]
            hi = np.r_[baseline[metric], sub[f"{metric}_p975"]]
            ax.plot(levels, means, "o-", label=LABELS[group])
            ax.fill_between(levels, lo, hi, alpha=.10)
        ax.set(xlabel="해당 집단을 대체하도록 선택한 테스트 행 (%)", ylabel=name, title=name)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle(f"Diffusion + Monte Carlo 피처 집단 스트레스 · RF 고정 · {args.repeats}회 반복\n음영: 반복 결과의 2.5~97.5 백분위 범위 (평균의 신뢰구간 아님)", fontsize=14)
    actual = summary[summary.level.eq(max(args.levels))].set_index("group")
    note = "최대 대체 비율에서 실제 값이 바뀐 행: " + " / ".join(
        f"{LABELS[g]} {actual.loc[g, 'actual_changed_row_pct']:.1f}%" for g in GROUPS)
    fig.text(.5, .012, note, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(out / "group_stress_curves.png", dpi=180)
    plt.close(fig)

    # Select no model/threshold using this test: these are descriptive stress results.
    severe = summary[summary.level.eq(max(args.levels))].sort_values("average_precision_change_from_clean")
    save(severe, "severe_stress_ranking")

    config = dict(data=str(data_path.resolve()), data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  split="one stratified 80/20 split; no CV", seed=args.seed, rf_trees=args.trees,
                  train_rows=len(ti), full_test_rows=len(vi), fixed_mc_rows=len(si), repeats=args.repeats,
                  levels=args.levels, groups=GROUPS, diffusion_training_rows=len(ti), donors=args.donors,
                  diffusion_steps=args.steps, diffusion_timesteps=args.timesteps, device=device,
                  nearest_candidates=args.neighbors, threshold=.5, duration_excluded=True,
                  oversampling=False, diffusion_seconds=diffusion_seconds, official_commit=OFFICIAL_COMMIT)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = f"""# 피처 집단 스트레스 테스트 (Diffusion + Monte Carlo)

데이터셋: `{data_path.resolve()}` ({len(data):,}행)

## 1. 기준 성능 (클린 테스트셋)
- F1: {baseline['f1']:.4f}
- AP: {baseline['average_precision']:.4f}
- Recall: {baseline['recall']:.4f}
- Precision: {baseline['precision']:.4f}

## 2. 100% 극단 대체 시 피처 그룹별 성능 하락 순위
```
{severe[['group', 'actual_changed_row_pct', 'f1_mean', 'average_precision_mean', 'average_precision_change_from_clean']].round(4).to_string(index=False)}
```
"""
    (out / "README.md").write_text(report, encoding="utf-8")
    print("\nClean MC baseline:", baseline)
    print(severe[["group", "actual_changed_row_pct", "f1_mean", "average_precision_mean", "average_precision_change_from_clean"]].round(4).to_string(index=False))
    print(f"\nSaved: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bank Marketing Feature Stress Testing with Diffusion proposals")
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    default_output = ROOT / "output/bank_feature_stress_mc"

    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--donors", type=int, default=2048)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--eval-rows", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--levels", type=float, nargs="+", default=[.1, .25, .5, .75, 1.])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run(parser.parse_args())
