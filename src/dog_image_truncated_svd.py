"""
================================================================================
강아지(Dog) 이미지 Truncated SVD 기반 손실 압축 및 저순위 근사(Low-Rank Approximation)
================================================================================
- 원본 참조 코드: C:/Users/ace/Downloads/human_activity_recognition_truncated_svd.py
- 핵심 원리:
  - 2차원 채널 이미지 행렬 A (H × W)에 대해 특이값 분해(SVD)를 수행:
      A = U Σ V^T = Σ_{i=1}^{r} σ_i u_i v_i^T
  - 상위 k개 특이값(Singular Values) 및 특이벡터만 유지(Truncated SVD):
      A_k = U_k Σ_k V_k^T (최적의 저순위 근사, Eckart-Young-Mirsky 정리)
  - 저장 데이터:
      원본: H × W 개 부동소수점
      압축: k × (H + W + 1) 개 부동소수점
      압축률(Compression Ratio) = (H × W) / [k × (H + W + 1)]
- 컬러(RGB 3채널) 및 그레이스케일 압축 동시 지원
- 평가 지표: 복원 MSE, PSNR (Peak Signal-to-Noise Ratio), 압축률, 설명 분산 비율
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD

# 윈도우 콘솔 한글 인코딩 대응
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "dog.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "dog_svd"

DOG_IMAGE_URL = (
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099712_Labrador_retriever.JPEG"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="강아지 이미지를 Truncated SVD를 이용하여 손실 압축 및 복원합니다."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="압축할 강아지 이미지 파일 경로 (기본값: data/dog.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="압축 결과 및 시각화를 저장할 폴더 (기본값: output/dog_svd)",
    )
    parser.add_argument(
        "--components-list",
        type=int,
        nargs="+",
        default=[5, 20, 50, 100],
        help="비교할 주성분(k, rank) 개수 목록 (기본값: 5 20 50 100)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="흑백(Grayscale) 단일 채널로 변환하여 압축 수행",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="재현성을 위한 난수 시드 (기본값: 42)",
    )
    return parser.parse_args()


def ensure_image_exists(image_path: Path) -> Path:
    """
    이미지 파일이 없을 경우 오픈 샘플 강아지 이미지를 자동 다운로드합니다.
    """
    if not image_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[다운로드] 샘플 강아지 이미지 다운로드 중: {image_path.name}...")
        try:
            req = urllib.request.Request(DOG_IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(image_path, "wb") as f:
                f.write(resp.read())
            print(f"  >> 다운로드 완료: {image_path} ({os.path.getsize(image_path):,} bytes)")
        except Exception as e:
            print(f"  [경고] 다운로드 실패({e}), 자체 인공 강아지 패턴을 생성합니다.")
            img = Image.new("RGB", (400, 300), color=(220, 190, 150))
            img.save(image_path)
    return image_path


def load_image_data(image_path: Path, grayscale: bool = False) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    이미지를 로드하여 [0.0, 1.0] 범위의 정규화된 float32 넘파이 배열로 변환합니다.
    """
    img = Image.open(image_path)
    if grayscale:
        img = img.convert("L")
        img_arr = np.array(img, dtype=np.float32) / 255.0
        # 형태: (H, W, 1)로 통일
        img_arr = img_arr[:, :, np.newaxis]
    else:
        img = img.convert("RGB")
        img_arr = np.array(img, dtype=np.float32) / 255.0

    return img_arr, img_arr.shape


def compress_and_reconstruct_channel(
    channel: np.ndarray, n_components: int, random_state: int
) -> tuple[TruncatedSVD, np.ndarray, np.ndarray]:
    """
    단일 2D 채널(H × W)에 대해 TruncatedSVD를 적용하고 복원합니다.
    - svd.fit_transform(channel) -> U_k * Σ_k (H × k)
    - svd.components_            -> V_k^T (k × W)
    - svd.inverse_transform(...) -> 복원된 행렬 A_k (H × W)
    """
    max_components = min(channel.shape) - 1
    k = min(n_components, max_components)

    svd = TruncatedSVD(
        n_components=k,
        algorithm="randomized",
        n_iter=7,
        random_state=random_state,
    )
    compressed = svd.fit_transform(channel).astype(np.float32)
    reconstructed = svd.inverse_transform(compressed).astype(np.float32)
    return svd, compressed, reconstructed


def compress_full_image(
    image: np.ndarray, n_components: int, random_state: int
) -> tuple[list[TruncatedSVD], list[np.ndarray], np.ndarray, dict[str, float]]:
    """
    이미지의 모든 채널(R, G, B 또는 Grayscale)에 대해 SVD 압축을 수행하고 지표를 계산합니다.
    """
    h, w, c = image.shape
    reconstructed_image = np.zeros_like(image)
    svd_models = []
    compressed_channels = []

    total_original_values = image.size
    total_stored_values = 0
    total_explained_variance = 0.0

    for ch_idx in range(c):
        channel_data = image[:, :, ch_idx]
        svd, compressed, reconstructed = compress_and_reconstruct_channel(
            channel_data, n_components, random_state
        )
        reconstructed_image[:, :, ch_idx] = reconstructed
        svd_models.append(svd)
        compressed_channels.append(compressed)

        # 저장 필요 데이터: U_k*Σ_k (H × k) + V_k^T (k × W)
        total_stored_values += compressed.size + svd.components_.size
        total_explained_variance += float(svd.explained_variance_ratio_.sum())

    # 픽셀 값 범위 클리핑 [0.0, 1.0]
    reconstructed_image = np.clip(reconstructed_image, 0.0, 1.0)

    # 지표 산출
    mse = float(np.mean((image - reconstructed_image) ** 2))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
    compression_ratio = float(total_original_values / total_stored_values)
    savings_percent = float((1.0 - total_stored_values / total_original_values) * 100)
    avg_explained_variance = float(total_explained_variance / c)

    metrics = {
        "k": n_components,
        "mse": mse,
        "psnr_db": psnr,
        "compression_ratio": compression_ratio,
        "savings_percent": savings_percent,
        "explained_variance_ratio": avg_explained_variance,
        "original_values": total_original_values,
        "stored_values": total_stored_values,
    }
    return svd_models, compressed_channels, reconstructed_image, metrics


def save_and_visualize_results(
    image: np.ndarray,
    components_list: list[int],
    output_dir: Path,
    random_state: int,
    grayscale: bool,
) -> list[dict[str, float]]:
    """
    단계별 SVD 압축 실행, 복원 이미지 파일 저장, NPZ 바이너리 저장, 비교 차트 생성
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w, c = image.shape
    all_summaries = []
    reconstructed_images = []

    print(f"\n[SVD 압축 진행]: 원본 이미지 해상도: {w} × {h} (총 {image.size:,} 픽셀 값)")
    print("-" * 75)
    print(f"{'k (특이값 수)':<12}{'복원 MSE':<14}{'PSNR (dB)':<14}{'압축률':<14}{'용량 절감율':<14}")
    print("-" * 75)

    for k in components_list:
        svds, compressed_list, recon_img, metrics = compress_full_image(
            image, k, random_state
        )
        all_summaries.append(metrics)
        reconstructed_images.append(recon_img)

        comp_ratio_str = f"{metrics['compression_ratio']:.2f}배"
        savings_str = f"{metrics['savings_percent']:.1f}%"
        print(
            f"{metrics['k']:<12}{metrics['mse']:<14.6f}{metrics['psnr_db']:<14.2f}"
            f"{comp_ratio_str:<14}{savings_str:<14}"
        )

        # 1. 복원된 이미지 파일 저장 (JPEG)
        recon_uint8 = (recon_img.squeeze() * 255).astype(np.uint8) if grayscale else (recon_img * 255).astype(np.uint8)
        img_out = Image.fromarray(recon_uint8)
        img_path = output_dir / f"dog_svd_k{k}.jpg"
        img_out.save(img_path, quality=95)

        # 2. 압축된 바이너리 데이터 저장 (NPZ)
        npz_dict = {}
        for idx, (comp, svd) in enumerate(zip(compressed_list, svds)):
            ch_name = f"ch_{idx}"
            npz_dict[f"{ch_name}_U_sigma"] = comp
            npz_dict[f"{ch_name}_Vt"] = svd.components_
        np.savez_compressed(output_dir / f"dog_compressed_k{k}.npz", **npz_dict)

    print("-" * 75)

    # 3. 종합 요약 JSON 저장
    summary_path = output_dir / "dog_svd_summary.json"
    summary_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  * 압축 통계 JSON 저장 완료: {summary_path}")

    # ==========================================================================
    # 4. 시각화 1: 원본 vs k별 복원 이미지 갤러리 비교 차트
    # ==========================================================================
    n_plots = 1 + len(components_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))

    # 원본
    orig_display = image.squeeze() if grayscale else image
    cmap = "gray" if grayscale else None
    axes[0].imshow(orig_display, cmap=cmap)
    axes[0].set_title(f"원본 이미지\n(100% 해상도, {w}×{h})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # k별 복원
    for i, (k, recon_img, metrics) in enumerate(
        zip(components_list, reconstructed_images, all_summaries), start=1
    ):
        display_img = recon_img.squeeze() if grayscale else recon_img
        axes[i].imshow(display_img, cmap=cmap)
        axes[i].set_title(
            f"k = {k}\nPSNR: {metrics['psnr_db']:.1f} dB\n압축: {metrics['compression_ratio']:.1f}x (절감 {metrics['savings_percent']:.1f}%)",
            fontsize=10.5,
        )
        axes[i].axis("off")

    plt.suptitle("강아지 이미지 Truncated SVD(저순위 근사) 압축률 및 화질 복원 비교", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    comp_chart_path = output_dir / "dog_svd_comparison.png"
    plt.savefig(comp_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  * 이미지 복원 비교 차트 저장 완료: {comp_chart_path}")

    # ==========================================================================
    # 5. 시각화 2: 특이값 감쇠(Singular Value Decay) & 화질 트레이드오프
    # ==========================================================================
    # 전체 SVD 계산 (특이값 스펙트럼 분석용)
    first_ch = image[:, :, 0]
    full_svd = TruncatedSVD(n_components=min(h, w) - 1, random_state=random_state)
    full_svd.fit(first_ch)
    s_vals = full_svd.singular_values_
    cum_energy = np.cumsum(s_vals**2) / np.sum(s_vals**2) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 특이값 크기 감쇠 곡선
    ax1.plot(range(1, len(s_vals) + 1), s_vals, color="#2E86AB", lw=2)
    for k in components_list:
        if k <= len(s_vals):
            ax1.axvline(k, color="#F24236", ls="--", alpha=0.7)
            ax1.scatter([k], [s_vals[k - 1]], color="#F24236", s=40, zorder=5)
    ax1.set_title("특이값 스펙트럼 감쇠 곡선 (Singular Value Spectrum)", fontsize=12)
    ax1.set_xlabel("특이값 순번 (k)", fontsize=10)
    ax1.set_ylabel("특이값 크기 (σ_i)", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls=":", alpha=0.6)

    # 압축률 vs PSNR 트레이드오프 곡선
    k_vals = [m["k"] for m in all_summaries]
    psnr_vals = [m["psnr_db"] for m in all_summaries]
    ratio_vals = [m["compression_ratio"] for m in all_summaries]

    ax2.plot(k_vals, psnr_vals, marker="o", color="#E63946", lw=2, label="PSNR 화질 (dB)")
    ax2.set_xlabel("주성분 수 (k)", fontsize=10)
    ax2.set_ylabel("PSNR 화질 (dB, 높을수록 우수)", color="#E63946", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#E63946")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(k_vals, ratio_vals, marker="s", color="#1D3557", lw=2, ls="--", label="압축률 (배)")
    ax2_twin.set_ylabel("압축률 (x 배율, 높을수록 경량)", color="#1D3557", fontsize=10)
    ax2_twin.tick_params(axis="y", labelcolor="#1D3557")

    ax2.set_title("주성분(k)에 따른 화질(PSNR) vs 압축률 트레이드오프", fontsize=12)
    plt.tight_layout()
    decay_chart_path = output_dir / "dog_svd_tradeoff.png"
    plt.savefig(decay_chart_path, dpi=300)
    plt.close()
    print(f"  * 특이값 감쇠 및 트레이드오프 차트 저장 완료: {decay_chart_path}")

    return all_summaries


def main() -> None:
    args = parse_args()
    print("#" * 80)
    print("  DOG IMAGE COMPRESSION USING TRUNCATED SVD (LOW-RANK APPROXIMATION)")
    print("#" * 80)

    # 1. 이미지 로드 및 검증
    img_path = ensure_image_exists(args.image_path.resolve())
    image_arr, shape = load_image_data(img_path, grayscale=args.grayscale)
    print(f"\n[1] 대상 이미지 로드: {img_path}")
    print(f"  * 해상도 (H × W × C): {shape[0]} × {shape[1]} × {shape[2]}")
    print(f"  * 모드: {'그레이스케일 (1채널)' if args.grayscale else '컬러 (RGB 3채널)'}")

    # 2. 압축, 복원 및 시각화 저장
    print(f"\n[2] Truncated SVD 기반 단계별 압축 수행 (k = {args.components_list})...")
    t0 = time.time()
    summaries = save_and_visualize_results(
        image_arr,
        args.components_list,
        args.output_dir.resolve(),
        args.random_state,
        args.grayscale,
    )
    duration = time.time() - t0

    print(f"\n[3] 모든 압축 및 시각화 생성 완료! (소요 시간: {duration:.2f}초)")
    print(f"  * 결과 디렉터리: {args.output_dir.resolve()}")
    print("#" * 80)


if __name__ == "__main__":
    main()
