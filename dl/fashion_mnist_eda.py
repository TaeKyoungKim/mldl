"""TensorFlow Fashion-MNIST 데이터 로드, EDA, 특정 샘플 시각화.

기본 실행:
    python fashion_mnist_eda.py

다른 샘플 확인:
    python fashion_mnist_eda.py --sample-index 1500 --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output" / "fashion_mnist_eda"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist() -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    try:
        from tensorflow.keras.datasets import fashion_mnist
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow가 설치되어 있지 않습니다. TensorFlow가 지원하는 Python 환경에서 "
            "`pip install tensorflow`을 실행한 뒤 다시 시도하세요."
        ) from exc
    return fashion_mnist.load_data()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fashion-MNIST 데이터 EDA")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=1000,
        help="시각화할 train 데이터의 0-based 인덱스 (기본값: 1000)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true", help="저장한 그래프를 화면에도 표시")
    return parser.parse_args()


def class_distribution(labels: np.ndarray, split: str) -> pd.DataFrame:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return pd.DataFrame(
        {
            "split": split,
            "class_id": np.arange(len(CLASS_NAMES)),
            "class_name": CLASS_NAMES,
            "count": counts,
            "ratio": counts / len(labels),
        }
    )


def save_class_distribution_plot(distribution: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=distribution,
        x="class_name",
        y="count",
        hue="split",
        palette="Set2",
    )
    plt.title("Fashion-MNIST class distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_pixel_distribution_plot(
    x_train: np.ndarray, x_test: np.ndarray, path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    bins = np.arange(257) - 0.5
    plt.hist(
        x_train.ravel(), bins=bins, density=True, alpha=0.55, label="train"
    )
    plt.hist(x_test.ravel(), bins=bins, density=True, alpha=0.45, label="test")
    plt.title("Pixel intensity distribution")
    plt.xlabel("Pixel intensity (0-255)")
    plt.ylabel("Density")
    plt.xlim(-1, 256)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_class_examples(x_train: np.ndarray, y_train: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    for class_id, ax in enumerate(axes.flat):
        index = int(np.flatnonzero(y_train == class_id)[0])
        ax.imshow(x_train[index], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{class_id}: {CLASS_NAMES[class_id]}")
        ax.axis("off")
    fig.suptitle("One training example from each Fashion-MNIST class")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_class_average_images(
    x_train: np.ndarray, y_train: np.ndarray, path: Path
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    for class_id, ax in enumerate(axes.flat):
        mean_image = x_train[y_train == class_id].mean(axis=0)
        ax.imshow(mean_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{class_id}: {CLASS_NAMES[class_id]}")
        ax.axis("off")
    fig.suptitle("Mean image by class")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_sample_image(
    image: np.ndarray, label: int, sample_index: int, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(
        f"Train index {sample_index}\nLabel {label}: {CLASS_NAMES[label]}"
    )
    axes[0].axis("off")

    sns.heatmap(
        image,
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=255,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pixel intensity"},
    )
    axes[1].set_title("28 × 28 pixel values")
    fig.suptitle("Fashion-MNIST sample inspection")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] TensorFlow에서 Fashion-MNIST 데이터 로드")
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

    if not 0 <= args.sample_index < len(x_train):
        raise ValueError(
            f"sample-index는 0 이상 {len(x_train) - 1} 이하여야 합니다."
        )

    print("[2/4] 데이터 구조와 클래스 분포 계산")
    train_distribution = class_distribution(y_train, "train")
    test_distribution = class_distribution(y_test, "test")
    distribution = pd.concat(
        [train_distribution, test_distribution], ignore_index=True
    )
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)

    sample_image = x_train[args.sample_index]
    sample_label = int(y_train[args.sample_index])
    nonzero_ratio = float(np.count_nonzero(sample_image) / sample_image.size)

    summary = {
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "image_shape": list(x_train.shape[1:]),
        "train_dtype": str(x_train.dtype),
        "pixel_min": int(min(x_train.min(), x_test.min())),
        "pixel_max": int(max(x_train.max(), x_test.max())),
        "train_pixel_mean": float(x_train.mean()),
        "train_pixel_std": float(x_train.std()),
        "test_pixel_mean": float(x_test.mean()),
        "test_pixel_std": float(x_test.std()),
        "sample_index": args.sample_index,
        "sample_position_human_readable": args.sample_index + 1,
        "sample_label": sample_label,
        "sample_class_name": CLASS_NAMES[sample_label],
        "sample_pixel_min": int(sample_image.min()),
        "sample_pixel_max": int(sample_image.max()),
        "sample_pixel_mean": float(sample_image.mean()),
        "sample_nonzero_ratio": nonzero_ratio,
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[3/4] EDA 그래프 생성")
    save_class_distribution_plot(
        distribution, output_dir / "class_distribution.png"
    )
    save_pixel_distribution_plot(
        x_train, x_test, output_dir / "pixel_intensity_distribution.png"
    )
    save_class_examples(x_train, y_train, output_dir / "class_examples.png")
    save_class_average_images(
        x_train, y_train, output_dir / "class_average_images.png"
    )
    save_sample_image(
        sample_image,
        sample_label,
        args.sample_index,
        output_dir / f"train_sample_{args.sample_index}.png",
    )

    print("[4/4] 완료")
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print(
        f"인덱스 {args.sample_index}는 사람이 세는 순서로 {args.sample_index + 1}번째이며, "
        f"라벨은 {sample_label} ({CLASS_NAMES[sample_label]})입니다."
    )
    print(f"결과 폴더: {output_dir}")

    if args.show:
        from PIL import Image

        Image.open(output_dir / f"train_sample_{args.sample_index}.png").show()


if __name__ == "__main__":
    main()
