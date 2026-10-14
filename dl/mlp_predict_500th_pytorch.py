import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt



def main():

    # ============================================================
    # 1. 데이터셋 로드 및 전처리
    # ============================================================

    # ToTensor()
    # 1) 이미지 → Tensor 변환
    # 2) 픽셀값 0~255 → 0.0~1.0으로 변환
    transform = transforms.ToTensor()

    # Fashion-MNIST 학습 데이터
    full_train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Fashion-MNIST 테스트 데이터
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )



    # Fashion-MNIST 클래스 이름
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]



    # ============================================================
    # 2. Train / Validation 데이터 분리
    # TensorFlow의 validation_split=0.1과 비슷하게 구성
    # ============================================================

    train_size = int(len(full_train_dataset) * 0.9)
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )



    # ============================================================
    # 3. DataLoader 생성
    # batch_size=64
    # ============================================================

    batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )



    # ============================================================
    # 4. DNN 모델 구조 정의
    #
    # Input(28x28)
    # → Flatten(784)
    # → Hidden 1(128, ReLU)
    # → Hidden 2(64, ReLU)
    # → Output(10)
    # ============================================================

    class FashionDNN(nn.Module):

        def __init__(self):
            super().__init__()

            self.flatten = nn.Flatten()

            self.fc1 = nn.Linear(784, 128)
            self.relu1 = nn.ReLU()

            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()

            self.fc3 = nn.Linear(64, 10)



        def forward(self, x):

            # (batch, 1, 28, 28)
            # →
            # (batch, 784)
            x = self.flatten(x)

            x = self.fc1(x)
            x = self.relu1(x)

            x = self.fc2(x)
            x = self.relu2(x)

            # 출력층
            x = self.fc3(x)

            return x



    # 모델 생성
    model = FashionDNN()

    # ============================================================
    # PyTorch 초기화를 Keras(Glorot Uniform/zeros)와 동일하게 설정
    # ============================================================
    for layer in [model.fc1, model.fc2, model.fc3]:
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    print("\n모델 구조")
    print(model)



    # ============================================================
    # 5. Loss / Optimizer 설정
    # ============================================================

    # TensorFlow:
    # loss='sparse_categorical_crossentropy'
    #
    # PyTorch:
    # CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    # TensorFlow:
    # optimizer='adam'
    #
    # PyTorch:
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )



    # ============================================================
    # 6. 모델 학습
    # ============================================================

    epochs = 5

    print("\n모델 학습 시작...")



    for epoch in range(epochs):

        # -------------------------
        # Training
        # -------------------------
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0



        for images, labels in train_loader:

            # 이전 gradient 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(images)

            # Loss 계산
            loss = criterion(outputs, labels)

            # 역전파
            loss.backward()

            # Weight 업데이트
            optimizer.step()



            train_loss += loss.item()

            # 가장 높은 값을 가진 클래스 선택
            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)

            train_correct += (
                predicted == labels
            ).sum().item()



        train_accuracy = (
            100 * train_correct / train_total
        )



        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0



        # gradient 계산하지 않음
        with torch.no_grad():

            for images, labels in val_loader:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                _, predicted = torch.max(
                    outputs,
                    1
                )

                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()



        val_accuracy = (
            100 * val_correct / val_total
        )



        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss / len(train_loader):.4f} "
            f"Train Accuracy: {train_accuracy:.2f}% "
            f"Val Loss: {val_loss / len(val_loader):.4f} "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )



    # ============================================================
    # 7. 테스트 데이터 500번째 데이터 예측
    # Index = 499
    # ============================================================

    target_index = 499

    image, actual_label = test_dataset[target_index]

    actual_class_name = class_names[actual_label]



    # 모델 평가 모드
    model.eval()



    with torch.no_grad():

        # 현재 shape
        # (1, 28, 28)
        #
        # batch 차원 추가
        # →
        # (1, 1, 28, 28)

        sample_image = image.unsqueeze(0)

        # 모델 예측
        output = model(sample_image)

        # PyTorch의 CrossEntropyLoss에서는
        # 모델 내부에 Softmax를 넣지 않기 때문에
        # 예측할 때 따로 Softmax 적용
        probabilities = torch.softmax(
            output,
            dim=1
        )

        # 가장 확률 높은 클래스
        predicted_label = torch.argmax(
            probabilities,
            dim=1
        ).item()

        predicted_class_name = (
            class_names[predicted_label]
        )

        confidence = (
            probabilities[0][predicted_label].item()
            * 100
        )



    # ============================================================
    # 8. 콘솔 결과 출력
    # ============================================================

    print("\n" + "=" * 60)

    print(
        f"테스트 데이터 "
        f"[{target_index + 1}번째] 샘플 예측 결과"
    )

    print(
        f"실제 정답 (Actual)    : "
        f"[{actual_label}] "
        f"{actual_class_name}"
    )

    print(
        f"예측 결과 (Predicted) : "
        f"[{predicted_label}] "
        f"{predicted_class_name} "
        f"(신뢰도: {confidence:.2f}%)"
    )

    print(
        f"일치 여부             : "
        f"{'정답 (O)' if actual_label == predicted_label else '오답 (X)'}"
    )

    print("=" * 60 + "\n")



    # ============================================================
    # 9. 결과 시각화
    # ============================================================

    plt.figure(figsize=(9, 4))



    # ------------------------------------------------
    # (1) 500번째 이미지
    # ------------------------------------------------

    plt.subplot(1, 2, 1)

    # Tensor → numpy
    plt.imshow(
        image.squeeze().numpy(),
        cmap='gray'
    )

    is_correct = (
        actual_label == predicted_label
    )

    title_color = (
        'blue'
        if is_correct
        else 'red'
    )

    plt.title(
        f"Actual: {actual_class_name}\n"
        f"Pred: {predicted_class_name} "
        f"({confidence:.1f}%)",
        color=title_color
    )

    plt.axis('off')



    # ------------------------------------------------
    # (2) 클래스별 확률 분포
    # ------------------------------------------------

    plt.subplot(1, 2, 2)

    probs = (
        probabilities[0]
        .cpu()
        .numpy()
    )



    bar_plot = plt.bar(
        range(10),
        probs,
        color='#777777'
    )



    # 실제 정답
    bar_plot[
        actual_label
    ].set_color('blue')



    # 틀린 경우 예측값 빨간색
    if not is_correct:

        bar_plot[
            predicted_label
        ].set_color('red')



    plt.xticks(
        range(10),
        class_names,
        rotation=45,
        ha='right'
    )

    plt.ylim([0, 1])

    plt.ylabel("Probability")

    plt.title(
        "Class Probability Distribution"
    )



    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()