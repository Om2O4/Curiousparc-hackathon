import torch
import torch.nn as nn


# ============================================================
# CARDIACAI
# 12-LEAD ECG RESNET + SE ATTENTION
# ============================================================


# ============================================================
# CONVOLUTIONAL BLOCK
# ============================================================

class ConvBNAct(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None
    ):

        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),

            nn.BatchNorm1d(
                out_channels
            ),

            nn.GELU()
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# SQUEEZE-AND-EXCITATION BLOCK
# ============================================================

class SEBlock(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(

            nn.Linear(
                channels,
                hidden
            ),

            nn.GELU(),

            nn.Linear(
                hidden,
                channels
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        batch, channels, _ = x.shape

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        y = self.pool(x)

        y = y.view(
            batch,
            channels
        )

        # ----------------------------------------------------
        # Learn channel importance
        # ----------------------------------------------------

        y = self.fc(y)

        y = y.view(
            batch,
            channels,
            1
        )

        return x * y


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout=0.0
    ):

        super().__init__()

        # ----------------------------------------------------
        # Main branch
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        # ----------------------------------------------------
        # Squeeze-and-Excitation
        # ----------------------------------------------------

        self.se = SEBlock(
            out_channels
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Skip connection
        # ----------------------------------------------------

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.act(out)

        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        out = self.conv2(out)

        out = self.bn2(out)

        # ----------------------------------------------------
        # Channel attention
        # ----------------------------------------------------

        out = self.se(out)

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        out = self.dropout(out)

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        out = out + identity

        out = self.act(out)

        return out


# ============================================================
# CARDIAC RESNET
# ============================================================

class CardiacResNet(nn.Module):

    def __init__(
        self,
        num_classes=5
    ):

        super().__init__()

        # ====================================================
        # INPUT
        # ====================================================

        # Input:
        #
        # (batch, 12, 5000)
        #
        # 12 ECG leads
        # 5000 samples
        #
        # ====================================================

        self.stem = nn.Sequential(

            nn.Conv1d(
                12,
                64,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.GELU(),

            nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        # ====================================================
        # RESIDUAL STAGE 1
        # ====================================================

        self.layer1 = nn.Sequential(

            ResidualBlock(
                64,
                64,
                stride=1,
                dropout=0.05
            ),

            ResidualBlock(
                64,
                64,
                stride=1,
                dropout=0.05
            )
        )

        # ====================================================
        # RESIDUAL STAGE 2
        # ====================================================

        self.layer2 = nn.Sequential(

            ResidualBlock(
                64,
                128,
                stride=2,
                dropout=0.08
            ),

            ResidualBlock(
                128,
                128,
                stride=1,
                dropout=0.08
            )
        )

        # ====================================================
        # RESIDUAL STAGE 3
        # ====================================================

        self.layer3 = nn.Sequential(

            ResidualBlock(
                128,
                256,
                stride=2,
                dropout=0.10
            ),

            ResidualBlock(
                256,
                256,
                stride=1,
                dropout=0.10
            ),

            ResidualBlock(
                256,
                256,
                stride=1,
                dropout=0.10
            )
        )

        # ====================================================
        # RESIDUAL STAGE 4
        # ====================================================

        self.layer4 = nn.Sequential(

            ResidualBlock(
                256,
                512,
                stride=2,
                dropout=0.15
            ),

            ResidualBlock(
                512,
                512,
                stride=1,
                dropout=0.15
            )
        )

        # ====================================================
        # GLOBAL POOLING
        # ====================================================

        self.global_pool = nn.AdaptiveAvgPool1d(
            1
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                512,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        x = self.stem(x)

        # ----------------------------------------------------
        # Residual stages
        # ----------------------------------------------------

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        x = self.global_pool(x)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        x = self.classifier(x)

        return x


# ============================================================
# MODEL INFORMATION
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# MODEL TEST
# ============================================================

def test_model():

    print("=" * 70)
    print("CARDIACAI - MODEL ARCHITECTURE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:")
    print(device)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating CardiacResNet...")

    model = CardiacResNet(
        num_classes=5
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = count_parameters(
        model
    )

    print()
    print("Trainable parameters:")
    print(f"{parameters:,}")

    # --------------------------------------------------------
    # Create fake ECG batch
    # --------------------------------------------------------

    print()
    print("Creating test input...")

    x = torch.randn(
        8,
        12,
        5000,
        device=device
    )

    print(
        "Input shape:",
        tuple(x.shape)
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print()
    print("Running forward pass...")

    model.eval()

    with torch.no_grad():

        output = model(x)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("Output shape:")
    print(tuple(output.shape))

    print()
    print("Expected:")
    print("(8, 5)")

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert output.shape == (
        8,
        5
    ), (
        f"Wrong output shape: "
        f"{output.shape}"
    )

    assert torch.isfinite(
        output
    ).all(), (
        "Model produced NaN or Inf"
    )

    # --------------------------------------------------------
    # Test backward pass
    # --------------------------------------------------------

    print()
    print("Testing backward pass...")

    model.train()

    x = torch.randn(
        2,
        12,
        5000,
        device=device
    )

    target = torch.randint(
        0,
        2,
        (2, 5),
        device=device
    ).float()

    criterion = nn.BCEWithLogitsLoss()

    output = model(x)

    loss = criterion(
        output,
        target
    )

    loss.backward()

    print(
        f"Test loss: {loss.item():.6f}"
    )

    assert torch.isfinite(
        loss
    ), "Loss is NaN or Inf"

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    gradient_found = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            if torch.isfinite(
                parameter.grad
            ).all():

                gradient_found = True

                break

    assert gradient_found, (
        "No valid gradients found"
    )

    print()
    print("Backward pass PASSED")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL TEST PASSED")
    print("=" * 70)

    print()
    print("Architecture:")
    print("12-lead ECG")
    print("        ↓")
    print("Conv1D Stem")
    print("        ↓")
    print("Residual Blocks")
    print("        ↓")
    print("SE Attention")
    print("        ↓")
    print("Global Average Pooling")
    print("        ↓")
    print("Fully Connected")
    print("        ↓")
    print("5 Outputs")

    print()
    print("Output labels:")
    print(
        "NORM | MI | STTC | HYP | CD"
    )

    print()
    print("Ready for:")
    print("1. Training loop")
    print("2. Weighted loss")
    print("3. Validation metrics")
    print("4. Checkpointing")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_model()