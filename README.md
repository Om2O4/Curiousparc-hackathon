# PCG Heart Sound Classification (FunnelNet)

A clean, standalone implementation for training a Phonocardiogram (PCG) heart sound classification model using the PhysioNet / CinC 2016 dataset.

## Project Structure

```text
FunnelNet/
├── dataset/
│   └── cinc/               # Place extracted CinC 2016 dataset folders here
├── train_pcg.py            # Main standalone script for training the PCG model
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup & Training

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   Place the downloaded CinC 2016 dataset subfolders inside `dataset/cinc/`:
   ```text
   dataset/cinc/
   ├── training-a/
   ├── training-b/
   ...
   ```

3. **Run Training**:
   ```bash
   py train_pcg.py
   ```
