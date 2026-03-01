# CINIC-10 Image Classification Project

This project trains and compares image classification models on the CINIC-10 dataset using PyTorch.

---

## 1. Clone the Repository

```bash
git clone <REPO_URL>
cd <REPO_NAME>
````

---

## 2. Create Python Virtual Environment

Python 3.10 is recommended.

### Mac / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download CINIC-10 Dataset

Download the dataset from:

[https://www.kaggle.com/datasets/mengcius/cinic10](https://www.kaggle.com/datasets/mengcius/cinic10)

Unzip it and place it inside the project directory with the following structure:

```
data/
  cinic10/
    train/
    valid/
    test/
```

Each of the `train`, `valid`, and `test` folders must contain 10 class subfolders.

IMPORTANT:

* Do NOT commit the dataset to GitHub.
* The `data/` folder is ignored by git.

---

## 5. Test Dataset Loading

After placing the dataset correctly, run:

```bash
python -m src.train
```

Expected output example:

```
Using device: cuda
Batch images shape: torch.Size([64, 3, 32, 32])
Batch labels shape: torch.Size([64])
```

If you see similar output, the dataset pipeline is working correctly.

---

## Project Structure

```
src/        - source code
configs/    - experiment configuration files
scripts/    - helper scripts
results/    - experiment summary tables
data/       - dataset (not tracked by git)
runs/       - training outputs (not tracked by git)
```

---

## Next Steps

After confirming dataset loading works:

1. Implement baseline CNN model
2. Implement training loop
3. Run first baseline experiment
4. Record validation and test accuracy
