# num-classify

Handwritten digit classification and exploration using the MNIST dataset.

---

## 📌 Overview

`num-classify` is a machine learning project focused on exploring, visualizing, and classifying handwritten digits (0–9) using the standard **MNIST** dataset. Built with Python 3.12 and modern scientific libraries, the project leverages [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management and virtual environment orchestration.

---

## 📁 Project Structure

```text
num-classify/
├── data/
│   └── raw/
│       ├── mnist_train.csv     # Training dataset (60,000 samples)
│       └── mnist_test.csv      # Testing dataset (10,000 samples)
├── docs/                       # Project specifications and guidelines
├── src/
│   └── visualize.ipynb         # Notebook for data loading & 28x28 pixel visualization
├── .python-version             # Pinned Python version (3.12)
├── pyproject.toml              # Project dependencies & metadata
├── uv.lock                     # Deterministic dependency lockfile
└── README.md                   # Project documentation
```

---

## 🛠️ Prerequisites & Installation

### 1. Install Git Tools (GitHub Desktop & Git LFS)

Before cloning and setting up the repository, make sure you have installed the required Git tools:

- **[GitHub Desktop](https://desktop.github.com/):** Simplifies cloning, branching, and commit management with an intuitive GUI. Download and install it from [desktop.github.com](https://desktop.github.com/).
- **[Git LFS (Large File Storage)](https://git-lfs.com/):** Essential for versioning large files such as dataset CSVs and model checkpoints without bloating the Git history.
  - Download and install from [git-lfs.com](https://git-lfs.com/), or install via Chocolatey:
    ```powershell
    choco install git-lfs
    ```
  - Once installed, initialize Git LFS on your system by running:
    ```bash
    git lfs install
    ```

---

### 2. Install Chocolatey & `uv` (Windows)

This project uses **[uv](https://github.com/astral-sh/uv)** for fast, reliable package and environment management.

Open **PowerShell as Administrator** and install [Chocolatey](https://chocolatey.org/):

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Once Chocolatey is installed, install `uv`:

```powershell
choco install uv
```

> **Tip (Alternative Install):** You can also install `uv` directly using the official installer:
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```
> On macOS/Linux:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

---

### 3. Clone Repository & Set Up the Environment

Clone the repository (via GitHub Desktop or CLI) and install all dependencies:

```bash
# Navigate to the project directory
cd num-classify

# Synchronize dependencies and configure virtual environment
uv sync
```

---

## 🚀 Usage

### Running Jupyter Notebook

Launch Jupyter to explore the notebooks in `src/`:

```bash
uv run jupyter notebook
```

Alternatively, open `src/visualize.ipynb` directly in your favorite IDE (such as VS Code, Cursor, or PyCharm) using the `.venv` kernel created by `uv`.

### Visualizing Digits

The `src/visualize.ipynb` notebook loads the raw MNIST CSV data, reshapes 784 pixel intensity values into $28 \times 28$ grayscale images, and displays sample digits alongside their corresponding class labels.

---

## 📊 Dataset Details

The dataset files should be placed in `data/raw/` (downloaded from the provided Google Drive link or repository):

- **Features:** 784 grayscale pixel values ($28 \times 28$), ranging from `0` to `255`.
- **Target:** `label` representing the digit class ($0$ through $9$).
- **Sets:**
  - `mnist_train.csv`: Training samples
  - `mnist_test.csv`: Evaluation samples

---

## 📦 Dependencies

- Python `>=3.12`
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter Notebook](https://jupyter.org/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).