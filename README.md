# Low-Complexity User Matching and Stream-Based Power Allocation for Multicarrier RSMA Systems

This repository contains the Python simulation framework developed to reproduce the results and algorithms proposed in the research of Rate Splitting Multiple Access (RSMA) for multicarrier systems.

## 📄 Reference Paper

This code is a reproduction of the methods and simulations presented in the following article:

> **E. V. Pereira and F. R. M. Lima**, "Low-Complexity User Matching and Stream-Based Power Allocation for Multicarrier RSMA Systems," in *IEEE Access*, vol. 13, pp. 191470-191484, 2025.
> **DOI:** [10.1109/ACCESS.2025.3630122](https://doi.org/10.1109/ACCESS.2025.3630122)

### Keywords
> Resource management; Optimization; Benchmark testing; Receivers; Quality of service; NOMA; 5G mobile communication; Wireless networks; Vectors; 6G mobile communication; RSMA; multicarrier; power allocation; user matching; weighted sum rate.

---

## Disclaimer on Reproducibility

Please note that this simulation relies on Monte Carlo methods with random channel generation (path loss and Rayleigh fading).

While this code faithfully implements the algorithms described in the paper (WGOBUM, SBPA, etc.), **numerical results may diverge slightly** from the exact figures published in the article due to:
1.  Differences in random seeds.
2.  Variations in hyperparameters or simulation size (number of iterations).
3.  The stochastic nature of the wireless channel models.

However, the **trends, relative performance, and convergence behavior** of the algorithms should remain consistent with the published findings.

---

## 🛠️ Installation and Requirements

This project relies on **IBM CPLEX** and **CVXPY** for optimization tasks. These libraries have strict compatibility requirements.

**IMPORTANT:** You must use **Python 3.10**. Newer versions (like Python 3.12 or 3.13) **will not work** with the current CPLEX version. We strongly recommend using **Anaconda** to handle this environment automatically.

### Option 1: Using Anaconda (Recommended)

1.  **Open the Anaconda Prompt** (or your terminal if Conda is in your PATH).
2.  Navigate to the project folder.
3.  Create the environment using the provided `environment.yml` file. This command will download Python 3.10 and all necessary libraries:

    ```bash
    conda env create -f environment.yml
    ```

4.  Activate the environment:

    ```bash
    conda activate rsma_optimization
    ```

### Option 2: Manual Installation (Pip)

If you are not using Anaconda, ensure you have **Python 3.10** installed on your system. Then, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## 🚀 Usage

### 1. Running the Simulation
To execute the main simulation loop, which iterates through different numbers of subcarriers ($N$) and performs the user matching and power allocation:

```bash
python main.py
```
📂 Project Structure
main.py: Main script for system configuration and Monte Carlo loops.

results.py: Script for processing data and plotting figures.

environment.yml: Conda environment configuration file.

channelModel.py: Functions for channel generation.

lowComplexityUserMatching.py: Implementation of the proposed matching algorithm.

SBPA.py: Stream-Based Power Allocation implementation.

Results/: Directory where CSV logs are stored.

⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
