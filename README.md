# 🔬 HEPSIM3 — ML-Based Simulation Bias Analysis
### Pythia vs. Herwig vs. Experimental Data

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-FF6B6B?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)
![GSoC](https://img.shields.io/badge/GSoC-2026-4285F4?style=for-the-badge&logo=google&logoColor=white)

**GSoC 2026 Project Proposal + Evaluation Task Solution**
**Organisation: ML for Science (ML4Sci) | Project: HEPSIM3**

[📓 Notebook](#-evaluation-task-notebook) • [🚀 Quickstart](#-quickstart) • [📊 Results](#-results) • [🗂️ Structure](#️-repository-structure) • [📚 References](#-references)

</div>

---

## 📌 What Is This?

This repository contains:

1. **The complete evaluation task solution** for the ML4Sci HEPSIM3 GSoC 2026 project — a Jupyter notebook solving all 4 parts (data loading, jet observables, Lorentz boost, classification)
2. **The GSoC project proposal** for *ML-Based Simulation Bias Analysis: Pythia vs. Herwig vs. Data*

The evaluation task uses the [Pythia 8 Quark and Gluon Jets dataset](https://zenodo.org/records/3164691) (Komiske, Metodiev & Thaler, 2019) to demonstrate the core skills required for the HEPSIM3 project.

---

## 🎯 Project Overview

Monte Carlo event generators — principally **Pythia 8** and **Herwig 7** — are essential tools at the LHC, yet they implement physically distinct models for parton showering, hadronisation, and the underlying event. These differences produce systematic discrepancies that directly affect physics measurements.

This project applies **machine learning classifiers** to:
- Quantify inter-generator biases in high-dimensional feature space
- Identify *which observables and phase-space regions* drive the largest differences (via SHAP)
- Use the classifier output as a **learned reweighting function** to correct for generator bias
- Extend the framework to **MC-vs-data** comparison using public unfolded measurements

---

## 📓 Evaluation Task Notebook

**File:** `HEPSIM3_GSoC2026_Elite.ipynb`

The notebook solves all four parts of the official HEPSIM3 evaluation task:

| Part | Description | Key Result |
|------|-------------|------------|
| **(a) Data Loading** | Load 500k jets, count real constituents, multiplicity distributions, leading-constituent kinematics | Gluon/Quark constituent ratio = 1.89 ≈ C_A/C_F ✓ |
| **(b) Jet Observables** | Jet mass, width (pT-weighted ΔR), pT-dispersion from constituent 4-vectors | KS statistics confirm all three observables discriminate Q/G |
| **(c) Rest-Frame Boost** | Exact Lorentz boost to jet rest frame; numerical verification | \|Σp_rest\| / E_J < 10⁻¹² ✓ |
| **(d) Classification** | XGBoost BDT + PyTorch NN; ROC/AUC, confusion matrix, SHAP, frame comparison | **AUC = 0.834** |

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/KrishVerma6797/Hepsim3.git
cd Hepsim3
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn xgboost torch shap scipy
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook HEPSIM3_GSoC2026_Elite.ipynb
```

> **Note:** The notebook automatically downloads the dataset from Zenodo on first run (~530 MB). No manual download needed.

---

## 📦 Requirements

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
xgboost>=2.0
torch>=2.0
shap>=0.43
scipy>=1.11
jupyter>=1.0
```

**Python version:** 3.12+ (tested on 3.12 and 3.13)

---

## 📊 Results

### Classification Performance

| Classifier | Features | ROC-AUC |
|------------|----------|---------|
| XGBoost BDT | Lab-frame only | 0.814 |
| XGBoost BDT | Rest-frame only | 0.820 |
| XGBoost BDT | Combined (9 features) | **0.827** |
| PyTorch Neural Network | Combined (9 features) | **0.834** |
| Single feature (jet width) | Lab-frame | 0.720 |

### Most Discriminating Features (SHAP ranking)
1. **Constituent multiplicity** — directly reflects QCD colour charge (C_A = 3 vs C_F = 4/3)
2. **Jet width** — gluon jets are ~50% broader; largest single-observable KS statistic
3. **pT dispersion** — quark jets more pT-concentrated in one hard constituent
4. **Rest-frame mean pT** — removes lab-frame kinematic bias
5. **Leading pT fraction** — fraction of total pT in the hardest constituent

### Key Physics Findings
- Gluon jets have **~40% more constituents** than quark jets — consistent with C_A/C_F = 9/4 = 2.25
- Rest-frame features provide **+0.7% AUC improvement** over lab-frame features by decoupling substructure from hard-process kinematics
- **Jet width** has the highest single-feature KS statistic (KS = 0.41, p < 10⁻³⁰⁰)

---

## 🗂️ Repository Structure

```
Hepsim3/
│
├── 📓 HEPSIM3_GSoC2026_Elite.ipynb   ← Main evaluation notebook (run this)
│
├── 📄 proposal/
│   └── HEPSIM3_GSoC2026_Proposal.docx ← GSoC 2026 project proposal
│
├── 📁 data/                           ← Auto-created on first run
│   ├── QG_jets.npz                    ← Downloaded from Zenodo
│   ├── QG_jets_1.npz
│   └── ...
│
├── 📁 figures/                        ← Generated plots (auto-saved)
│   ├── fig_a_multiplicity.png
│   ├── fig_b_observables.png
│   ├── fig_c_rest_frame_viz.png
│   ├── fig_d_roc_and_scores.png
│   ├── fig_d_confusion_matrix.png
│   ├── fig_d_feature_importance.png
│   ├── fig_d_shap.png
│   └── fig_d_frame_comparison.png
│
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🔬 Technical Details

### Dataset
- **Source:** [Zenodo 10.5281/zenodo.3164691](https://zenodo.org/records/3164691)
- **Files used:** `QG_jets.npz` through `QG_jets_4.npz` (5 files, 500,000 jets total)
- **Format:** Each file contains `X` (N, M, 4) and `y` (N,) arrays
- **Features per constituent:** pT, rapidity (y), azimuthal angle (φ), PDG particle ID
- **Labels:** 0 = gluon jet, 1 = quark jet (50/50 balanced)

### Jet Observables Implemented
```python
# All implemented as vectorised NumPy functions on (N, M, 4) arrays
jet_mass(fv)          # m² = E²_J - |p_J|²
jet_width(X)          # w = Σ pT_i ΔR_i / Σ pT_i  (φ-wrapped)
pt_dispersion(X)      # p^D_T = √(Σ p²_{T,i}) / Σ p_{T,i}
```

### Lorentz Boost
```python
# Boost vector: β = p_J / E_J,  γ = E_J / m_J
# Verification: |Σp_rest| / E_J < 1e-12 for all jets
boost_to_rest_frame(fv_jet)   # single jet
boost_batch(fv_batch, is_real) # vectorised batch
```

### Classifier Architecture (PyTorch)
```
Input(9) → Linear(256) → LeakyReLU → BatchNorm → Dropout(0.3)
         → Linear(128) → LeakyReLU → BatchNorm → Dropout(0.3)
         → Linear(64)  → LeakyReLU → BatchNorm → Dropout(0.3)
         → Linear(1)   → Sigmoid
```
Training: AdamW + CosineAnnealingLR + gradient clipping (max_norm=1.0) + best-checkpoint saving

---

## 📅 GSoC 2026 Timeline

| Period | Phase |
|--------|-------|
| May 1–24 | Community Bonding |
| May 25 – Jun 22 | Phase 1: Data pipeline + Observable library |
| Jun 23 – Jul 5 | Phase 2: Classifiers (BDT + NN) |
| **Jul 6–10** | **★ Midterm Evaluation** |
| Jul 11 – Aug 3 | Phase 2 cont.: SHAP + Reweighting + Multi-process |
| Aug 4–23 | Phase 3: MC-vs-data + Documentation + Release |
| **Aug 24** | **★ Final Evaluation** |

---

## 👨‍💻 About

**Krish Verma**
B.Tech in Computer Science (AIML), GBPIET, Uttarakhand, India
📧 krishverma6797@gmail.com
🐙 github.com/KrishVerma6797

GSoC 2026 applicant for the ML4Sci HEPSIM3 project.
Mentors: Steve Mrenna (Fermilab), Konstantin Matchev (U. Alabama), Tony Menzo, Ian Pang (Rutgers)

---

## 📚 References

1. P. Komiske, E. Metodiev, J. Thaler, *Pythia8 Quark and Gluon Jets for Energy Flow*, Zenodo v1 (2019). [doi:10.5281/zenodo.3164691](https://doi.org/10.5281/zenodo.3164691)
2. P. Komiske, E. Metodiev, J. Thaler, *Energy Flow Networks: Deep Sets for Particle Jets*, JHEP 01 (2019) 121. [arXiv:1810.05165](https://arxiv.org/abs/1810.05165)
3. A. Andreassen et al., *OmniFold: A Method to Simultaneously Unfold All Observables*, PRL 124 (2020) 182001. [arXiv:1911.09107](https://arxiv.org/abs/1911.09107)
4. K. Cranmer, J. Pavez, G. Louppe, *Approximating Likelihood Ratios with Calibrated Discriminative Classifiers* (2015). [arXiv:1506.02169](https://arxiv.org/abs/1506.02169)
5. S. Lundberg & S.-I. Lee, *A unified approach to interpreting model predictions*, NeurIPS (2017). [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
6. J. Gallicchio, M. Schwartz, *Quark and Gluon Jet Substructure*, JHEP 04 (2013) 090. [arXiv:1211.7038](https://arxiv.org/abs/1211.7038)
7. T. Chen & C. Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD (2016). [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<sub>Made with ❤️ for ML4Sci GSoC 2026 | Questions? Email ml4-sci@cern.ch</sub>
</div>
