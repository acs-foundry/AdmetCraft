# AdmetCraft v1.2.0

**Local-First, Pure-Rust Quantitative ADMET, QSAR + Tox21 (32 TDC Endpoints + 90% Conformal CI + Applicability Domain), SEA Off-Target Fishing, Phase I/II Metabolite Guard, Mechanistic IVIVE → Minimal-PBPK Human Therapeutic Dose Simulator & TPP MPO Workbench** (`#![forbid(unsafe_code)]`).

**AdmetCraft v1.2** combines exact physicochemical, ionization ($\text{p}K_a$/$\log D_{7.4}$), solubility, druglikeness (**QED**, **Pfizer CNS MPO**, **Lipinski/Ghose/Veber/Egan/Muegge**, **Pfizer 3/75**, **GSK 4/400**, **Golden Triangle**, **Ro3**, **bRo5**), and structural alerts (**PAINS** / **Brenk**) with:
- **32 pure-Rust trained TDC QSAR GBDT models** (20 core ADMET + **12 Tox21 Nuclear Receptor & Cellular Stress pathways**, mean held-out Bemis-Murcko scaffold **ROC AUC = 0.833**, mean 90% conformal coverage **92.7%** — read with the singleton rate below, not on its own),
- **Local SEA (*Similarity Ensemble Approach*) Off-Target Fishing** against the curated **Bowes Safety-20** human anti-target panel (`< 2 ms`/compound, hardware POPCNT ECFP4 Tanimoto + Gumbel E-values, extensible via `build-sea-index`),
- **Phase I/II Metabolite Guard** (11 deterministic graph biotransformations: benzylic/allylic/aromatic hydroxylation, O-/N-dealkylation, furan/thiophene/alkene epoxidation, aniline N-hydroxylation, thioether S-oxidation, ester hydrolysis, phenolic O-glucuronidation/sulfation, acyl-glucuronidation, and NAPQI-type quinone-imine oxidation — automatically re-screened for reactive bioactivation & DILI/AMES/hERG toxicity escalation),
- **Mechanistic IVIVE → Minimal-PBPK Human Therapeutic Dose Simulator (`pk-dose`)**: scales calibrated QSAR `f_u,p`, `CLint`, `VDss`, `Caco-2`, `HIA`, and FaSSIF pH 6.5 solubility (Henderson-Hasselbalch ionization + bile-salt micelles + Butler-Dressman DCS dynamic sink) through Austin (2002) `fu_inc`, the well-stirred liver model, and a mass-conserving 4th-order Runge-Kutta (`RK4`) multi-compartment ODE engine (`Oral | IV | SC` + optional saturable TMDD clearance hook for peptides/biologics), propagating 90% Split Conformal QSAR intervals via 200-draw deterministic Monte Carlo into QD/BID human dose brackets, hERG margins, BCS Class I–IV, and standalone SVG `C_u(t)` profiles,
- **Atomic Sites of Metabolism (SoM)** and configurable **Target Product Profile (`TPP MPO`)** scoring—completely offline across all CPU cores.

---

## Architecture (7-Crate Portable Workspace)

- `crates/chem-core` (L0): SMILES & V2000 MOL/SDF parsers, RFC-4180 CSV importer, molecular graph, GF(2) SSSR ring perception, Hückel $4n+2$ aromaticity perception, stack-safe canonical SMILES, automatic desalting, stereochemistry warnings, 2D coordinate layout, and recursive SMARTS matcher.
- `crates/admet-calc` (L1): Parallel (`rayon`) descriptor, ionization ($\text{p}K_a$/$\log D_{7.4}$), lipophilicity, solubility, BOILED-Egg, QED/CNS MPO, industrial rules, PAINS/Brenk/hERG/AMES/DILI, matched-pair comparison (`compare_smiles`), and shared QSAR/SoM/TPP/PBPK data types.
- `crates/admet-qsar` (L1.5): **100% pure-Rust Rayon-parallel histogram GBDT trainer & inference engine** (`#![forbid(unsafe_code)]`), circular **ECFP4 / Morgan radius-2** (`1024`-bit POPCNT Tanimoto + counts) + 56 dense descriptors, deterministic **Bemis-Murcko scaffold splitter (70/10/20)**, Platt calibration, **Split Conformal Prediction (90% CI & Mondrian sets)**, **Applicability Domain** (`InDomain / Borderline / OutOfDomain`), **SEA Off-Target Fishing**, **Phase I/II Metabolite Guard**, **Mechanistic IVIVE → Minimal-PBPK (`pbpk.rs`)**, **atomic Sites of Metabolism (SoM)** with MedChem blocking fixes, **TPP MPO** (`Oral`, `CNS`, `Peripheral`, `Oncology`), and `CRAFT_DATA_DIR` model-pack loader (`$CRAFT_DATA_DIR/models/admetcraft_v1/admetcraft_tdc_v1.json`).
- `crates/admet-report` (L2): CSV (95+ columns incl. QSAR/AD/TPP/SoM/Tox21/SEA/Metabolites), JSON, SVG (Radar, BOILED-Egg, and 48 h steady-state PK `C_u(t)` profile with 90% CI ribbon), and pure-Rust multi-page PDF 1.4 vector report writer with **zlib `/Filter /FlateDecode`** compression.
- `crates/admet-engine` (L3): Session state (with live TPP preset switching), undo/redo history, unified `CommandRegistry`, multi-criteria filtering, and `PathSandbox` (`openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`).
- `apps/admetcraft-cli` (L4): Headless CLI (`eval`, `profile`, `compare`, `batch`, `metabolites`, `offtargets`, `build-sea-index`, `pk-dose`, `train-qsar`, `model-info`) and strict **JSON-RPC 2.0 / MCP (2025-06-18)** server (`stdio` and authenticated loopback-default `--tcp`) with `outputSchema` on all 9 tools (incl. `predict_metabolites`, `predict_offtargets`, and `predict_human_pk_dose`).
- `apps/admetcraft` (L5): Native `egui`/`eframe` desktop GUI with 2D molecular sketcher, **QSAR (TDC + 90% CI + SoM + TPP)** tab, **Human PK / Minimal-PBPK Dose Simulator** panel with live `C_u(t)` sparkline, live **2D Sites of Metabolism hotspot overlay**, **TPP Preset selector**, KPI badge strip (`QED`, `CNS MPO`, `LogD7.4`, `hERG`, `TPP MPO`, `AD`), interactive *BOILED-Egg* plot, Batch Summary Table, Light/Dark themes, and `Ctrl+K` Command Palette.

---

## Quantitative QSAR & Tox21 Validation (32 Official TDC Endpoints)

All **32 models** (20 core ADMET + **12 official Tox21 cellular toxicity & nuclear receptor pathways**) are trained locally in pure Rust (`admetcraft-cli train-qsar`, ~22 s on 24 threads) using a strict **zero-leakage Bemis-Murcko Scaffold Split (70% train / 10% conformal calibration / 20% held-out test)**.

- **Pre-trained model pack download:** Download `admetcraft-tdc-models-v1.1.0.json.gz` from the GitHub Release assets, decompress (`gunzip`), and place at `$CRAFT_DATA_DIR/models/admetcraft_v1/admetcraft_tdc_v1.json` (with `CRAFT_DATA_DIR` unset, `$XDG_DATA_HOME/CraftBio/models/admetcraft_v1/admetcraft_tdc_v1.json`).
- **Re-training from raw TDC datasets:** Raw `.tab` files originate from official Harvard Dataverse / [Therapeutics Data Commons (`PyTDC`)](https://tdcommons.ai) (including `tox21.tab`, Dataverse ID `4259612`), placed in `$CRAFT_DATA_DIR/models/tdc_raw/`.
- **v1.1.0 Held-Out Scaffold Summary (32 endpoints):** Mean Classification Scaffold **ROC AUC = 0.833** (`25` endpoints: `0.856` on 13 core ADMET, `0.811` across 12 Tox21 incl. `SR-MMP 0.906` & `NR-AhR 0.902`) | Mean Regression Scaffold **Spearman ρ = 0.668** (`7` endpoints) | Mean **90% Split-Conformal Coverage = 92.7%**.

> **How to read the coverage figure.** The conformal quantile has a floor of `0.15`
> (`gbdt.rs`, `clamp(0.15, 0.99)`). Where the data would put it lower, the floor widens the
> prediction set instead: coverage goes **above** the nominal 90%, and more predictions come back
> as the uninformative `{0,1}` set that contains both labels. In the shipped v1 pack **9 of the 25
> binary endpoints sit exactly on that floor**, and the mean singleton (informative) rate is
> **0.538** — about half the predictions do not commit to a label. Three endpoints are extreme:
> `tox21_nr_ar_lbd` 99.2% coverage at a 15.9% informative rate, `tox21_nr_ar` 99.4% / 22.9%, and
> `dili` 97.9% / 17.9%. Each model carries its own `conformal_singleton_rate`; use the pair.
> High coverage with a low singleton rate means the model is abstaining, not succeeding.

### Core 20 ADMET Endpoints Reference Table

| Endpoint | Category | Task | Train / Cal / Test | Held-Out Scaffold Metric | Secondary Metric | 90% Conformal Cov. |
|---|---|---|---:|---:|---:|---:|
| `herg_karim` (hERG Cardiotoxicity) | Toxicity | Binary | 9,412 / 1,345 / 2,688 | ROC AUC **0.803** | PR 0.747 / Brier 0.177 | **89.5%** |
| `ames` (AMES Mutagenicity) | Toxicity | Binary | 5,095 / 728 / 1,455 | ROC AUC **0.825** | PR 0.791 / Brier 0.170 | **91.4%** |
| `dili` (Drug-Induced Liver Injury) | Toxicity | Binary | 331 / 47 / 95 | ROC AUC **0.839** | PR 0.819 / Brier 0.181 | **97.9%** |
| `clintox` (Clinical Trial Toxicity) | Toxicity | Binary | 1,031 / 147 / 295 | ROC AUC **0.882** | PR 0.357 / Brier 0.035 | **91.2%** |
| `cyp1a2_veith` (CYP1A2 Inhibitor) | Metabolism | Binary | 8,804 / 1,258 / 2,515 | ROC AUC **0.886** | PR 0.858 / Brier 0.136 | **89.7%** |
| `cyp2c9_veith` (CYP2C9 Inhibitor) | Metabolism | Binary | 8,463 / 1,209 / 2,418 | ROC AUC **0.864** | PR 0.778 / Brier 0.146 | **90.5%** |
| `cyp2c19_veith` (CYP2C19 Inhibitor) | Metabolism | Binary | 8,864 / 1,266 / 2,533 | ROC AUC **0.855** | PR 0.834 / Brier 0.153 | **90.0%** |
| `cyp2d6_veith` (CYP2D6 Inhibitor) | Metabolism | Binary | 9,190 / 1,313 / 2,625 | ROC AUC **0.847** | PR 0.666 / Brier 0.106 | **89.3%** |
| `cyp3a4_veith` (CYP3A4 Inhibitor) | Metabolism | Binary | 8,628 / 1,233 / 2,465 | ROC AUC **0.856** | PR 0.809 / Brier 0.154 | **88.8%** |
| `pgp_broccatelli` (P-gp Efflux/Inh.) | Absorption | Binary | 853 / 122 / 244 | ROC AUC **0.926** | PR 0.910 / Brier 0.107 | **92.6%** |
| `hia_hou` (Intestinal Absorption) | Absorption | Binary | 405 / 58 / 115 | ROC AUC **0.991** | PR 0.998 / Brier 0.030 | **94.8%** |
| `bioavailability_ma` (Oral F20%) | Absorption | Binary | 448 / 64 / 128 | ROC AUC **0.643** | PR 0.883 / Brier 0.160 | **92.2%** |
| `bbb_martins` (BBB Penetration) | Distribution | Binary | 1,427 / 204 / 408 | ROC AUC **0.909** | PR 0.964 / Brier 0.087 | **94.1%** |
| `caco2_wang` (Caco-2 Permeability) | Absorption | Regression | 637 / 91 / 182 | Spearman **0.753** | RMSE 0.476 / MAE 0.375 | **90.1%** |
| `ppbr_az` (Human PPB %) | Distribution | Regression | 1,130 / 161 / 323 | Spearman **0.657** | RMSE 14.62 / MAE 9.15 | **92.6%** |
| `vdss_lombardo` (VDss L/kg) | Distribution | Reg ($\log_{10}$) | 791 / 113 / 226 | Spearman **0.715** | RMSE 0.509 / MAE 0.375 | **92.5%** |
| `clearance_microsome_az` (CLint) | Excretion | Reg ($\log_{10}$) | 771 / 110 / 221 | Spearman **0.579** | RMSE 0.485 / MAE 0.406 | **87.8%** |
| `half_life_obach` (Human $t_{1/2}$) | Excretion | Reg ($\log_{10}$) | 467 / 67 / 133 | Spearman **0.312** | RMSE 0.618 / MAE 0.473 | **86.5%** |
| `solubility_aqsoldb` (AqSolDB) | Physicochem | Regression | 6,875 / 982 / 1,964 | Spearman **0.866** | RMSE 1.259 / MAE 0.863 | **90.3%** |
| `lipophilicity_astrazeneca` (LogD) | Physicochem | Regression | 2,940 / 420 / 840 | Spearman **0.797** | RMSE 0.719 / MAE 0.554 | **91.1%** |

**Summary:** Mean Classification Scaffold ROC AUC = **0.856** (`13` endpoints) | Mean Regression Scaffold Spearman $\rho$ = **0.668** (`7` endpoints) | Mean 90% Split-Conformal Coverage = **91.1%**.

### Physicochemical & Rule Parity (clean-room study, see the caveat)

> **What is in this repository:** `tests/fixtures/reference_molecules.json` holds **15 molecules** with
> published reference *ranges*, not the 45-molecule comparison the figures below were taken from, and there
> is no script here that reproduces them against RDKit or Indigo. Until that comparison and its output are
> committed, read the numbers below as a record of a study done elsewhere, not as something you can verify
> from this checkout.

- Reported from that study: **TPSA (S&P and N/O only)** `MAE = 0.0000`; **Exact Mass** max err `0.00005 Da`; **MW** MAE `0.0038 Da`; **MR** $r = 0.9997$; **WLOGP** $r = 0.9844$; **QED** $r = 0.9723$; **PAINS** and **Brenk** agreement 100%. Sample size 45 molecules; the fixture committed here covers 15 of them.
- **Honest Scope:** `half_life_obach` ($\rho = 0.312$) and `bioavailability_ma` (AUC `0.643`) reflect inherent in vivo pharmacokinetic variability; predictions outside the training applicability domain (`Out-of-Domain`, max ECFP4 Tanimoto `< 0.25`) automatically inflate their 90% conformal uncertainty bands and flag `Uncertain (Both)` rather than fabricating high confidence.

---

## Building, Testing & Usage

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release --workspace
cargo build --workspace --release
./packaging/install.sh
```

```bash
# Train local 20-endpoint TDC QSAR pack into CRAFT_DATA_DIR/models/admetcraft_v1/
admetcraft-cli train-qsar --max-trees 360 --max-depth 5

# Inspect held-out scaffold validation metrics of the installed model pack
admetcraft-cli model-info

# Complete profile card (Physicochem + 20 TDC QSAR + 90% CI + AD + SoM + TPP MPO)
admetcraft-cli profile "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC" --name Verapamil --tpp oral

# Parallel batch triage (.smi, .csv, .sdf -> .csv, .json, .pdf, .svg)
admetcraft-cli batch candidates.smi --tpp oncology --out admet_report.csv

# Human PK / Minimal-PBPK therapeutic dose simulation (QD/BID + 90% CI + SVG curve)
admetcraft-cli pk-dose "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1" --name Gefitinib --potency-nm 30 --tpp oncology --out-svg gefitinib_pk.svg

# Sandboxed MCP server over stdio (9 tools with outputSchema)
admetcraft-cli mcp --read-root ./inputs --write-root ./outputs
```

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).
