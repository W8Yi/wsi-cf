# TCGA Balanced Core (90/10) Technical Report

Generated: 2026-03-09  
Location: `/common/users/wq50/SAE_path/metadata/manifests/tcga_balanced_core_90_10`

## 1) Scope and Inputs

This report summarizes the current `Balanced Core` subset built from:

- `metadata/labels/master/case_labels_master.tsv`
- `metadata/labels/master/slide_labels_master.tsv`
- Selection script: `metadata/labels/scripts/build_tcga_balanced_core.py`

Output files:

- `manifest.json`
- `summary.json`
- `balanced_core_cases.tsv`
- `balanced_core_slides.tsv`

## 2) Selection Design

Balanced Core is built with these rules:

1. Project-balanced case cap: up to 40 cases per TCGA project.
2. Diversity-aware case ranking inside each project:
   - Cases get label tokens from available annotations (HPV, immune subtype, MSI, PAM50, stage, grade, OS/PFS, TP53/KRAS mutation, tumor purity bin).
   - Greedy selection prioritizes cases adding unseen and rare tokens.
3. Deterministic split:
   - Case-level split (no train/test case leakage).
   - Target test fraction = 10% inside each project.
4. Representative slide policy:
   - 1 slide per case (`slides_per_case=1`), prioritizing primary tumor and DX1.

## 3) Dataset Composition

### 3.1 Global counts

- Projects: 33
- Cases: 1319 total
- Train cases: 1187
- Test cases: 132
- Slides: 1319 total (1 per case)
- H5 paths in manifest: 1187 train + 132 test

### 3.2 Project balance

- All projects contribute almost equally.
- 32 projects contribute 40 cases each.
- `TCGA-CHOL` contributes 39 cases (all available in source table).
- Project share is therefore approximately:
  - 3.03% per project (40/1319)
  - 2.96% for `TCGA-CHOL` (39/1319)

### 3.3 Token coverage

From `summary.json`:

- Selected unique tokens: 65
- Total unique tokens in source pool: 65
- Coverage ratio: 1.00

This means the selected subset preserved all discovered token categories used by the selector.

### 3.4 Storage estimate (WSI)

Using `metadata/indexes/tcga_wsi_index.json` slide sizes:

- Train WSIs: ~1.391 TB (1.265 TiB), 1186 resolved + 1 unresolved
- Test WSIs: ~0.165 TB (0.150 TiB), 132 resolved
- Total Balanced Core WSIs: ~1.556 TB (1.415 TiB), 1318 resolved + 1 unresolved

## 4) Label Composition and Percentages (Case-level)

Percentages below are over total cases in each split.

### 4.1 All cases (n=1319)

- `hpv_status`: Unknown 95.45%, HPV+ 2.43%, HPV- 2.12%
- `immune_subtype`: C4 20.85%, C1 19.33%, C2 18.73%, C3 16.76%, C6 9.55%, C5 4.78%, Unknown 10.01%
- `msi_status`: Unknown 93.93%, MSI 4.02%, NonMSI 2.05%
- `pam50_subtype`: Unknown 94.01%, Normal 2.50%, LumA 1.74%, LumB 0.83%, Her2 0.53%, Basal 0.38%
- `stage`: Unknown 33.43%, Stage II 8.87%, Stage IV 7.58%, Stage III 5.76%, Stage I 5.38%, Stage IIB 5.08%, Stage IIIA 4.17%, Stage IIIC 3.94%
- `tumor_grade`: Unknown 63.99%, G3 9.33%, G2 8.04%, G1 6.14%, GX 5.46%, G4 3.87%, LOW GRADE 1.59%, HIGH GRADE 1.44%
- `os_status`: Alive 60.20%, Dead 39.80%
- `pfs_status`: CENSORED 54.97%, PROGRESSION 43.90%, Unknown 1.14%
- `tp53_mutated`: 0 = 66.79%, 1 = 31.24%, Unknown 1.97%
- `kras_mutated`: 0 = 85.82%, 1 = 12.21%, Unknown 1.97%

Known-label availability rates (all):

- `hpv_status`: 4.55%
- `immune_subtype`: 89.99%
- `msi_status`: 6.07%
- `pam50_subtype`: 5.99%
- `stage`: 66.57%
- `tumor_grade`: 36.01%
- `os_status`: 100.00%
- `pfs_status`: 98.86%
- `tp53_mutated`: 98.03% (after mutation_has_data gating)
- `kras_mutated`: 98.03% (after mutation_has_data gating)
- `tumor_purity`: 97.80%

### 4.2 Train cases (n=1187)

- `hpv_status`: Unknown 95.45%, HPV+ 2.44%, HPV- 2.11%
- `immune_subtype`: C4 21.40%, C1 19.04%, C2 18.62%, C3 16.85%, C6 9.27%, C5 4.72%, Unknown 10.11%
- `msi_status`: Unknown 93.93%, MSI 3.96%, NonMSI 2.11%
- `pam50_subtype`: Unknown 94.02%, Normal 2.27%, LumA 1.85%, LumB 0.84%, Her2 0.59%, Basal 0.42%
- `stage`: Unknown 33.45%, Stage II 9.10%, Stage IV 7.83%, Stage I 5.56%, Stage IIB 5.05%, Stage III 4.97%, Stage IIIA 4.47%, Stage IIIC 3.96%
- `tumor_grade`: Unknown 63.94%, G3 9.60%, G2 7.50%, G1 6.32%, GX 5.48%, G4 3.96%
- `os_status`: Alive 60.66%, Dead 39.34%
- `pfs_status`: CENSORED 56.02%, PROGRESSION 42.80%, Unknown 1.18%
- `tp53_mutated`: 0 = 66.64%, 1 = 31.26%, Unknown 2.11%
- `kras_mutated`: 0 = 85.34%, 1 = 12.55%, Unknown 2.11%

### 4.3 Test cases (n=132)

- `hpv_status`: Unknown 95.45%, HPV+ 2.27%, HPV- 2.27%
- `immune_subtype`: C1 21.97%, C2 19.70%, C4 15.91%, C3 15.91%, C6 12.12%, C5 5.30%, Unknown 9.09%
- `msi_status`: Unknown 93.94%, MSI 4.55%, NonMSI 1.52%
- `pam50_subtype`: Unknown 93.94%, Normal 4.55%, LumB 0.76%, LumA 0.76%
- `stage`: Unknown 33.33%, Stage III 12.88%, Stage II 6.82%, Stage IIIB 6.06%, Stage IIB 5.30%, Stage IV 5.30%
- `tumor_grade`: Unknown 64.39%, G2 12.88%, G3 6.82%, GX 5.30%, G1 4.55%, G4 3.03%, HIGH GRADE 3.03%
- `os_status`: Alive 56.06%, Dead 43.94%
- `pfs_status`: PROGRESSION 53.79%, CENSORED 45.45%, Unknown 0.76%
- `tp53_mutated`: 0 = 68.18%, 1 = 31.06%, Unknown 0.76%
- `kras_mutated`: 0 = 90.15%, 1 = 9.09%, Unknown 0.76%

## 5) HNSC HPV Detail

In selected HNSC cases:

- Total HNSC cases: 40
- All split: HPV+ 10, HPV- 27, Unknown 3
- Train HNSC (36): HPV+ 9, HPV- 24, Unknown 3
- Test HNSC (4): HPV+ 1, HPV- 3, Unknown 0

Balanced Core includes both HPV classes but does not enforce strict 1:1 HPV balancing.

## 6) Notes and Constraints

1. This subset is optimized for broad pan-cancer coverage, not for one-task class balancing.
2. Labels with high unknown rates (`hpv_status`, `msi_status`, `pam50_subtype`) are limited by source availability, not by split logic.
3. For task-specific training (for example HPV prediction), create a second task-focused manifest while keeping this core unchanged for general benchmarking.
