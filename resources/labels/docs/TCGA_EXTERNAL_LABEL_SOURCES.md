# External TCGA Label Sources (for WSI Benchmarking)

This file lists external sources (papers/tools/datasets) that provide labels usable with TCGA slides.

## 1) High-value, immediately usable for your benchmark

### A. cBioPortal PanCancer clinical table (already integrated locally)
- Source repo: https://github.com/GerkeLab/TCGAclinical
- File used: `data/cBioportal_data.tsv`
- What labels it contains:
  - Pan-cancer acronyms
  - `Subtype` (many disease-specific molecular subtypes, e.g. BRCA/LGG/HNSC)
  - OS/PFS status and times
  - AJCC stage fields (partial)
- Why important:
  - Very high overlap with your slide cohort (you already mapped ~95%+)

### B. GDC API clinical/biospecimen fields (already integrated locally)
- API: https://api.gdc.cancer.gov/files
- What labels it contains:
  - `project_id`, `primary_site`, `disease_type`, `sample_type`
  - `primary_diagnosis`, `tumor_grade` (partial), AJCC fields (partial)
  - vital status and follow-up/death-related fields
- Why important:
  - Official source, broad coverage

### C. TCGAbiolinks molecular subtype tables
- `PanCancerAtlas_subtypes` docs: https://rdrr.io/bioc/TCGAbiolinks/man/PanCancerAtlas_subtypes.html
- `TCGAquery_subtype` docs: https://rdrr.io/bioc/TCGAbiolinks/man/TCGAquery_subtype.html
- Subtypes vignette: https://bioconductor.org/packages/release/bioc/vignettes/TCGAbiolinks/inst/doc/extension.html
- Why important:
  - Standard way many papers retrieve TCGA subtype labels

### D. Pan-Cancer Clinical Data Resource (survival endpoints)
- Paper: https://pubmed.ncbi.nlm.nih.gov/29625055/
- Companion table mirrored by GerkeLab repo above
- Why important:
  - Curated OS/DSS/DFI/PFI endpoint definitions used in many studies

## 2) TCGA marker papers that define disease-specific labels

These are commonly used to derive subtype labels in tools like TCGAbiolinks:

- HNSC (HPV-related biology / groups):
  - https://www.nature.com/articles/nature14129
- CESC integrated characterization:
  - https://www.nature.com/articles/nature21386
- Pan-cancer integrated clustering (iClusters/subtypes):
  - https://pubmed.ncbi.nlm.nih.gov/29625048/

## 3) Pathology-focused labeled datasets derived from TCGA

### A. BRCA-M2C (cell-level dot annotations on TCGA BRCA patches)
- Repo: https://github.com/TopoXLab/Dataset-BRCA-M2C
- Labels:
  - cell class IDs: lymphocyte / tumor-epithelial / stromal
- Use case:
  - weak supervision / cell-concept benchmark extension

### B. RCdpia (renal carcinoma region/tumor annotations from TCGA)
- Paper: https://arxiv.org/abs/2403.11211
- Dataset page in paper: `http://39.171.241.18:8888/RCdpia/`
- Use case:
  - region-level supervision for kidney projects

### C. PathText / WsiCaption (report-derived TCGA slide text labels)
- Paper: https://arxiv.org/abs/2311.16480
- Repo: https://github.com/cpystan/Wsi-Caption
- Labels:
  - slide-level report text/captions extracted from TCGA pathology reports
- Use case:
  - report-driven concept mining (grade/receptor mentions, etc.)

### D. CTIS-QA (TCGA-BRCA report template extraction)
- Paper: https://arxiv.org/abs/2601.01769
- Repo: https://github.com/HLSvois/CTIS-QA
- Labels:
  - structured report-derived diagnostic attributes in QA format
- Use case:
  - clinically grounded closed-ended labels for WSI QA/classification

## 4) How to use these for your benchmark

Recommended order:
1. Start with robust slide-level labels (already in `master/slide_labels_master.tsv`):
   - `cbio_subtype`, `gdc_primary_site`, `gdc_disease_type`, `cbio_os_status`
2. Add focused disease-specific tasks:
   - HNSC HPV+/-, CESC subtype, BRCA subtype, LGG subtype
3. Add sparse but high-value labels:
   - external HPV consensus labels in `sources/curated/hpv/nature2017_cesc_hpv_consensus.tsv`
4. Later, add patch/region labels from BRCA-M2C/RCdpia and report-derived labels from PathText/CTIS-QA.

## 5) Caveats

- Some labels are sparse or class-imbalanced (especially HPV subtype bins).
- Some sources need careful sample ID harmonization (TCGA barcode parsing).
- For fair evaluation, always split by patient and check site/project confounding.
