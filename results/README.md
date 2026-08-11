# Results — CAMPUS-SenseRL / KL-CMAPPO

**Proposed method:** KL-CMAPPO  
**BC:** training-stage **initialization** (not a competing final method)  
**Semantic expert:** demonstration teacher / strong reference

## Layout

```text
results/
  figures/main/            # manuscript figures
  figures/supplementary/   # ablations companions, RL-vs-init, etc.
  csv_json/                # frozen tables used by the figures
  Main_Results.md
  Figure_Index.md
```

## Main figures (current)

1. Campus deployment  
2. Real KL-CMAPPO timeline  
3. Method workflow (KL-CMAPPO dominant; BC = initialization)  
4. Key MAE: **Fixed 60 / Delta+heartbeat / KL-CMAPPO**  
5. Training validation MAE  
6. Ablation of RL stages (includes BC initialization)  
7. Held-out transfer (expert reference + KL-CMAPPO)  
8. Packet-loss robustness (KL-CMAPPO)  
9. Cumulative transmissions  

Matched-budget **BC initialization vs KL-CMAPPO** is in **supplementary** (`figS01_*`).

## Regenerate

```bash
python scripts/26_paper_figures_simple.py
```

No training is performed.
