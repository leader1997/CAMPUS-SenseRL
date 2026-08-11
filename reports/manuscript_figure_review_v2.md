# Manuscript figure review v2

Generated without training from frozen artifacts. Main pack: `paper_outputs/manuscript_figures_v2/main/`.
Prior simplified pack archived at `paper_outputs/legacy_complex_figures/results_figures_prior/`.

## fig01
1. Single question? **Is this a real campus deployment with held-out sensors?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **no (spatial only)**
4. BC as second proposed method? **no — RL development not labeled BC**
5. More than one metric? **no**
6. Split obvious? **metadata**
7. Error bars legitimate? **N/A**
8. KL-CMAPPO conclusion visible? **yes — dark vs diamond cohorts**
9. Could be simpler? **no**

## fig02
1. Single question? **What does adaptive TX look like?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **Fixed15 only as count annotation**
4. BC as second proposed method? **no**
5. More than one metric? **no (one axis CO2)**
6. Split obvious? **validation illustrative**
7. Error bars legitimate? **illustrative only**
8. KL-CMAPPO conclusion visible? **yes — red TX markers**
9. Could be simpler? **no**

## fig03
1. Single question? **How is KL-CMAPPO trained?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **BC as init stage only**
4. BC as second proposed method? **no — init not competitor**
5. More than one metric? **schematic**
6. Split obvious? **N/A**
7. Error bars legitimate? **N/A**
8. KL-CMAPPO conclusion visible? **yes — large red box**
9. Could be simpler? **no**

## fig04
1. Single question? **At similar savings, who preserves CO2 best?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **Fixed60/Delta/KL only**
4. BC as second proposed method? **no BC**
5. More than one metric? **MAE only (+ TX↓ tags)**
6. Split obvious? **validation**
7. Error bars legitimate? **KL yes; baselines n=1**
8. KL-CMAPPO conclusion visible? **yes**
9. Could be simpler? **no**

## fig05
1. Single question? **Does reduction miss important events?**
2. Understandable in ~10s? **yes — Fixed60 collapses**
3. Unnecessary methods? **same three methods**
4. BC as second proposed method? **no BC**
5. More than one metric? **recall only**
6. Split obvious? **frozen TEST**
7. Error bars legitimate? **n=1 frozen report (honest)**
8. KL-CMAPPO conclusion visible? **yes**
9. Could be simpler? **no**

## fig06
1. Single question? **Did training enter MAE≤9 feasible region?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **KL only**
4. BC as second proposed method? **N/A**
5. More than one metric? **MAE only**
6. Split obvious? **val training dynamics**
7. Error bars legitimate? **yes 5-seed**
8. KL-CMAPPO conclusion visible? **yes**
9. Could be simpler? **no**

## fig07
1. Single question? **What do KL + constraints contribute?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **ablation variants**
4. BC as second proposed method? **BC as initialization only**
5. More than one metric? **MAE only (+ TX↓)**
6. Split obvious? **val seed 42**
7. Error bars legitimate? **no multi-seed — caption required**
8. KL-CMAPPO conclusion visible? **yes — full KL red**
9. Could be simpler? **no**

## fig08
1. Single question? **Does policy transfer to unseen sensors?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **KL only (dev vs heldout)**
4. BC as second proposed method? **no BC**
5. More than one metric? **MAE only**
6. Split obvious? **TEST vs TEST**
7. Error bars legitimate? **dev n=1 frozen; heldout n=5**
8. KL-CMAPPO conclusion visible? **yes**
9. Could be simpler? **no**

## fig09
1. Single question? **Graceful degradation under packet loss?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **KL only**
4. BC as second proposed method? **N/A**
5. More than one metric? **MAE only**
6. Split obvious? **val stress**
7. Error bars legitimate? **yes 5-seed**
8. KL-CMAPPO conclusion visible? **yes**
9. Could be simpler? **no**

## fig10
1. Single question? **What do savings look like over time?**
2. Understandable in ~10s? **yes**
3. Unnecessary methods? **Fixed15/60/KL**
4. BC as second proposed method? **no BC**
5. More than one metric? **cumulative TX only**
6. Split obvious? **validation**
7. Error bars legitimate? **illustrative**
8. KL-CMAPPO conclusion visible? **yes — direct labels**
9. Could be simpler? **no**

