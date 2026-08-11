# Baseline rationale

## Fixed 15
Reference periodic system and definition of communication savings (0% reduction).

## Fixed 30 / Fixed 60
Simple periodic baselines. **Fixed 60** is especially important because its TX reduction (~75.3%) is close to KL-CMAPPO (~77.8%), providing an intuitive nearly matched-load comparison in the main MAE bar chart.

## Delta + heartbeat
Simple adaptive heuristic (send-on-change + heartbeat). Kept in the **main results table** and supplementary all-baseline MAE figure. Not required in every main graph.

## Semantic expert
Teacher/reference that generates BC demonstrations. Remains a strong handcrafted method and **must appear in the main table**. Shown in graphs only when the question concerns expert→BC→RL progression or full benchmarking (supplementary).

## BC
Direct learned predecessor of KL-CMAPPO. Essential for proving the RL fine-tuning contribution.

## KL-CMAPPO
Proposed final policy (no safety shield).
