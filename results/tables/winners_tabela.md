| stage | run_id | configuration | selection_metric | value | notes |
|---|---|---|---|---|---|
| Q2 | q2_lr0.01_m0.9 | lr=0.01; momentum=0.9 | macro_f1_val | 0.94773 | converged=False; epochs=150 |
| Q3_top1 | q3_l1_n60_s42 | layers=1; neurons=60 | macro_f1_val | 0.94773 | loss_val=0.1934; time_s=154.1 |
| Q3_top2 | q3_l1_n100_s42 | layers=1; neurons=100 | macro_f1_val | 0.94726 | loss_val=0.1924; time_s=143.6 |
| Q3_top3 | q3_l1_n40_s42 | layers=1; neurons=40 | macro_f1_val | 0.94708 | loss_val=0.1875; time_s=107.5 |
| Q3_top4 | q3_l1_n90_s42 | layers=1; neurons=90 | macro_f1_val | 0.94700 | loss_val=0.1906; time_s=126.3 |
| Q4 | q4_frac100 | fraction=1.0 | macro_f1_val | 0.94773 | análise de sensibilidade; Q6 usa o desenvolvimento completo |
| Q5 | q5_low_redundancy | subset=low_redundancy; n_features=8 | macro_f1_val | 0.94518 | Perimeter; Extent; Solidity; Roundness; ShapeFactor1; ShapeFactor2; ShapeFactor3; ShapeFactor4 |
| Q6 | q6_q3_l1_n100_s42 | layers=1; neurons=100; lr=0.01; momentum=0.9; features=8 | macro_f1_test | 0.94022 | weighted_f1_test=0.9280 |
| Q7 | q6_q3_l1_n100_s42 | k=5; desenvolvimento 70% | macro_f1_val_mean | 0.94535 | variance=0.000024; std=0.004933 |
