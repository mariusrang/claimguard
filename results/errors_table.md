## Error analysis (110 errors annotated by hand)

| category                 |   Climate-FEVER |   SciFact |   total | share   |   mean conf. | main confusion (gold→pred)   |
|:-------------------------|----------------:|----------:|--------:|:--------|-------------:|:-----------------------------|
| wrong_sentences          |               7 |        25 |      32 | 29%     |         0.7  | SUPPORTED→NEI                |
| related_but_insufficient |               2 |        17 |      19 | 17%     |         0.73 | NEI→SUPPORTED                |
| refuted_vs_nei           |               4 |        11 |      15 | 14%     |         0.73 | NEI→REFUTED                  |
| domain_vocabulary        |               0 |        15 |      15 | 14%     |         0.75 | NEI→SUPPORTED                |
| evidence_not_retrieved   |               0 |         9 |       9 | 8%      |         0.76 | REFUTED→NEI                  |
| numeric                  |               1 |         4 |       5 | 5%      |         0.73 | SUPPORTED→NEI                |
| multi_evidence           |               0 |         5 |       5 | 5%      |         0.63 | SUPPORTED→NEI                |
| negation_quantifier      |               1 |         4 |       5 | 5%      |         0.74 | REFUTED→SUPPORTED            |
| gold_label_questionable  |               0 |         5 |       5 | 5%      |         0.84 | NEI→SUPPORTED                |
