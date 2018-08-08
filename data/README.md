# DATA README

## ZINC.FL

Fragrance-like subset of the ZINC database. Originally sourced from [here](gdb.unibe.ch/downloads)<br>
37661 molecules. SMILES strings followed by various database identifiers.

## MMP12

MMP-12 inhibitor activity for a 50x50 assay series. Originally sourced from [here](https://pubs.acs.org/doi/abs/10.1021/ml100191f)<br>
`Index   Tag   Atag   Btag   pIC50_coded   Smiles`<br>
pIC50 values are codes as follows:
 - > 0: pIC50 value
 - = 0: Inactive
 - = -1: Assay failed
 - = -2: Not assayed
 - = -3: Not made
