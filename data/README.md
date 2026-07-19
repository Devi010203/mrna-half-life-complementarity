# Data placement

The repository includes the fold-assignment CSV but does not redistribute the Saluki-released target, sequence, or feature matrices.

Place the required files as follows:

```text
data/
|-- processed/human/
|   |-- normalized.csv.gz
|   `-- splits/cv_k5_seed42.csv
`-- saluki_features/human/
    |-- alldataGeneticBiochemical.tsv.gz
    `-- alldataBiochemical.tsv.gz
```

`normalized.csv.gz` must contain the processed human `HALFLIFE` target, transcript identifiers, and the sequence columns required by the mRNA-LM and motif/k-mer scripts. The two feature tables must preserve the released Saluki feature names and identifier alignment.

The analysis expects 12,981 final human transcripts and five test folds of 2,600, 2,600, 2,600, 2,600, and 2,581 transcripts. Run `python scripts/check_setup.py` after placing the files.

Do not replace `TE.He` with an expression label: it is treated as a translation-related feature. ENCORI and `GENETIC.MIR.*` are grouped as miRNA-related annotations where applicable; only columns originating from a CWCS source should be labelled CWCS.

