# Pretrained checkpoint placement

The sequence-only scripts expect the pretrained region-specific model directories below unless their paths are overridden in the model configuration:

```text
models/
|-- mrna_5utr_model_p2_cp85600_best/
|-- codonbert/
`-- mrna_3utr_model_p2_cp99900_best/
```

These checkpoints are not redistributed in this repository. Obtain the official pretrained weights from the corresponding mRNA-LM release and retain the original license and citation information.

The main AggHead experiment fine-tunes adapters on the processed human half-life target; it is not simply an inference-only use of a published downstream half-life checkpoint. Detailed LoRA and sequence-window settings remain in the scripts and Supplementary Methods.

