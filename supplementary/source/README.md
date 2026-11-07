# Editable Supplementary Information

The canonical submission file is `../ESM_1_Supplementary_Information.pdf`, the
author-approved Windows/Acrobat export dated 7 November 2026.
`ESM_1_Supplementary_Information.docx` and `SUPPLEMENTARY_EN.md` are content-aligned
editable companions, including equations, Tables S1–S14 and Figures S1–S2.
The exact Word file used for that Windows export was not supplied; these
companions incorporate the approved reference/language corrections and the
PDF's S13b heading. They do not guarantee identical pagination, mathematical
typesetting or PDF bytes. Neither replaces the author's main article.

Scientific table cells, equations and embedded figure pixels were preserved.
S7 and S13–S14 extend the same document; short introductory and
S6.4 cross-references identify the added scope. S2 adds the relocated external
prediction scatter panels, while S7.3 cites main Figure 2b,c for the three-target
comparison. The current PDF has the final supplementary header, author–year
reference wording, the Zenodo dataset DOI and the clarified separate SHAP-model
description. Figure S1's caption and image share page 32; Figure S2 is on page 33.
The author-selected PDF is distributed without modification. The three reader workbooks are
regenerated separately by `scripts/tables.py`; formal numbers come from CSVs,
not manual transcription of the DOCX or display rounding.

For a local DOCX-to-PDF export, use a compatible office renderer, for example:

```bash
mkdir -p outputs/supplement_pdf
soffice --headless --convert-to pdf --outdir outputs/supplement_pdf \
  supplementary/source/ESM_1_Supplementary_Information.docx
```

Office rendering and fonts can affect pagination; inspect the output before
submission. The supplied 33-page PDF was inspected, including its equation,
tables and both figure pages. Acrobat downsampled the embedded figures to about
300 ppi; the unchanged standalone PNGs in `../../figures/` are 600 dpi, and their
PDF/SVG counterparts remain available. The DOCX retains the original embedded
image pixels. No new image resampling was performed during synchronization.
Scholarly
prose is not automatically included in the repository's data license; see
`DATA_LICENSES.md` and the supplementary rights guide.
