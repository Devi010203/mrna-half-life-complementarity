# Editable Supplementary Information

`ESM_1_Supplementary_Information.docx` is the layout-preserving source of the
complete supplement PDF, including equations, Tables S1–S14 and Figure S1.
`SUPPLEMENTARY_EN.md` is an editable text/table companion; the DOCX controls
submission layout. Neither file is a replacement for the author's main article.

The original S1–S12 table XML, equations, embedded figure and section settings
were preserved. S7 and S13–S14 extend the same document; short introductory and
S6.4 cross-references identify the added scope. The PDF was exported from the
revised DOCX, not overlaid on an earlier PDF. The three reader workbooks are
regenerated separately by `scripts/tables.py`; formal numbers come from CSVs,
not manual transcription of the DOCX or display rounding.

For a local DOCX-to-PDF export, use a compatible office renderer, for example:

```bash
mkdir -p outputs/supplement_pdf
soffice --headless --convert-to pdf --outdir outputs/supplement_pdf \
  supplementary/source/ESM_1_Supplementary_Information.docx
```

Office rendering and fonts can affect pagination; inspect the output before
submission. The supplied PDF was visually checked after export. Scholarly
prose is not automatically included in the repository's data license; see
`DATA_LICENSES.md` and the supplementary rights guide.
