# Optional External-PC1 construction

The original environment is R4.5.3, missMDA1.22, preprocessCore1.72.0 and
jsonlite2.0.0. `environment.yml` retains the original Linux conda dependency
versions/builds; package/session records provide provenance. It is separate from
both uv environments. **The conda lock alone is insufficient:** the original
successful environment rebuilt preprocessCore 1.72.0 from its official source
with `--disable-threading`. This source override is not captured by conda's
package metadata. The binary build can fail with `pthread_create()` return code
22 even with `R_THREADS=1`. Restore the recorded build before construction; see
[the exact source/build record](preprocesscore_build_info.txt).

```bash
set -euo pipefail
conda env create --prefix .release_work/r_env -f environments/r/environment.yml
mkdir -p .release_work/r_sources outputs
curl --fail --location \
  --output .release_work/r_sources/preprocessCore_1.72.0.tar.gz \
  https://bioconductor.org/packages/3.22/bioc/src/contrib/preprocessCore_1.72.0.tar.gz
printf '%s\n' '4472052371147cfbf91d223eb15f3b030d236de9b4bba6b58a205c368d53884d  .release_work/r_sources/preprocessCore_1.72.0.tar.gz' | sha256sum --check -
conda run --prefix .release_work/r_env R CMD INSTALL \
  --configure-args=--disable-threading \
  .release_work/r_sources/preprocessCore_1.72.0.tar.gz
conda run --prefix .release_work/r_env Rscript --vanilla -e \
  'stopifnot(as.character(packageVersion("preprocessCore"))=="1.72.0"); x<-matrix(c(5,2,3,4,1,6),3,2); y<-preprocessCore::normalize.quantiles(x); stopifnot(isTRUE(all.equal(y,matrix(c(5.5,1.5,3.5,3.5,1.5,5.5),3,2))))'
conda run --prefix .release_work/r_env Rscript scripts/build_external_pc1.R . outputs/pc1_rebuild
conda run --prefix .release_work/r_env Rscript scripts/build_no_qn.R . outputs/no_qn_rebuild
```

Choose new output names; the scripts reject an existing output directory.
Original algorithm/seed/ncp selection, imputation, quantile normalization,
orientation and leave-one-dataset-out construction are retained. Inputs are the
saved processed raw matrices and retained sample metadata, not FASTQ/BAM files.
This is not an end-to-end reanalysis of source sequencing experiments. The
frozen reference targets remain in `results/external/targets/`; downstream
Python statistics do not require rebuilding them.

The dependency source retains LGPL (>=2); neither its source nor installed binary
is bundled here. The pinned source SHA256 above must match before installation.
The source override restores the recorded scientific runtime; it does not change
the package version or normalization algorithm.

Primary, context-balanced and no-QN targets were independently reconstructed in
a fresh Linux environment with these versions and this source-build option.
Their 13,630, 9,786 and 13,630 saved gene-level values, respectively, matched the
frozen CSVs exactly. The Primary target retained ncp=12, orientation multiplier
-1, the 11,381 matched genes and specified model correlations (absolute tolerance
1e-10). All 14 leave-one-dataset-out targets were generated; gene counts and
summary statistics agreed with the frozen references. Original per-gene LODO
references were not bundled, so no value-by-value agreement is claimed for them.
No reference-dependent reorientation or fitting was applied. This validates
processed-matrix construction, not raw-read reprocessing, Balanced/No-QN
model-performance bootstrap intervals or target-concordance intervals.

The test installed 289 SHA256-verified, version/build-matched conda distribution
archives into a new prefix, then compiled the recorded preprocessCore source;
it did not clone an installed environment. An independent download of that
official source matched the recorded SHA256. Online installation of the entire
conda environment was not validated because upstream package-index requests
failed. The recorded-version build was tested in a Linux cluster environment,
not on other platforms.
