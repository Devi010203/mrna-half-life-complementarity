#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(missMDA)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript scripts/build_no_qn.R REPOSITORY NEW_OUTPUT_DIR")
external_root <- normalizePath(args[[1]])
input_dir <- file.path(external_root,"data","external","construction")
pc1_dir <- file.path(external_root,"results","external","targets")
output_dir <- args[[2]]
if (dir.exists(output_dir) || file.exists(output_dir)) stop("Choose a new output directory")
parent <- normalizePath(dirname(output_dir),mustWork=TRUE)
if (parent != file.path(external_root,"outputs") && !startsWith(parent,paste0(external_root,"/outputs/"))) stop("Output parent must exist under repository outputs/")
output_dir <- file.path(parent,basename(output_dir))
if (basename(output_dir) %in% c(".","..")) stop("Invalid output name")
dir.create(output_dir)

read_numeric_matrix <- function(path) {
  frame <- read.csv(gzfile(path), check.names = FALSE, stringsAsFactors = FALSE)
  identifiers <- frame[[1L]]
  matrix <- as.matrix(frame[-1L])
  storage.mode(matrix) <- "double"
  rownames(matrix) <- identifiers
  matrix
}

safe_cor <- function(x, y, method = "pearson") {
  keep <- is.finite(x) & is.finite(y)
  if (sum(keep) < 3L) return(NA_real_)
  cor(x[keep], y[keep], method = method)
}

cat("Reading frozen Stage 6.5 raw matrix; Primary target will remain read-only\n")
raw <- read_numeric_matrix(file.path(input_dir, "external_matrix_raw_ge10.csv.gz"))
sample_meta <- read.csv(
  file.path(input_dir, "retained_sample_metadata.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
coverage <- read.csv(file.path(input_dir, "external_gene_coverage.csv"), stringsAsFactors = FALSE)
primary <- read.csv(file.path(pc1_dir, "external_pc1_primary.csv"), stringsAsFactors = FALSE)

stopifnot(
  nrow(raw) == 13630L,
  ncol(raw) == 62L,
  identical(colnames(raw), sample_meta$sample),
  all(rowSums(is.finite(raw)) >= 10L),
  identical(rownames(raw), primary$feature_ID)
)

scaled <- scale(raw)
if (any(!is.finite(attr(scaled, "scaled:scale")))) {
  stop("Non-finite sample SD after base::scale()")
}

requested_ncp_max <- 20L
effective_ncp_max <- min(requested_ncp_max, nrow(scaled) - 1L, ncol(scaled) - 1L)
set.seed(42L)
ncp_estimate <- estim_ncpPCA(
  scaled,
  ncp.min = 0L,
  ncp.max = effective_ncp_max,
  verbose = FALSE
)
selected_ncp <- as.integer(ncp_estimate$ncp)
cat(sprintf("No-QN selected ncp=%d\n", selected_ncp))

set.seed(42L)
imputed <- imputePCA(scaled, ncp = selected_ncp)$completeObs
dimnames(imputed) <- dimnames(raw)
if (any(!is.finite(imputed))) stop("No-QN imputed matrix contains non-finite values")

# The defining sensitivity change: do not call normalize.quantiles().
pca <- stats::prcomp(as.matrix(imputed))
pc1_unoriented <- pca$x[, 1L]
orientation_correlations <- vapply(
  seq_len(ncol(raw)),
  function(index) safe_cor(pc1_unoriented, raw[, index], method = "pearson"),
  numeric(1L)
)
names(orientation_correlations) <- colnames(raw)
orientation_median <- median(orientation_correlations, na.rm = TRUE)
orientation_multiplier <- if (orientation_median < 0) -1 else 1
external_pc1_no_qn <- pc1_unoriented * orientation_multiplier

target <- data.frame(
  feature_ID = rownames(raw),
  external_PC1_noQN = external_pc1_no_qn,
  n_measurements = rowSums(is.finite(raw))
)
write.csv(target, file.path(output_dir, "external_pc1_no_qn.csv"), row.names = FALSE)

variance_ratio <- pca$sdev^2 / sum(pca$sdev^2)
variance <- data.frame(
  component = paste0("PC", seq_along(variance_ratio)),
  eigenvalue = pca$sdev^2,
  explained_variance_ratio = variance_ratio,
  cumulative_explained_variance = cumsum(variance_ratio)
)
write.csv(variance, file.path(output_dir, "external_pc1_no_qn_variance.csv"), row.names = FALSE)

orientation <- data.frame(
  sample = names(orientation_correlations),
  pearson_PC1_unoriented_vs_raw_neg_log_kdeg = as.numeric(orientation_correlations),
  median_pearson = orientation_median,
  sign_multiplier = orientation_multiplier,
  orientation_definition = "External-PC1-noQN higher = longer/more stable RNA"
)
write.csv(orientation, file.path(output_dir, "external_pc1_no_qn_orientation.csv"), row.names = FALSE)

common <- intersect(primary$feature_ID, target$feature_ID)
primary_values <- primary$external_PC1[match(common, primary$feature_ID)]
no_qn_values <- target$external_PC1_noQN[match(common, target$feature_ID)]
primary_variance <- read.csv(file.path(pc1_dir, "external_pca_variance.csv"), stringsAsFactors = FALSE)
primary_config <- read_json(file.path(input_dir, "pc1_config.json"), simplifyVector = TRUE)
comparison <- data.frame(
  n_primary = nrow(primary),
  n_no_qn = nrow(target),
  n_common = length(common),
  spearman = safe_cor(primary_values, no_qn_values, "spearman"),
  pearson = safe_cor(primary_values, no_qn_values, "pearson"),
  primary_pc1_explained_variance = primary_variance$explained_variance_ratio[primary_variance$component == "PC1"],
  no_qn_pc1_explained_variance = variance_ratio[[1L]],
  primary_selected_ncp = primary_config$estim_ncpPCA$selected_ncp,
  no_qn_selected_ncp = selected_ncp,
  interpretation = "assigned in the Stage 10 report from observed concordance; not used to tune the method"
)
write.csv(comparison, file.path(output_dir, "primary_vs_no_qn.csv"), row.names = FALSE)

config <- list(
  analysis = "no-quantile-normalization sensitivity",
  representation = "-log(kdeg)",
  genes = nrow(raw),
  samples = ncol(raw),
  measurement_threshold = 10L,
  scaling = "base::scale with sample SD",
  estim_ncpPCA = list(
    seed = 42L,
    ncp_min = 0L,
    requested_ncp_max = requested_ncp_max,
    effective_ncp_max = effective_ncp_max,
    method_cv = "gcv",
    selected_ncp = selected_ncp
  ),
  imputation = "missMDA::imputePCA with selected ncp and package defaults",
  quantile_normalization = "SKIPPED BY DESIGN",
  pca = "stats::prcomp(center=TRUE default, scale.=FALSE default)",
  orientation = list(
    rule = "median Pearson correlation of unoriented PC1 with retained raw -log(kdeg) sample columns",
    median_pearson = orientation_median,
    multiplier = orientation_multiplier,
    model_blind = TRUE
  ),
  primary_target_modified = FALSE,
  model_predictions_read = FALSE
)
write_json(config, file.path(output_dir, "external_pc1_no_qn_config.json"), pretty = TRUE, auto_unbox = TRUE)

session_lines <- c(
  paste0("R version: ", R.version.string),
  paste0("missMDA version: ", packageVersion("missMDA")),
  paste0("stats version: ", packageVersion("stats")),
  paste0("selected ncp: ", selected_ncp),
  "seed: 42",
  "quantile normalization: skipped by sensitivity definition",
  "model predictions read: false",
  "Primary target modified: false",
  "",
  "sessionInfo():",
  capture.output(sessionInfo())
)
writeLines(session_lines, file.path(output_dir, "session_info.txt"))

cat(sprintf(
  "Completed no-QN sensitivity: genes=%d ncp=%d PC1_variance=%.8f Spearman=%.8f Pearson=%.8f\n",
  nrow(target), selected_ncp, variance_ratio[[1L]], comparison$spearman, comparison$pearson
))
