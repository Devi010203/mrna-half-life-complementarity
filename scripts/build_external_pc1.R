#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(missMDA)
  library(preprocessCore)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript scripts/build_external_pc1.R REPOSITORY NEW_OUTPUT_DIR")
external_root <- normalizePath(args[[1]])
input_dir <- file.path(external_root, "data", "external", "construction")
pc1_dir <- args[[2]]
if (dir.exists(pc1_dir) || file.exists(pc1_dir)) stop("Choose a new output directory")
parent <- normalizePath(dirname(pc1_dir), mustWork=TRUE)
if (parent != file.path(external_root,"outputs") && !startsWith(parent,paste0(external_root,"/outputs/"))) stop("Output parent must exist under repository outputs/")
pc1_dir <- file.path(parent,basename(pc1_dir))
if (basename(pc1_dir) %in% c(".","..")) stop("Invalid output name")
dir.create(pc1_dir)
lodo_dir <- file.path(pc1_dir, "lodo")
dir.create(lodo_dir)

read_numeric_matrix <- function(path) {
  frame <- read.csv(gzfile(path), check.names = FALSE, stringsAsFactors = FALSE)
  ids <- frame[[1]]
  matrix <- as.matrix(frame[-1])
  storage.mode(matrix) <- "double"
  rownames(matrix) <- ids
  list(ids = ids, matrix = matrix)
}

write_csv_gz <- function(frame, path) {
  connection <- gzfile(path, open = "wt")
  on.exit(close(connection), add = TRUE)
  write.csv(frame, connection, row.names = FALSE, na = "")
}

safe_cor <- function(x, y, method = "pearson") {
  keep <- is.finite(x) & is.finite(y)
  if (sum(keep) < 3L) return(NA_real_)
  cor(x[keep], y[keep], method = method)
}

estimate_ncp <- function(scaled_matrix, seed, requested_max = 20L) {
  effective_max <- min(requested_max, nrow(scaled_matrix) - 1L, ncol(scaled_matrix) - 1L)
  if (effective_max < 1L) stop("Matrix is too small for PCA ncp estimation")
  set.seed(seed)
  estimate <- estim_ncpPCA(
    scaled_matrix,
    ncp.min = 0,
    ncp.max = effective_max,
    verbose = FALSE
  )
  list(selected = as.integer(estimate$ncp), effective_max = effective_max)
}

fit_consensus <- function(raw_matrix, seed = 42L, requested_max = 20L) {
  if (any(colSums(is.finite(raw_matrix)) < 2L)) stop("A retained column has fewer than two finite observations")
  scaled <- scale(raw_matrix)
  if (any(!is.finite(attr(scaled, "scaled:scale")))) stop("Non-finite sample SD after scale()")
  ncp <- estimate_ncp(scaled, seed = seed, requested_max = requested_max)
  set.seed(seed)
  imputed <- imputePCA(scaled, ncp = ncp$selected)$completeObs
  colnames(imputed) <- colnames(raw_matrix)
  rownames(imputed) <- rownames(raw_matrix)
  quantile_normalized <- normalize.quantiles(imputed)
  dimnames(quantile_normalized) <- dimnames(imputed)
  pca <- stats::prcomp(as.matrix(quantile_normalized))
  raw_pc1 <- pca$x[, 1]
  orientation_correlations <- vapply(
    seq_len(ncol(raw_matrix)),
    function(index) safe_cor(raw_pc1, raw_matrix[, index], method = "pearson"),
    numeric(1)
  )
  names(orientation_correlations) <- colnames(raw_matrix)
  median_correlation <- median(orientation_correlations, na.rm = TRUE)
  multiplier <- if (median_correlation < 0) -1 else 1
  list(
    raw = raw_matrix,
    scaled = scaled,
    imputed = imputed,
    quantile_normalized = quantile_normalized,
    pca = pca,
    external_pc1 = raw_pc1 * multiplier,
    orientation_correlations = orientation_correlations,
    orientation_median = median_correlation,
    orientation_multiplier = multiplier,
    selected_ncp = ncp$selected,
    requested_ncp_max = requested_max,
    effective_ncp_max = ncp$effective_max,
    seed = seed
  )
}

eta_squared <- function(values, groups) {
  keep <- is.finite(values) & !is.na(groups) & groups != ""
  values <- values[keep]
  groups <- factor(groups[keep])
  if (length(values) < 3L || nlevels(groups) < 2L) return(NA_real_)
  grand <- mean(values)
  ss_total <- sum((values - grand)^2)
  ss_between <- sum(vapply(split(values, groups), function(x) length(x) * (mean(x) - grand)^2, numeric(1)))
  if (ss_total == 0) NA_real_ else ss_between / ss_total
}

cat("Reading model-blind Stage 6.5 matrices\n")
raw_all_object <- read_numeric_matrix(file.path(input_dir, "external_matrix_raw_all.csv.gz"))
raw_all <- raw_all_object$matrix
sample_meta <- read.csv(file.path(input_dir, "retained_sample_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(identical(colnames(raw_all), sample_meta$sample))
gene_coverage <- read.csv(file.path(input_dir, "external_gene_coverage.csv"), stringsAsFactors = FALSE)

primary_counts <- rowSums(is.finite(raw_all))
primary_raw <- raw_all[primary_counts >= 10L, , drop = FALSE]
cat(sprintf("Primary: %d genes x %d samples; estimating ncp\n", nrow(primary_raw), ncol(primary_raw)))
primary <- fit_consensus(primary_raw, seed = 42L, requested_max = 20L)
cat(sprintf("Primary selected ncp=%d; orientation multiplier=%d\n", primary$selected_ncp, primary$orientation_multiplier))

write_csv_gz(data.frame(feature_ID = rownames(primary$scaled), primary$scaled, check.names = FALSE), file.path(pc1_dir, "external_matrix_scaled.csv.gz"))
write_csv_gz(data.frame(feature_ID = rownames(primary$imputed), primary$imputed, check.names = FALSE), file.path(pc1_dir, "external_matrix_imputed.csv.gz"))
write_csv_gz(data.frame(feature_ID = rownames(primary$quantile_normalized), primary$quantile_normalized, check.names = FALSE), file.path(pc1_dir, "external_matrix_quantile_normalized.csv.gz"))

scores <- as.data.frame(primary$pca$x[, seq_len(min(5L, ncol(primary$pca$x))), drop = FALSE])
scores$PC1_unoriented <- primary$pca$x[, 1]
scores$external_PC1 <- primary$external_pc1
scores <- data.frame(feature_ID = rownames(scores), scores, check.names = FALSE)
write.csv(scores, file.path(pc1_dir, "external_pca_scores.csv"), row.names = FALSE)

loadings <- as.data.frame(primary$pca$rotation[, seq_len(min(5L, ncol(primary$pca$rotation))), drop = FALSE])
loadings$PC1_unoriented <- primary$pca$rotation[, 1]
loadings$external_PC1_loading <- primary$pca$rotation[, 1] * primary$orientation_multiplier
loadings <- data.frame(sample = rownames(loadings), loadings, check.names = FALSE)
write.csv(loadings, file.path(pc1_dir, "external_pca_loadings.csv"), row.names = FALSE)

variance_ratio <- primary$pca$sdev^2 / sum(primary$pca$sdev^2)
variance <- data.frame(
  component = paste0("PC", seq_along(variance_ratio)),
  eigenvalue = primary$pca$sdev^2,
  explained_variance_ratio = variance_ratio,
  cumulative_explained_variance = cumsum(variance_ratio)
)
write.csv(variance, file.path(pc1_dir, "external_pca_variance.csv"), row.names = FALSE)

primary_target <- data.frame(feature_ID = rownames(primary_raw), external_PC1 = primary$external_pc1)
primary_target <- merge(primary_target, gene_coverage, by = "feature_ID", all.x = TRUE, sort = FALSE)
primary_target <- primary_target[match(rownames(primary_raw), primary_target$feature_ID), ]
write.csv(primary_target, file.path(pc1_dir, "external_pc1_primary.csv"), row.names = FALSE)

orientation <- data.frame(
  sample = names(primary$orientation_correlations),
  pearson_PC1_unoriented_vs_raw_neg_log_kdeg = as.numeric(primary$orientation_correlations),
  median_pearson = primary$orientation_median,
  sign_multiplier = primary$orientation_multiplier,
  orientation_definition = "External-PC1 higher = longer/more stable RNA"
)
write.csv(orientation, file.path(pc1_dir, "external_pc1_orientation.csv"), row.names = FALSE)

sample_correlations <- do.call(rbind, lapply(seq_len(ncol(primary_raw)), function(index) {
  raw_values <- primary_raw[, index]
  overlap <- is.finite(raw_values) & is.finite(primary$external_pc1)
  data.frame(
    sample = colnames(primary_raw)[index],
    n_overlap_genes = sum(overlap),
    pearson = safe_cor(primary$external_pc1, raw_values, "pearson"),
    spearman = safe_cor(primary$external_pc1, raw_values, "spearman")
  )
}))
sample_correlations <- merge(sample_meta, sample_correlations, by = "sample", sort = FALSE)
sample_correlations <- sample_correlations[match(colnames(primary_raw), sample_correlations$sample), ]
q1 <- quantile(sample_correlations$spearman, 0.25, na.rm = TRUE)
iqr <- IQR(sample_correlations$spearman, na.rm = TRUE)
sample_correlations$diagnostic_flag <- ifelse(
  sample_correlations$spearman < 0,
  "REVERSE",
  ifelse(sample_correlations$spearman < q1 - 1.5 * iqr, "LOW_IQR_OUTLIER", "OK")
)
write.csv(sample_correlations, file.path(pc1_dir, "pc1_sample_correlations.csv"), row.names = FALSE)

loading_audit <- merge(sample_meta, loadings, by = "sample", sort = FALSE)
loading_audit <- loading_audit[match(loadings$sample, loading_audit$sample), ]
write.csv(loading_audit, file.path(pc1_dir, "pc1_loading_audit.csv"), row.names = FALSE)
dataset_loading <- do.call(rbind, lapply(split(loading_audit, loading_audit$dataset), function(frame) {
  data.frame(
    dataset = frame$dataset[1],
    samples = nrow(frame),
    sum_squared_PC1_loading = sum(frame$external_PC1_loading^2),
    mean_absolute_PC1_loading = mean(abs(frame$external_PC1_loading)),
    max_absolute_PC1_loading = max(abs(frame$external_PC1_loading))
  )
}))
write.csv(dataset_loading, file.path(pc1_dir, "pc1_loading_dataset_summary.csv"), row.names = FALSE)

group_variables <- c("dataset", "cell_line", "method", "library_type")
group_loading <- do.call(rbind, lapply(group_variables, function(variable) {
  do.call(rbind, lapply(split(loading_audit, loading_audit[[variable]]), function(frame) {
    data.frame(
      group_type = variable,
      group = frame[[variable]][1],
      samples = nrow(frame),
      mean_loading = mean(frame$external_PC1_loading),
      mean_absolute_loading = mean(abs(frame$external_PC1_loading)),
      max_absolute_loading = max(abs(frame$external_PC1_loading))
    )
  }))
}))
write.csv(group_loading, file.path(pc1_dir, "pc1_loading_group_summary.csv"), row.names = FALSE)

technical_rows <- list()
counter <- 1L
for (component in c("PC2", "PC3")) {
  values <- loading_audit[[component]]
  for (variable in group_variables) {
    technical_rows[[counter]] <- data.frame(
      component = component,
      variable = variable,
      association_type = "eta_squared",
      value = eta_squared(values, loading_audit[[variable]])
    )
    counter <- counter + 1L
  }
  label_time <- suppressWarnings(as.numeric(loading_audit$label_time))
  technical_rows[[counter]] <- data.frame(component = component, variable = "label_time", association_type = "pearson", value = safe_cor(values, label_time, "pearson"))
  counter <- counter + 1L
  technical_rows[[counter]] <- data.frame(component = component, variable = "label_time", association_type = "spearman", value = safe_cor(values, label_time, "spearman"))
  counter <- counter + 1L
}
technical <- do.call(rbind, technical_rows)
write.csv(technical, file.path(pc1_dir, "pc2_pc3_technical_associations.csv"), row.names = FALSE)

cat("Running leave-one-dataset-out refits\n")
datasets <- unique(sample_meta$dataset)
lodo_cores <- suppressWarnings(as.integer(Sys.getenv("EXTERNAL_PC1_LODO_CORES", "4")))
if (!is.finite(lodo_cores) || lodo_cores < 1L) lodo_cores <- 1L
lodo_cores <- min(lodo_cores, length(datasets))
cat(sprintf("LODO workers=%d (independent full refits)\n", lodo_cores))

run_lodo <- function(index) {
  removed <- datasets[[index]]
  keep_samples <- sample_meta$sample[sample_meta$dataset != removed]
  lodo_all <- raw_all[, keep_samples, drop = FALSE]
  counts <- rowSums(is.finite(lodo_all))
  lodo_raw <- lodo_all[counts >= 10L, , drop = FALSE]
  cat(sprintf("LODO %02d/%02d remove=%s genes=%d samples=%d\n", index, length(datasets), removed, nrow(lodo_raw), ncol(lodo_raw)))
  fit <- fit_consensus(lodo_raw, seed = 42L, requested_max = 20L)
  common <- intersect(rownames(primary_raw), rownames(lodo_raw))
  full_values <- primary$external_pc1[match(common, rownames(primary_raw))]
  lodo_values <- fit$external_pc1[match(common, rownames(lodo_raw))]
  result <- data.frame(
    removed_dataset = removed,
    full_genes = nrow(primary_raw),
    lodo_genes = nrow(lodo_raw),
    common_genes = length(common),
    selected_ncp = fit$selected_ncp,
    effective_ncp_max = fit$effective_ncp_max,
    orientation_multiplier = fit$orientation_multiplier,
    spearman_full_vs_lodo = safe_cor(full_values, lodo_values, "spearman"),
    pearson_full_vs_lodo = safe_cor(full_values, lodo_values, "pearson")
  )
  write_csv_gz(
    data.frame(feature_ID = rownames(lodo_raw), external_PC1 = fit$external_pc1),
    file.path(lodo_dir, paste0("external_pc1_without_", gsub("[^A-Za-z0-9]+", "_", removed), ".csv.gz"))
  )
  result
}

lodo_rows <- if (lodo_cores > 1L && .Platform$OS.type == "unix") {
  parallel::mclapply(seq_along(datasets), run_lodo, mc.cores = lodo_cores, mc.preschedule = FALSE)
} else {
  lapply(seq_along(datasets), run_lodo)
}
if (any(vapply(lodo_rows, inherits, logical(1), what = "try-error"))) {
  stop("At least one LODO refit failed; inspect the preceding worker error")
}
lodo_summary <- do.call(rbind, lodo_rows)
write.csv(lodo_summary, file.path(pc1_dir, "pc1_robustness_summary.csv"), row.names = FALSE)

cat("Running dataset x cell-line context-balanced sensitivity\n")
context_object <- read_numeric_matrix(file.path(input_dir, "context_balanced_matrix_raw_all.csv.gz"))
context_all <- context_object$matrix
context_counts <- rowSums(is.finite(context_all))
context_raw <- context_all[context_counts >= 10L, , drop = FALSE]
context_fit <- fit_consensus(context_raw, seed = 42L, requested_max = 20L)
context_target <- data.frame(
  feature_ID = rownames(context_raw),
  external_PC1_context_balanced = context_fit$external_pc1,
  n_context_measurements = context_counts[rownames(context_raw)]
)
write.csv(context_target, file.path(pc1_dir, "external_pc1_context_balanced.csv"), row.names = FALSE)

context_orientation <- data.frame(
  context_id = names(context_fit$orientation_correlations),
  pearson_PC1_unoriented_vs_raw_neg_log_kdeg = as.numeric(context_fit$orientation_correlations),
  median_pearson = context_fit$orientation_median,
  sign_multiplier = context_fit$orientation_multiplier
)
write.csv(context_orientation, file.path(pc1_dir, "context_balanced_orientation.csv"), row.names = FALSE)

common <- intersect(primary_target$feature_ID, context_target$feature_ID)
primary_common <- primary_target$external_PC1[match(common, primary_target$feature_ID)]
context_common <- context_target$external_PC1_context_balanced[match(common, context_target$feature_ID)]
comparison <- data.frame(
  n_primary = nrow(primary_target),
  n_context_balanced = nrow(context_target),
  n_common = length(common),
  spearman = safe_cor(primary_common, context_common, "spearman"),
  pearson = safe_cor(primary_common, context_common, "pearson"),
  interpretation = "assigned in final report from observed concordance; not used as a pass threshold"
)
write.csv(comparison, file.path(pc1_dir, "primary_vs_context_balanced.csv"), row.names = FALSE)

config <- list(
  seed = 42L,
  representation = "-log(kdeg)",
  primary_threshold = 10L,
  scaling = "base::scale with sample SD",
  estim_ncpPCA = list(ncp_min = 0L, requested_ncp_max = 20L, effective_ncp_max = primary$effective_ncp_max, method = "Regularized", scale = TRUE, method_cv = "gcv", selected_ncp = primary$selected_ncp, randomness = "default gcv path is deterministic; additional seed sweep not applicable"),
  imputation = "missMDA::imputePCA with selected ncp and package defaults after set.seed(42)",
  normalization = "preprocessCore::normalize.quantiles",
  pca = "stats::prcomp(center=TRUE default, scale.=FALSE default)",
  orientation_median_pearson = primary$orientation_median,
  orientation_multiplier = primary$orientation_multiplier,
  context_balanced = list(threshold = 10L, contexts = ncol(context_raw), genes = nrow(context_raw), selected_ncp = context_fit$selected_ncp, effective_ncp_max = context_fit$effective_ncp_max),
  lodo_workers = lodo_cores,
  model_predictions_read = FALSE
)
write_json(config, file.path(pc1_dir, "pc1_config.json"), pretty = TRUE, auto_unbox = TRUE)

session_lines <- c(
  paste0("R version: ", R.version.string),
  paste0("missMDA version: ", packageVersion("missMDA")),
  paste0("preprocessCore version: ", packageVersion("preprocessCore")),
  paste0("stats version: ", packageVersion("stats")),
  paste0("selected ncp: ", primary$selected_ncp),
  "seed: 42",
  "estim_ncpPCA method.cv: gcv (deterministic default; seed stability sweep not applicable)",
  "",
  "sessionInfo():",
  capture.output(sessionInfo())
)
writeLines(session_lines, file.path(pc1_dir, "session_info.txt"))

cat(sprintf("Completed Stage 7-9: primary genes=%d, selected ncp=%d, context genes=%d\n", nrow(primary_target), primary$selected_ncp, nrow(context_target)))
