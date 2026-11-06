# Compare new Axis profiles with already computed original-kodonz baselines.
# No baseline profile is recomputed, replaced or used to select Axis outputs.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Expected axis-profiles.tsv baseline-profiles.tsv output.tsv")
options(digits = 17)
library(dtw)
read_profiles <- function(path) {
  read.delim(path, stringsAsFactors = FALSE, check.names = FALSE,
             quote = "\"", comment.char = "", na.strings = character())
}
axis <- read_profiles(args[[1]])
baseline <- read_profiles(args[[2]])
if (anyDuplicated(axis$case_id)) stop("Multiple Axis outputs for a requested case")
if (anyDuplicated(baseline[c("case_id", "model")])) stop("Duplicate baseline profile")
if (!setequal(axis$case_id, baseline$case_id)) stop("Case sets differ")
rows <- vector("list", nrow(baseline))
for (j in seq_len(nrow(baseline))) {
  i <- match(baseline$case_id[[j]], axis$case_id)
  if (axis$organism[[i]] != baseline$organism[[j]]) stop("Host mismatch")
  row <- data.frame(case_id = axis$case_id[[i]], organism = axis$organism[[i]],
                    model1 = axis$model[[i]], model2 = baseline$model[[j]],
                    status = "unscored_input", distance = NA_real_, normalized_distance = NA_real_)
  if (axis$status[[i]] == "scored" && baseline$status[[j]] == "scored") {
    a <- as.numeric(strsplit(axis$minmax[[i]], ",", fixed = TRUE)[[1]])
    b <- as.numeric(strsplit(baseline$minmax[[j]], ",", fixed = TRUE)[[1]])
    if (any(!is.finite(c(a, b)))) stop("Nonfinite scored input profile")
    comparison <- dtw(a, b)
    row$status <- "scored"
    row$distance <- comparison$distance
    row$normalized_distance <- comparison$normalizedDistance
    if (!all(is.finite(c(row$distance, row$normalized_distance)))) stop("Nonfinite DTW result")
  }
  rows[[j]] <- row
}
write.table(do.call(rbind, rows), args[[3]], sep = "\t", row.names = FALSE, quote = TRUE)
cat("AXIS_VS_CACHED_BASELINE_DTW_COMPLETE", length(rows), "pairs\n")
