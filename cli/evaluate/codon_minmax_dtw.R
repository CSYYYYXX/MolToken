# Worker-side implementation of the paper's original kodonz + dtw calls.
# Arguments: kodonz source root, input TSV, profile TSV, pairwise-distance TSV.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("Expected kodonz-source input.tsv profiles.tsv pairs.tsv")
options(digits = 17)
library(dtw)
source_root <- args[[1]]
kz <- new.env(parent = globalenv())
for (name in c("CodonTable0", "cft_ecoli", "cft_saccharomyces",
               "cft_arabidopsis", "cft_mouse", "cft_human")) {
  load(file.path(source_root, "data", paste0(name, ".rda")), envir = kz)
}
for (name in c("arg_check.r", "load.fasta.r", "pminmax.r")) {
  sys.source(file.path(source_root, "R", name), envir = kz)
}
species <- c("Escherichia coli" = "ecoli",
             "Saccharomyces cerevisiae" = "saccharomyces",
             "Arabidopsis thaliana" = "arabidopsis",
             "Mus musculus" = "mouse", "Homo sapiens" = "human")
records <- read.delim(args[[2]], stringsAsFactors = FALSE, check.names = FALSE,
                      quote = "\"", comment.char = "", na.strings = character())
stopifnot(all(c("case_id", "organism", "model", "dna") %in% names(records)))
if (nrow(records) == 0) stop("No input records")
if (anyDuplicated(records[c("case_id", "model")])) stop("Duplicate case/model input")
profiles <- vector("list", nrow(records))
statuses <- rep("scored", nrow(records))
for (i in seq_len(nrow(records))) {
  dna <- records$dna[[i]]
  host <- records$organism[[i]]
  if (!host %in% names(species)) stop(paste("Unmapped host:", host))
  if (nchar(dna) == 0) {
    statuses[[i]] <- "empty_or_decode_failure"
  } else if (!grepl("^[ACGT]+$", dna)) {
    statuses[[i]] <- "noncanonical_dna"
  } else if (nchar(dna) %% 3 != 0) {
    statuses[[i]] <- "incomplete_codon"
  } else if (nchar(dna) < 54) {
    statuses[[i]] <- "shorter_than_18_codons"
  } else {
    seq <- kz$load.fasta(x = dna)
    profiles[[i]] <- kz$pminmax(seq[[1]], z = 18, y = 0, spp = species[[host]])
    # Non-degenerate-only windows can have a zero denominator in pminmax.
    # Preserve the raw profile and flag this expected undefined metric; do not
    # replace NaN/Inf with zero or attempt DTW on it.
    if (any(!is.finite(profiles[[i]]))) statuses[[i]] <- "undefined_minmax"
  }
}
profile_rows <- records[c("case_id", "organism", "model")]
profile_rows$status <- statuses
profile_rows$positions <- vapply(profiles, length, integer(1))
profile_rows$minmax <- vapply(profiles, function(x) paste(x, collapse = ","), character(1))
write.table(profile_rows, args[[3]], sep = "\t", row.names = FALSE, quote = TRUE)

pair_rows <- list()
for (case_id in unique(records$case_id)) {
  indices <- which(records$case_id == case_id)
  if (length(unique(records$organism[indices])) != 1) stop("Case spans multiple hosts")
  for (i in indices) for (j in indices) {
    row <- data.frame(case_id = case_id, organism = records$organism[[i]],
                      model1 = records$model[[i]], model2 = records$model[[j]],
                      status = "unscored_input", distance = NA_real_, normalized_distance = NA_real_)
    if (statuses[[i]] == "scored" && statuses[[j]] == "scored") {
      result <- dtw(profiles[[i]], profiles[[j]])
      row$status <- "scored"
      row$distance <- result$distance
      row$normalized_distance <- result$normalizedDistance
      if (!all(is.finite(c(row$distance, row$normalized_distance)))) stop("Non-finite DTW result")
    }
    pair_rows[[length(pair_rows) + 1]] <- row
  }
}
write.table(do.call(rbind, pair_rows), args[[4]], sep = "\t", row.names = FALSE, quote = TRUE)
cat("AUTHOR_MINMAX_DTW_COMPLETE", nrow(records), "profiles", length(pair_rows), "pairs\n")
