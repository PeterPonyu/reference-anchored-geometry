script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
paper_dir <- dirname(script_dir)
source(file.path(script_dir, "figure_style.R"))
# A dropped data row must stop figure generation instead of disappearing silently.
options(warn = 2)
# Calibration, chance-correction and coherence-baseline figures. Every value is
# read from evidence/followup-20261010/results; nothing is refitted here.
measurement_file <- tempfile(fileext = ".pdf")
cairo_pdf(measurement_file, width = 8.4, height = 8.3, family = "Arial")
results <- file.path(paper_dir, "evidence/followup-20261010/results")
read <- function(name) read.csv(file.path(results, name), stringsAsFactors = FALSE)
output_path <- function(name) file.path(paper_dir, "figures", paste0(name, ".pdf"))
colours <- c("#285B8A", "#C6792B", "#6D548C", "#477B4B")
grey <- "#7A8590"
metric_colours <- c("Reachability R" = "#285B8A", "Distance Spearman" = "#C6792B", "Q global" = "#6D548C",
                    "AUC-RNX" = "#477B4B", "Edge survival E*" = "#9A9FA5", "Far-vs-edge AUROC" = "#B8BCC1",
                    "Trustworthiness" = "#9A9FA5", "LCMC" = "#9A9FA5",
                    "Cohesiveness" = "#9A9FA5", "Steadiness" = "#9A9FA5")
window_breaks <- function(limits) {
  for (n in 3:8) {
    values <- pretty(limits, n = n)
    values <- values[values >= limits[1] - 1e-9 & values <= limits[2] + 1e-9]
    if (length(values) >= 2) return(values)
  }
  values
}

# ---------------------------------------------------------------- calibration
identity <- read("a1_reachability_identity.csv")
identity$set <- factor(c(fixture = "Fixture", scale24 = "Scale objects", diversity = "Real objects")[identity$group],
                       levels = c("Fixture", "Scale objects", "Real objects"))
identity <- identity[is.finite(identity$R_archived_fn) & is.finite(identity$R_component_formula), ]
identity <- identity[tolower(as.character(identity$ref_connected)) == "true", ]
stopifnot(nrow(identity) == 322, max(identity$abs_diff) == 0)
p_identity <- ggplot(identity, aes(R_component_formula, R_archived_fn, colour = set, shape = set)) +
  geom_abline(slope = 1, intercept = 0, colour = grey, linewidth = 0.4, linetype = "dashed") +
  geom_point(size = 2, alpha = 0.85) +
  scale_colour_manual(values = colours[1:3]) + scale_shape_manual(values = c(16, 17, 15)) +
  scale_x_continuous(limits = c(0.3, 1.02), breaks = c(0.4, 0.6, 0.8, 1)) +
  scale_y_continuous(limits = c(0.3, 1.02), breaks = c(0.4, 0.6, 0.8, 1)) +
  labs(title = "Reachability is a candidate statistic", x = "Candidate component pair fraction",
       y = "Reachability R", colour = NULL, shape = NULL) +
  guides(colour = guide_legend(ncol = 1), shape = guide_legend(ncol = 1)) +
  theme(legend.position = "inside", legend.position.inside = c(0.78, 0.33))

fragmented <- read("a1_fragmented_reference_fixture.csv")
fragmented <- aggregate(cbind(R_archived, pair_precision, component_ari) ~ candidate, fragmented, mean)
candidate_labels <- c(faithful = "Faithful", merged_all = "All arms merged", merged_two = "One arm merged",
                      over_fragmented = "Trunk split", row_permuted = "Rows permuted")
fragmented$candidate <- factor(candidate_labels[fragmented$candidate], levels = unname(candidate_labels))
fragmented_long <- rbind(
  data.frame(candidate = fragmented$candidate, metric = "Reachability R", value = fragmented$R_archived),
  data.frame(candidate = fragmented$candidate, metric = "Pair precision", value = fragmented$pair_precision),
  data.frame(candidate = fragmented$candidate, metric = "Component ARI", value = fragmented$component_ari))
fragmented_long$metric <- factor(fragmented_long$metric, levels = c("Reachability R", "Pair precision", "Component ARI"))
fragmented_long$candidate <- factor(fragmented_long$candidate, levels = unname(candidate_labels))
fragment_axis <- c(Faithful = "Faithful", "All arms merged" = "All arms\nmerged",
                   "One arm merged" = "One arm\nmerged", "Trunk split" = "Trunk\nsplit",
                   "Rows permuted" = "Rows\npermuted")
p_fragmented <- ggplot(fragmented_long, aes(candidate, value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.82), width = 0.76) +
  scale_fill_manual(values = c(colours[1], colours[2], "#9AA1A8")) +
  scale_y_continuous(limits = c(0, 1.08), breaks = c(0, 0.5, 1), expand = c(0, 0)) +
  scale_x_discrete(labels = fragment_axis) +
  labs(title = "A fragmented reference", x = NULL, y = "Score (three seeds)", fill = NULL) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  theme(axis.text.x = element_text(lineheight = 0.88))

graded <- read("a3_graded_raw.csv")
response_curve <- function(topology, n_key, k, metrics) {
  rows <- graded[graded$topology == topology & graded$n_key == n_key & graded$k == k, ]
  out <- do.call(rbind, lapply(names(metrics), function(label) {
    column <- metrics[[label]]
    by_seed <- split(rows, rows$seed)
    per_seed <- do.call(rbind, lapply(by_seed, function(r) {
      base <- r[[column]][r$amplitude == 0]
      full <- r[[column]][r$amplitude == max(r$amplitude)]
      data.frame(amplitude = r$amplitude, change = r[[column]] - base, full = full - base)
    }))
    mean_change <- aggregate(change ~ amplitude, per_seed, mean)
    seed_sd <- aggregate(change ~ amplitude, per_seed, sd)
    full <- mean(per_seed$full[per_seed$amplitude == max(per_seed$amplitude)])
    data.frame(metric = label, amplitude = mean_change$amplitude, change = mean_change$change,
               seed_sd = seed_sd$change, normalized = mean_change$change / full,
               normalized_sd = seed_sd$change / abs(full), full = full,
               full_sd = sd(per_seed$change[per_seed$amplitude == max(per_seed$amplitude)]))
  }))
  out[out$amplitude > 0, ]
}
graded_metrics <- list("Reachability R" = "R", "Distance Spearman" = "distance_spearman", "Q global" = "Q_global",
                       "AUC-RNX" = "AUC_RNX", "Edge survival E*" = "E_star", "Far-vs-edge AUROC" = "AUROC",
                       "Trustworthiness" = "trustworthiness", "LCMC" = "LCMC",
                       "Cohesiveness" = "cohesiveness", "Steadiness" = "steadiness")
curve <- response_curve("Y", 450, 15, graded_metrics)
curve$metric <- factor(curve$metric, levels = names(graded_metrics))
neighbour_scale <- read("a3_neighbour_scale.csv")
radius <- neighbour_scale$median_kth_nn_distance[neighbour_scale$topology == "Y" &
  neighbour_scale$n_key == 450 & neighbour_scale$k == 15]
stopifnot(length(radius) == 1, is.finite(radius))
p_graded <- ggplot(curve[curve$metric %in% names(graded_metrics)[1:4], ],
                   aes(amplitude, normalized, colour = metric)) +
  geom_vline(xintercept = radius, linetype = "dashed", colour = grey, linewidth = 0.4) +
  annotate("text", x = radius * 1.12, y = 1.14, label = "15-neighbour\nradius", hjust = 0, vjust = 1,
           lineheight = 0.9, family = "Arial", size = 2.9, colour = "black") +
  geom_ribbon(aes(ymin = normalized - normalized_sd, ymax = normalized + normalized_sd,
                  fill = metric), alpha = 0.12, colour = NA, show.legend = FALSE) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.4) +
  scale_colour_manual(values = metric_colours) + scale_fill_manual(values = metric_colours) +
  scale_x_log10(breaks = c(0.003, 0.03, 0.3, 3), labels = c("0.003", "0.03", "0.3", "3")) +
  scale_y_continuous(breaks = c(0, 0.5, 1)) + coord_cartesian(ylim = c(-0.1, 1.15)) +
  labs(title = "Graded arm separation", x = "Arm shift (log scale)",
       y = "Fraction of full response", colour = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE))

full_response <- unique(curve[, c("metric", "full", "full_sd")])
full_response$metric <- factor(full_response$metric, levels = rev(names(graded_metrics)))
p_full <- ggplot(full_response, aes(full, metric, fill = metric)) +
  geom_vline(xintercept = 0, colour = "black", linewidth = 0.4) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_errorbar(aes(xmin = full - full_sd, xmax = full + full_sd), orientation = "y",
                width = 0.24, linewidth = 0.4) +
  geom_text(aes(label = ifelse(abs(full) < 0.0005 & full != 0, sprintf("%.1e", full),
                              sprintf("%.3f", full)), hjust = ifelse(full < 0, 1.1, -0.1)),
            family = "Arial", size = 3, colour = "black") +
  scale_fill_manual(values = metric_colours) +
  scale_x_continuous(limits = c(-0.85, 0.12), breaks = c(-0.6, -0.3, 0)) +
  labs(title = "Full comparator response", x = "Paired change at shift 9", y = NULL) +
  theme(axis.text.y = element_text(size = 8))

merge_rows <- graded[graded$topology == "clusters" & graded$n_key == 450 & graded$k == 15, ]
merge_metrics <- c("Reachability R" = "R", "Pair precision" = "pair_precision", "Distance Spearman" = "distance_spearman")
merge_long <- do.call(rbind, lapply(names(merge_metrics), function(label) {
  m <- aggregate(merge_rows[[merge_metrics[[label]]]], list(amplitude = merge_rows$amplitude), mean)
  data.frame(metric = label, amplitude = m$amplitude, value = m$x)
}))
merge_long$metric <- factor(merge_long$metric, levels = names(merge_metrics))
p_merge <- ggplot(merge_long, aes(amplitude, value, colour = metric)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.4) +
  scale_colour_manual(values = c("Reachability R" = colours[1], "Pair precision" = "#9AA1A8",
                                 "Distance Spearman" = colours[2])) +
  scale_y_continuous(limits = c(0, 1.05), breaks = c(0, 0.5, 1)) +
  scale_x_continuous(breaks = c(0, 0.5, 1)) +
  labs(title = "Merging disjoint clusters", x = "Merge fraction", y = "Score", colour = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
  theme(legend.text = element_text(size = 8))

small_k <- graded[graded$topology == "Y" & graded$amplitude == 0, ]
small_k$size <- factor(ifelse(small_k$n_key == 450, "n = 450", "n = 2,000"), levels = c("n = 450", "n = 2,000"))
small_summary <- do.call(rbind, lapply(split(small_k, list(small_k$k, small_k$size), drop = TRUE), function(r)
  data.frame(k = factor(r$k[1], levels = c(5, 15, 30)), size = r$size[1], mean = mean(r$R), low = min(r$R), high = max(r$R))))
p_small <- ggplot(small_summary, aes(k, mean, colour = size, shape = size)) +
  geom_linerange(aes(ymin = low, ymax = high), position = position_dodge(width = 0.5), linewidth = 0.7) +
  geom_point(size = 2.4, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = colours[1:2]) + scale_shape_manual(values = c(16, 17)) +
  scale_y_continuous(limits = c(0, 1.05), breaks = c(0, 0.5, 1)) +
  labs(title = "Unshifted faithful candidate", x = "Neighbours k", y = "Reachability R (20 seeds)",
       colour = NULL, shape = NULL) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE), shape = guide_legend(nrow = 1, byrow = TRUE))

pack_panels(list(p_identity, p_fragmented, p_graded, p_full, p_merge, p_small),
            output_path("calibration_reachability"), ncol = 2, width = 6.5, base_size = 10.5,
            panel_height = c(0.98, 0.98, 1.02, 1.28, 0.98, 0.98),
            col_gap = 0.10, row_gap = 0.06)

# ---------------------------------------------------------- chance correction
scale24 <- read("a2_scale24.csv")
scale_d50 <- scale24[scale24$rank == 50, ]
floor_long <- rbind(data.frame(id = scale_d50$id, k = scale_d50$k, quantity = "Observed E", value = scale_d50$E),
                    data.frame(id = scale_d50$id, k = scale_d50$k, quantity = "Permutation floor E0", value = scale_d50$E0),
                    data.frame(id = scale_d50$id, k = scale_d50$k, quantity = "Corrected E*", value = scale_d50$E_star))
floor_long$quantity <- factor(floor_long$quantity, levels = c("Observed E", "Corrected E*", "Permutation floor E0"))
p_floor <- ggplot(floor_long, aes(k, value, colour = quantity, group = interaction(id, quantity))) +
  geom_line(linewidth = 0.35, alpha = 0.45) +
  stat_summary(aes(group = quantity), fun = median, geom = "line", linewidth = 1.1) +
  scale_colour_manual(values = c("Observed E" = colours[1], "Corrected E*" = colours[2], "Permutation floor E0" = grey)) +
  scale_x_continuous(breaks = c(5, 15, 30)) + scale_y_continuous(limits = c(0, 0.85), breaks = c(0, 0.4, 0.8)) +
  labs(title = "Edge survival and its floor", x = "Neighbours k", y = "Score", colour = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE))

ranges <- read("a2_scale24_ranges.csv")
ranges$type <- ifelse(ranges$native == "True" | ranges$native == TRUE, "Native allocation", "Continuous coordinates")
p_ranges <- ggplot(ranges, aes(E_range, Estar_range, colour = type, shape = type)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = grey, linewidth = 0.4) +
  geom_point(size = 2.2) +
  scale_colour_manual(values = colours[c(3, 1)]) + scale_shape_manual(values = c(16, 17)) +
  scale_x_continuous(limits = c(0, 0.45), breaks = c(0, 0.2, 0.4)) +
  scale_y_continuous(limits = c(0, 0.45), breaks = c(0, 0.2, 0.4)) +
  labs(title = "Range over nine settings", x = "Range of E", y = "Range of E*", colour = NULL, shape = NULL) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE), shape = guide_legend(nrow = 1, byrow = TRUE))

diversity <- read("a2_diversity.csv")
section_labels <- c(bulb = "Olfactory bulb", pfc_151673 = "PFC-673", pfc_151507 = "PFC-507", lymph = "Lymph node",
                    breast = "Breast cancer", ovary = "Ovarian cancer", metastasis = "Liver metastasis")
method_labels <- c(pca = "PCA", scvi = "scVI", umap = "UMAP", random = "Random projection")
tissue <- diversity[diversity$reference == "tissue" & diversity$candidate %in% names(method_labels), ]
tissue$section <- factor(section_labels[tissue$object], levels = rev(unname(section_labels)))
tissue$method <- factor(method_labels[tissue$candidate], levels = unname(method_labels))
method_colours <- setNames(colours, unname(method_labels))
permuted <- diversity[diversity$reference == "tissue" & diversity$candidate == "permuted_pca", ]
permuted$section <- factor(section_labels[permuted$object], levels = rev(unname(section_labels)))
p_raw <- ggplot(tissue, aes(E, section, colour = method, shape = method)) +
  geom_point(data = permuted, aes(E, section), inherit.aes = FALSE, shape = 124, size = 4, colour = "black") +
  geom_point(size = 2.2, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = method_colours) + scale_shape_manual(values = c(16, 17, 15, 18)) +
  scale_x_continuous(limits = c(0, 0.66), breaks = c(0, 0.3, 0.6)) +
  labs(title = "Raw tissue edge survival", x = "E (bar: permuted PCA)", y = NULL, colour = NULL, shape = NULL) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE), shape = guide_legend(nrow = 1, byrow = TRUE))
p_corrected <- ggplot(tissue, aes(E_star, section, colour = method, shape = method)) +
  geom_vline(xintercept = 0, colour = grey, linewidth = 0.4) +
  geom_point(size = 2.2, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = method_colours) + scale_shape_manual(values = c(16, 17, 15, 18)) +
  scale_x_continuous(limits = c(-0.02, 0.52), breaks = c(0, 0.25, 0.5)) +
  labs(title = "Against each method's own null", x = "Corrected E*", y = NULL, colour = NULL, shape = NULL) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE), shape = guide_legend(nrow = 1, byrow = TRUE))

refits <- read("a5_refit_scores.csv")
refits <- refits[is.finite(refits$layer_knn_purity), ]
refits$section <- factor(c(pfc_151673 = "PFC-673", pfc_151507 = "PFC-507")[refits$object], levels = c("PFC-673", "PFC-507"))
refit_methods <- c(pca = "PCA", scvi = "scVI", umap = "UMAP", random = "Random projection", permuted_pca = "Permuted PCA")
refits$method <- factor(refit_methods[refits$method], levels = unname(refit_methods))
p_layers <- ggplot(refits, aes(tissue_E_star, layer_knn_purity, colour = method, shape = section)) +
  geom_point(size = 2.1, alpha = 0.9) +
  scale_colour_manual(values = c(method_colours, "Permuted PCA" = grey)) +
  scale_shape_manual(values = c(16, 17)) +
  scale_x_continuous(breaks = window_breaks) + scale_y_continuous(breaks = window_breaks) +
  labs(title = "External layer anchor", x = "Tissue-reference E*", y = "Layer neighbour purity",
       colour = NULL, shape = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE, order = 1),
         shape = guide_legend(nrow = 1, byrow = TRUE, order = 2)) +
  theme(legend.box = "vertical")

difference <- read("a2_scale24_native_minus_continuous.csv")

star_column <- "dEstar"
if (length(star_column) == 0) stop("Native-minus-continuous E* column not found")
p_native <- ggplot(difference, aes(.data[[star_column[1]]])) +
  geom_vline(xintercept = 0, colour = "black", linewidth = 0.4) +
  geom_histogram(bins = 24, fill = colours[3], colour = "white", linewidth = 0.2) +
  scale_x_continuous(breaks = c(-0.3, -0.15, 0)) +
  labs(title = "Native minus continuous", x = "Difference in E*", y = "Settings")

pack_panels(list(p_floor, p_ranges, p_raw, p_corrected, p_layers, p_native),
            output_path("chance_correction"), ncol = 2, width = 6.5, base_size = 10.5,
            panel_height = c(0.98, 0.98, 1.16, 1.16, 1.00, 1.00),
            col_gap = 0.10, row_gap = 0.06, shared = list(c(3, 4)))

# ------------------------------------------------------------------ coherence
allocations <- read("a4_coherence_allocations.csv")
tempered <- read("a4b_tempered_gmm.csv")
family_labels <- c(native = "Native allocation", gmm_full10 = "GMM, raw", random_matched = "Random smooth")
allocations$family_label <- family_labels[allocations$family]
coherence <- aggregate(C_unclipped ~ id + family_label, allocations[!is.na(allocations$family_label), ], mean)
tempered_full <- tempered[tempered$gmm == "K10_full_10pc", ]
coherence <- rbind(coherence, data.frame(id = tempered_full$id, family_label = "GMM, sharpness-matched",
                                         C_unclipped = tempered_full$C_gmm_tempered))
background <- sub("-.*", "", coherence$id)
bg_levels <- c("Blood", "Dentate", "Pancreas", "Lung")
coherence$background <- factor(c(setty = "Blood", dentate = "Dentate", endo = "Pancreas", lung = "Lung")[background],
                               levels = bg_levels)
coherence$seed_n <- as.integer(sub(".*-s", "", coherence$id))
coherence$family_label <- factor(coherence$family_label,
  levels = c("Native allocation", "GMM, sharpness-matched", "GMM, raw", "Random smooth"))
# Seeds are repeated fits of one background. A vertical offset only separates
# the three stored seeds; coherence is the horizontal position. The offset of
# 0.16 category-units puts adjacent seeds about 5 pt apart in this panel.
coherence$y <- (length(bg_levels) + 1 - as.integer(coherence$background)) + (coherence$seed_n - 1) * 0.16
p_coherence <- ggplot(coherence, aes(C_unclipped, y, colour = family_label, shape = family_label)) +
  geom_point(size = 1.7) +
  scale_colour_manual(values = c(colours[1], colours[2], "#9AA1A8", colours[3])) +
  scale_shape_manual(values = c(16, 17, 2, 15)) +
  scale_y_continuous(breaks = 4:1, labels = bg_levels, limits = c(0.55, 4.48), expand = c(0, 0)) +
  scale_x_continuous(limits = c(0.18, 1.02), breaks = c(0.2, 0.6, 1), expand = c(0, 0)) +
  labs(title = "Coherence against baselines", x = "Unclipped coherence C", y = NULL, colour = NULL, shape = NULL) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE), shape = guide_legend(nrow = 2, byrow = TRUE))

valid <- allocations[is.finite(allocations$one_minus_geary) & is.finite(allocations$C_unclipped), ]
valid$family_label[is.na(valid$family_label)] <- "Other baseline"
p_geary <- ggplot(valid, aes(one_minus_geary, C_unclipped)) +
  geom_abline(slope = 1, intercept = 0, colour = grey, linetype = "dashed", linewidth = 0.4) +
  geom_point(aes(colour = family == "native"), size = 1.4, alpha = 0.7, show.legend = FALSE) +
  scale_colour_manual(values = c("TRUE" = colours[1], "FALSE" = "#B8BCC1")) +
  scale_x_continuous(breaks = window_breaks) + scale_y_continuous(breaks = window_breaks) +
  labs(title = "Coherence and Geary's c", x = "1 - Geary's c", y = "Coherence C")

tempered_full$background <- c(setty = "Blood", dentate = "Dentate", endo = "Pancreas", lung = "Lung")[sub("-.*", "", tempered_full$id)]
p_margin <- ggplot(tempered_full, aes(background, C_native - C_gmm_tempered)) +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
  geom_point(size = 2.2, colour = colours[1], position = position_jitter(width = 0.08, height = 0, seed = 3)) +
  labs(title = "Margin over GMM", x = NULL, y = "Native minus\nmatched GMM")

null_effect <- read("a4_native_permutation_unclipped.csv")
null_effect$background <- factor(c(setty = "Blood", dentate = "Dentate", endo = "Pancreas", lung = "Lung")
                                [sub("-.*", "", null_effect$id)],
                                levels = c("Blood", "Dentate", "Pancreas", "Lung"))
p_null <- ggplot(null_effect, aes(background, z)) +
  geom_point(size = 2.2, colour = colours[1], position = position_jitter(width = 0.08, height = 0, seed = 3)) +
  scale_y_continuous(limits = c(0, 80), breaks = c(0, 40, 80)) +
  labs(title = "Permutation-null effect size", x = NULL, y = "Null-standardized\ncoherence z")

pack_panels(list(p_coherence, p_geary, p_margin, p_null), output_path("coherence_baselines"),
            ncol = 2, width = 6.5, base_size = 10.5,
            panel_height = c(1.55, 1.55, 1.42, 1.42),
            col_gap = 0.10, row_gap = 0.06)
dev.off()
unlink(measurement_file)
cat("Rendered calibration, chance-correction and coherence-baseline figures from the follow-up records.\n")
