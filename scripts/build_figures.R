script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
paper_dir <- dirname(script_dir)
source(file.path(script_dir, "figure_style.R"))
# Keep every grid measurement on an Arial-capable Cairo device, including
# legends constructed between saves. The default PostScript device lacks it.
measurement_file <- tempfile(fileext = ".pdf")
cairo_pdf(measurement_file, width = 8.4, height = 8.3, family = "Arial")
read_record <- function(path) jsonlite::fromJSON(file.path(paper_dir, path), simplifyVector = FALSE)
calibration <- read_record("evidence/measurement-inventory/calibration.json")
inventory <- read_record("evidence/measurement-inventory/rescore_summary.json")
null_record <- read_record("evidence/measurement-inventory/null_v2.json")
sensitivity <- read_record("evidence/scale-correspondence/results.json")
colours <- c("#285B8A", "#C6792B", "#6D548C", "#477B4B")
output_path <- function(name) file.path(paper_dir, "figures", paste0(name, ".pdf"))
coordinates <- read.csv(file.path(script_dir, "plot-inputs/coordinates.csv"))
coordinates$branch <- factor(coordinates$branch, levels = 0:2,
                             labels = c("Trunk", "Upper arm", "Lower arm"))
# At least two labelled ticks inside each displayed window, using the fewest
# pretty values; n.breaks alone can leave a lone zero tick on narrow arms.
window_breaks <- function(limits) {
  for (n in 3:8) {
    values <- pretty(limits, n = n)
    values <- values[values >= limits[1] - 1e-9 & values <= limits[2] + 1e-9]
    if (length(values) >= 2) return(values)
  }
  values
}
candidates <- c("faithful", "shatter_0.5", "shatter_2.5", "collapse_0", "noise_0.18", "permuted")
titles <- c("Faithful + jitter", "Arm shift 0.5", "Arm shift 2.5",
            "Fork projection", "Noise 0.18", "Row permutation")
plots <- lapply(seq_along(candidates), function(index) {
  selected <- subset(coordinates, candidate == candidates[index])
  horizontal <- range(selected$coordinate_1)
  vertical <- range(selected$coordinate_2)
  half_extent <- max(diff(horizontal), diff(vertical)) * 0.55
  ggplot(selected,
         aes(coordinate_1, coordinate_2, colour = branch)) +
    geom_point(size = 0.8, alpha = 0.8) +
    coord_equal(expand = FALSE) +
    scale_x_continuous(limits = mean(horizontal) + c(-half_extent, half_extent), breaks = window_breaks) +
    scale_y_continuous(limits = mean(vertical) + c(-half_extent, half_extent), breaks = window_breaks) +
    scale_colour_manual(values = colours[1:3]) +
    labs(title = titles[index], x = "Coordinate 1", y = "Coordinate 2", colour = NULL) +
    guides(colour = guide_legend(nrow = 1, override.aes = list(size = 1.5, alpha = 1)))
})
save_panels(plots, output_path("calibration_perturbations"), ncol = 3, width = 6.8, height = 4.7,
            shared_legend = TRUE, base_size = 11)

variant_names <- c("sharp", "truth", "diffuse", "near_uniform", "shuffled", "constant")
variant_labels <- c("Sharp", "Smooth", "Diffuse", "Near-\nuniform", "Shuffled", "Constant")
allocation_data <- do.call(rbind, lapply(calibration$allocation_rows, function(row) {
  data.frame(seed = paste("Seed", row$seed), variant = row$variant,
             coherence = row$v2$allocation_coherence_raw,
             perplexity = row$v2$allocation_perplexity_fraction,
             pair_js = row$v2$allocation_random_js)
}))
allocation_data$variant <- factor(allocation_data$variant, levels = variant_names, labels = variant_labels)
allocation_data$seed <- factor(allocation_data$seed,
                               levels = paste("Seed", sort(unique(as.integer(sub("Seed ", "", allocation_data$seed))))))
plots <- lapply(c("coherence", "perplexity", "pair_js"), function(metric) {
  panel <- ggplot(allocation_data, aes(variant, .data[[metric]], colour = seed, group = seed)) +
    geom_point(size = 2.8, position = position_dodge(width = 0.35)) +
    scale_colour_manual(values = colours[1:3]) +
    scale_x_discrete(labels = function(x) gsub("-\n", "-", x, fixed = TRUE),
                     guide = guide_axis(angle = 30)) +
    labs(title = c(coherence = "Raw allocation coherence", perplexity = "Allocation dispersion",
                   pair_js = "Absolute allocation variation")[[metric]],
         x = NULL, y = c(coherence = "Raw coherence", perplexity = "Perplexity / K",
                        pair_js = "All-pair JSD (nats)")[[metric]],
         colour = NULL)
  panel <- panel + if (metric == "pair_js") {
    scale_y_sqrt(limits = c(0, 0.7), breaks = c(0, 0.02, 0.1, 0.3, 0.6))
  } else scale_y_continuous(limits = c(-0.03, 1.08))
  if (metric == "perplexity") panel <- panel +
    geom_hline(yintercept = 0.95, linetype = "dashed", colour = "#777777") +
    annotate("text", x = 6, y = 1.0, label = "0.95 guard", family = "Arial",
             size = 3, colour = "black", hjust = 1, vjust = 0)
  panel
})

# This separate, already-validated numerical fixture is not a v2 inventory
# remeasurement. Keep its identity and eligibility rule explicit in the caption.
tail_record <- read_record("evidence/raag-v3/full-20261009-c/report.json")
stopifnot(tail_record$status == "PASS_P1_FIXTURES",
          tail_record$unit_suite$status == "PASS",
          tail_record$unit_suite$skipped == 0,
          all(vapply(tail_record$rows, function(row) row$check_status == "PASS", logical(1))))
tail_values <- do.call(rbind, lapply(Filter(function(row) {
  row$control %in% c("smooth", "shuffled") && row$actual$allocation_status == "eligible"
}, tail_record$rows), function(row) {
  data.frame(control = row$control, pair_js = row$actual$allocation_pair_js,
             coherence = row$actual$allocation_coherence)
}))
tail_panel <- ggplot(tail_values, aes(pair_js, coherence, group = control)) +
  geom_vline(xintercept = 1e-12, linetype = "dashed", colour = "#777777") +
  geom_line(colour = "#536574", linewidth = 0.7) +
  geom_point(colour = "#536574", size = 1.6) +
  annotate("text", x = 1e-6, y = 1.18, label = "Smooth", family = "Arial", size = 3.1) +
  annotate("text", x = 1e-6, y = 0.20, label = "Reassigned", family = "Arial", size = 3.1) +
  scale_x_log10(limits = c(5e-13, 0.1), breaks = c(1e-12, 1e-8, 1e-4),
                labels = c(expression(10^-12), expression(10^-8), expression(10^-4))) +
  # Reserve room above both curves so the regular-Arial labels clear markers.
  scale_y_continuous(limits = c(-0.03, 1.25), breaks = c(0, 0.3, 0.6, 0.9)) +
  labs(title = "Relative score vs absolute variation", x = "All-pair JSD (nats)",
       y = "Eligible coherence")
plots <- append(plots, list(tail_panel))
save_panels(plots, output_path("allocation_validity_guards"), width = 6.8, height = 4.4, shared_legend = TRUE)

statuses <- inventory$file_allocation_status_counts
inventory_data <- data.frame(category = c("Valid native", "Guarded native", "Non-native"),
                             count = c(statuses$valid, statuses$near_uniform + statuses$degenerate, statuses$non_native))
unique_statuses <- inventory$unique_allocation_status_counts
unique_data <- data.frame(category = c("Valid native", "Guarded native", "Non-native"),
                          count = c(unique_statuses$valid,
                                    unique_statuses$near_uniform + unique_statuses$degenerate,
                                    unique_statuses$non_native))
null_data <- data.frame(category = c("Eligible\nnative", "Coherence\n> null max.", "Joint\n> null max."),
                        count = c(null_record$n_quotable_native, null_record$n_native_coherence_above_max,
                                  null_record$n_native_joint_edge_separation_coherence_above_max))
plots <- lapply(seq_along(list(inventory_data, unique_data, null_data)), function(index) {
  values <- list(inventory_data, unique_data, null_data)[[index]]
  values$category <- factor(values$category, levels = values$category)
  ggplot(values, aes(category, count, fill = category)) + geom_col(width = 0.65, show.legend = FALSE) +
    geom_text(aes(label = count), vjust = -0.35, family = "Arial", size = 3.5, colour = "black") +
    scale_fill_manual(values = c(colours[1], colours[2], "#9AA1A8")) +
    scale_x_discrete(labels = function(label) gsub(" native", "\nnative", label, fixed = TRUE)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.16))) +
    labs(title = c("Saved-file applicability", "Distinct-object applicability",
                   "Ten-permutation counts")[index], x = NULL,
         y = c("Saved files", "Distinct objects", "Focused runs")[index])
})
null_margins <- sort(vapply(Filter(function(row) row$native && row$allocation_status == "valid",
                                    null_record$rows), function(row) {
  row$observed$allocation_coherence - max(vapply(row$permutations,
                               function(draw) draw$allocation_coherence_raw, numeric(1)))
}, numeric(1)))
stopifnot(length(null_margins) == null_record$n_quotable_native,
          sum(null_margins > 0) == null_record$n_native_coherence_above_max)
margin_values <- data.frame(rank = seq_along(null_margins), margin = null_margins)
plots <- append(plots, list(
  ggplot(margin_values, aes(rank, margin)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "#777777") +
    geom_point(aes(colour = margin > 0), size = 1.9, show.legend = FALSE) +
    scale_colour_manual(values = c("FALSE" = colours[2], "TRUE" = colours[1])) +
    scale_x_continuous(breaks = c(1, 20, 40, 60)) +
    labs(title = "Native-run null margins", x = "Runs ordered by margin",
         y = "Coherence margin")
))
save_panels(plots, output_path("inventory_null_comparison"), height = 4.3, width = 6.8, base_size = 11)

datasets <- c("setty", "dentate", "endo", "lung")
dataset_labels <- c(setty = "Blood", dentate = "Dentate", endo = "Pancreas", lung = "Lung")
metric_names <- c("reachability_survival", "edge_survival", "separation_survival", "allocation_coherence")
metric_labels <- c("Reachability", "Edge survival", "Separation", "Native coherence")
settings <- expand.grid(rank = c(10, 25, 50), neighbour_count = c(5, 15, 30))
settings$label <- paste(settings$neighbour_count, settings$rank, sep = "/")
plots <- lapply(seq_along(metric_names), function(index) {
  metric <- metric_names[index]
  groups <- expand.grid(native = if (index == 4) TRUE else c(TRUE, FALSE), dataset = datasets)
  values <- do.call(rbind, lapply(seq_len(nrow(groups)), function(group_index) {
    dataset <- groups$dataset[group_index]
    native <- groups$native[group_index]
    selected <- Filter(function(row) row$dataset == dataset && row$native == native, sensitivity$scale)
    {
      baseline <- mean(vapply(Filter(function(row) row$k == 15 && row$rank == 50, selected),
                              function(row) row[[metric]], numeric(1)))
      changes <- vapply(seq_len(nrow(settings)), function(setting_index) {
        rows <- Filter(function(row) row$k == settings$neighbour_count[setting_index] &&
                         row$rank == settings$rank[setting_index], selected)
        measured <- mean(vapply(rows, function(row) row[[metric]], numeric(1)))
        if (index == 1) measured else measured - baseline
      }, numeric(1))
    }
    data.frame(group = if (index == 4) unname(dataset_labels[dataset]) else
                 paste(dataset_labels[dataset], if (native) "P" else "Z", sep = " / "),
               setting = settings$label, change = changes)
  }))
  values$group <- factor(values$group, levels = rev(unique(values$group)))
  values$setting <- factor(values$setting, levels = settings$label)
  colour_limit <- max(abs(values$change), 0.01, na.rm = TRUE)
  panel <- ggplot(values, aes(setting, group, fill = change)) +
    geom_tile(colour = "#D0D0D0", linewidth = 0.35) +
    geom_text(aes(label = ifelse(is.na(change), "NA", ifelse(abs(change) < 0.005, "0",
                    sub("^(-?)0\\.", "\\1.", sprintf("%.2f", change)))),
                  colour = ifelse(!is.na(change) & abs(change) > colour_limit * 0.65,
                                  "white", "#18232D")),
              family = "Arial", size = 3.15, show.legend = FALSE) +
    scale_colour_identity() +
    geom_vline(xintercept = c(3.5, 6.5), colour = "black", linewidth = 0.3) +
    scale_x_discrete(labels = function(labels) {
      result <- sub("/", "\n", labels)
      result[labels == "15/50"] <- "15\n50*"
      result
    }) +
    labs(title = c("Reachability (absolute)", "Edge survival", "Separation", "Native coherence (P only)")[index],
         x = "Graph k / reference d",
         y = NULL,
         fill = if (index == 1) "Absolute score" else "Change") +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5,
                                  barwidth = unit(1.9, "in"), barheight = unit(0.14, "in"),
                                  theme = theme(legend.title = element_text(size = 10, margin = margin(b = 2)),
                                                legend.text = element_text(size = 9.5))))
  panel + if (index == 1) {
    scale_fill_gradient(low = "#F0F5FA", high = "#164D79", limits = c(0, 1),
                         breaks = c(0, 0.5, 1))
  } else {
    scale_fill_gradient2(low = "#3979AD", mid = "#F6F6F6", high = "#B94040",
                         midpoint = 0, limits = c(-colour_limit, colour_limit),
                         breaks = c(-colour_limit, 0, colour_limit),
                         labels = function(values) formatC(values, format = "f", digits = 3))
  }
})
save_panels(plots, output_path("scale_reference_sensitivity"), height = 5.7, width = 7.2, base_size = 11)

plots <- lapply(seq_along(metric_names), function(index) {
  metric <- metric_names[index]
  values <- do.call(rbind, lapply(sensitivity$correspondence, function(row) {
    data.frame(dataset = dataset_labels[row$dataset], model_seed = row$model_seed,
               fraction = row$requested_fraction * 100, score = row[[metric]])
  }))
  seed_means <- aggregate(score ~ dataset + model_seed + fraction, values, mean)
  means <- aggregate(score ~ dataset + fraction, values, mean)
  if (index == 1) {
    stopifnot(max(abs(means$score - 1)) < 1e-12)
    common <- unique(means[c("fraction", "score")])
    means$offset <- (match(means$dataset, dataset_labels[datasets]) - 2.5) * 1.6
    means$fraction_plot <- means$fraction + means$offset
    ggplot() +
      geom_line(data = common, aes(fraction, score), linewidth = 1.05,
                colour = "#536574") +
      geom_point(data = means, aes(fraction_plot, score, colour = dataset), size = 2.4) +
      annotate("text", x = 50, y = 0.80, label = "All backgrounds = 1.00",
               family = "Arial", size = 3.1, colour = "#172532") +
      scale_colour_manual(values = setNames(colours, dataset_labels[datasets])) +
      scale_y_continuous(limits = c(-0.03, 1.04)) +
      scale_x_continuous(breaks = c(0, 10, 25, 50, 100)) +
      labs(title = "Reachability (ceiling)", x = "Reassigned rows (%)", y = "Score", colour = NULL) +
      guides(colour = guide_legend(nrow = 1))
  } else {
    ggplot(means, aes(fraction, score, colour = dataset, group = dataset)) +
      geom_line(data = seed_means, aes(group = interaction(dataset, model_seed)), linewidth = 0.55, alpha = 0.55) +
      geom_line(linewidth = 1.05) + geom_point(size = 2.4) +
      scale_colour_manual(values = setNames(colours, dataset_labels[datasets])) +
      scale_y_continuous(limits = c(-0.03, 1.04)) +
      scale_x_continuous(breaks = c(0, 10, 25, 50, 100)) +
      labs(title = metric_labels[index], x = "Reassigned rows (%)", y = "Score", colour = NULL) +
      guides(colour = guide_legend(nrow = 1))
  }
})
save_panels(plots, output_path("row_correspondence_sensitivity"), width = 6.8, height = 3.9, shared_legend = TRUE)

diversity <- read_record("evidence/diversity-20261009/results.json")
method_levels <- c("pca", "scvi", "umap", "random", "permuted_pca")
method_labels <- c(pca = "PCA", scvi = "scVI", umap = "UMAP",
                   random = "Random", permuted_pca = "Permuted")
object_levels <- c("bulb", "pfc_151673", "pfc_151507", "lymph", "breast", "ovary",
                   "metastasis", "pbmc3k")
object_labels <- c(bulb = "Olfactory bulb", pfc_151673 = "PFC-673", pfc_151507 = "PFC-507",
                   lymph = "Lymph node", breast = "Breast cancer", ovary = "Ovarian cancer",
                   metastasis = "Liver metastasis", pbmc3k = "PBMC")
views <- read.csv(file.path(paper_dir, "evidence/diversity-20261009/object_views.csv"))
stopifnot(all(c("object", "view", "x", "y", "pc1") %in% names(views)),
          all(is.finite(as.matrix(views[c("x", "y", "pc1")]))))
object_record <- function(name) {
  selected <- Filter(function(object) object$object == name, diversity$objects)
  stopifnot(length(selected) == 1)
  selected[[1]]
}
plot_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  indices <- which(grepl("^guide-box", grob$layout$name))
  indices <- indices[!vapply(grob$grobs[indices], inherits, logical(1), "zeroGrob")]
  stopifnot(length(indices) == 1)
  grob$grobs[[indices]]
}
normal_heading <- function(text, size = 11) {
  textGrob(text, x = unit(0.025, "in"), just = "left",
           gp = gpar(fontfamily = "Arial", fontface = "plain", fontsize = size, col = "black"))
}
save_figure <- function(figure, name, height, width = 7.2) {
  ggsave(output_path(name), figure, device = cairo_pdf, width = width, height = height,
         units = "in", bg = "white", family = "Arial")
}
colour_limits <- function(object_name) range(views$pc1[views$object == object_name])
object_point <- function(object_name, view_name, title, show_legend = FALSE) {
  frame <- subset(views, object == object_name & view == view_name)
  stopifnot(nrow(frame) > 0)
  x_range <- range(frame$x)
  y_range <- range(frame$y)
  half_extent <- max(diff(x_range), diff(y_range)) * 0.55
  axis_labels <- switch(view_name, tissue = c("Tissue x", "Tissue y"),
                        pca = c("PC 1", "PC 2"), umap = c("UMAP 1", "UMAP 2"),
                        scvi = c("Latent 1", "Latent 2"))
  tissue_scale <- if (view_name == "tissue" && max(abs(c(x_range, y_range))) >= 1000) 1000 else 1
  if (tissue_scale == 1000) axis_labels <- paste0(axis_labels, " (10³)")
  ggplot(frame, aes(x, y, colour = pc1)) +
    geom_point(size = if (nrow(frame) < 800) 0.9 else 0.42, alpha = 0.9, stroke = 0) +
    coord_equal(expand = FALSE) +
    scale_colour_gradient2(low = "#2468A2", mid = "#E9E6DF", high = "#B33D38",
                           midpoint = 0, limits = colour_limits(object_name),
                           name = "Expression PC 1",
                           guide = if (show_legend) "colourbar" else "none") +
    scale_x_continuous(limits = mean(x_range) + c(-half_extent, half_extent), breaks = window_breaks,
                       labels = function(x) format(x / tissue_scale, trim = TRUE, scientific = FALSE)) +
    scale_y_continuous(limits = mean(y_range) + c(-half_extent, half_extent), breaks = window_breaks,
                       labels = function(x) format(x / tissue_scale, trim = TRUE, scientific = FALSE)) +
    labs(title = title, x = axis_labels[1], y = axis_labels[2]) +
    paper_theme(11) +
    theme(axis.text = element_text(size = 9.5, face = "plain"),
          axis.title = element_text(size = 10, face = "plain"),
          plot.title = element_text(size = 11, face = "plain", margin = margin(b = 3)),
          plot.margin = margin(3, 2, 1, 3)) +
    guides(colour = if (show_legend) guide_colourbar(title.position = "left",
                       barwidth = unit(1.55, "in"), barheight = unit(0.1, "in")) else "none")
}
heatmap_frame <- function(reference_name) {
  do.call(rbind, lapply(object_levels, function(name) {
    scores <- object_record(name)[[reference_name]]
    if (is.null(scores)) {
      stopifnot(reference_name == "spatial_reference", name == "pbmc3k")
      return(NULL)
    }
    data.frame(object = name, method = method_levels,
                edge = vapply(method_levels, function(method) scores[[method]]$edge_survival, numeric(1)))
  }))
}
score_heatmap <- function(frame, title) {
  stopifnot(all(is.finite(frame$edge)))
  present_rows <- unname(object_labels[object_levels[object_levels %in% frame$object]])
  has_pbmc <- "pbmc3k" %in% frame$object
  frame$object <- factor(frame$object, levels = rev(object_levels), labels = rev(object_labels[object_levels]))
  frame$method <- factor(frame$method, levels = method_levels, labels = method_labels[method_levels])
  panel <- ggplot(frame, aes(method, object, fill = edge)) +
    geom_tile(colour = "white", linewidth = 0.65) +
    geom_text(aes(label = sprintf("%.2f", edge),
                  colour = ifelse(edge > 0.58, "white", "#172532")),
              family = "Arial", size = 3.3, show.legend = FALSE) +
    scale_colour_identity() +
    scale_fill_gradient(low = "#F0F5FA", high = "#164D79", limits = c(0, 1),
                         breaks = c(0, 0.25, 0.5, 0.75, 1), na.value = "#DADDE1") +
    scale_x_discrete(expand = expansion(add = 0),
                      labels = c("PCA", "scVI", "UMAP", "RP", "Perm.")) +
    scale_y_discrete(limits = rev(unname(object_labels[object_levels])),
                     breaks = rev(present_rows), expand = expansion(add = 0)) +
    coord_cartesian(ylim = c(0.5, length(object_levels) + 0.5), expand = FALSE) +
    labs(title = title, x = NULL, y = NULL, fill = "Edge survival") +
    paper_theme(11) +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text = element_text(size = 9.5, face = "plain"),
          plot.title = element_text(size = 12, face = "plain"), plot.title.position = "plot") +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5,
                                  barwidth = unit(2.4, "in"), barheight = unit(0.12, "in"),
                                  theme = theme(legend.title = element_text(margin = margin(b = 3)))))
  if (!has_pbmc) panel <- panel +
    annotate("text", x = 3, y = 1, label = "PBMC has no tissue positions", family = "Arial",
             size = 2.8, colour = "#536574")
  panel
}

# Figure 8: six equally sized tissue frames above matched-row score matrices.
# The unexported PFC-507 coordinate view is not invented; its saved scores remain.
map_names <- c("bulb", "pfc_151673", "lymph", "breast", "ovary", "metastasis")
map_plots <- lapply(map_names, function(name) {
  n <- sum(views$object == name & views$view == "tissue")
  object_point(name, "tissue", paste0(object_labels[name], "\nn = ", format(n, big.mark = ",")))
})
map_panels <- align_panels(lapply(map_plots, ggplotGrob))
map_panels <- lapply(seq_along(map_panels), function(i) label_panel(map_panels[[i]], letters[i]))
map_grid <- arrange_compact(map_panels, ncol = 3, square_row_height = 2.40)
heat_plots <- list(score_heatmap(heatmap_frame("spatial_reference"), "Tissue-position reference"),
                   score_heatmap(heatmap_frame("expression_reference"), "Expression-PCA reference"))
heat_legend <- plot_legend(heat_plots[[1]])
heat_panels <- align_panels(lapply(heat_plots, function(plot) ggplotGrob(plot + theme(legend.position = "none"))))
heat_panels <- lapply(seq_along(heat_panels), function(i) label_panel(heat_panels[[i]], letters[i + 6], anchor = "axis"))
heat_grid <- gridExtra::arrangeGrob(
  nullGrob(), arrange_compact(heat_panels, ncol = 2), nullGrob(),
  ncol = 3, widths = unit(c(0.45, 6.30, 0.45), "in"), padding = unit(0, "pt")
)
section_figure <- gridExtra::arrangeGrob(
  map_grid,
  normal_heading("PC 1 colour: separate scale within each tissue section.", 10),
  heat_grid,
  heat_legend,
  # Reserve enough room for the colourbar title and tick labels; otherwise
  # the lower ticks are clipped in the exported PDF.
  ncol = 1, heights = unit(c(4.80, 0.18, 2.10, 0.61), "in")
)
save_figure(section_figure, "spatial_reference", height = 7.69)

# Figure 9: every coordinate view is labelled; saved records give the numerical
# comparison separately from the exported visualization/refit coordinates.
embedding_plots <- unlist(lapply(c("breast", "bulb"), function(name) {
  lapply(c("tissue", "pca", "umap", "scvi"), function(view) {
    object_point(name, view, c(tissue = "Tissue position", pca = "PCA",
                             umap = "UMAP", scvi = "scVI")[[view]])
  })
}), recursive = FALSE)
embedding_panels <- align_panels(lapply(embedding_plots, ggplotGrob))
embedding_panels <- lapply(seq_along(embedding_panels), function(i) label_panel(embedding_panels[[i]], letters[i]))
geometry_heatmap <- function(name) {
  record <- object_record(name)
  metric_keys <- c("reachability_survival", "edge_survival", "separation_survival")
  values <- do.call(rbind, lapply(c("spatial_reference", "expression_reference"), function(reference) {
    do.call(rbind, lapply(seq_along(metric_keys), function(i) {
      data.frame(method = method_levels,
                  metric = paste(if (reference == "spatial_reference") "Tissue" else "Expr.", c("R", "E", "S")[i], sep = "\n"),
                  score = vapply(method_levels, function(method) record[[reference]][[method]][[metric_keys[i]]], numeric(1)))
    }))
  }))
  values$method <- factor(values$method, levels = rev(method_levels), labels = rev(method_labels[method_levels]))
  values$metric <- factor(values$metric, levels = c("Tissue\nR", "Tissue\nE", "Tissue\nS", "Expr.\nR", "Expr.\nE", "Expr.\nS"))
  ggplot(values, aes(metric, method, fill = score)) +
    geom_tile(colour = "white", linewidth = 0.65) +
    geom_text(aes(label = sprintf("%.2f", score), colour = ifelse(score > 0.58, "white", "#172532")),
              family = "Arial", size = 3.3, show.legend = FALSE) +
    geom_vline(xintercept = 3.5, linewidth = 0.7, colour = "white") +
    scale_colour_identity() +
    scale_fill_gradient(low = "#F0F5FA", high = "#164D79", limits = c(0, 1), breaks = c(0, 0.5, 1)) +
    scale_x_discrete(expand = expansion(add = 0)) +
    scale_y_discrete(expand = expansion(add = 0)) +
    labs(title = paste0(object_labels[name], ": saved geometry scores"), x = NULL, y = NULL,
         fill = "Score\n(1.00 = metric ceiling)") +
    paper_theme(11) +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text = element_text(size = 9.5, face = "plain"),
          plot.title = element_text(size = 11.5, face = "plain"), plot.title.position = "plot") +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5,
                                  barwidth = unit(2.1, "in"), barheight = unit(0.1, "in"),
                                  theme = theme(legend.title = element_text(margin = margin(b = 3)))))
}
geometry_plots <- lapply(c("breast", "bulb"), geometry_heatmap)
geometry_legend <- plot_legend(geometry_plots[[1]])
geometry_panels <- align_panels(lapply(geometry_plots, function(plot) ggplotGrob(plot + theme(legend.position = "none"))))
geometry_panels <- lapply(seq_along(geometry_panels), function(i) label_panel(geometry_panels[[i]], letters[i + 8], anchor = "axis"))
object_heading <- function(name) {
  n <- sum(views$object == name & views$view == "pca")
  normal_heading(paste0(object_labels[name], " | ", format(n, big.mark = ","), " spots"))
}
embedding_figure <- gridExtra::arrangeGrob(
  object_heading("breast"), arrange_compact(embedding_panels[1:4], ncol = 4, square_row_height = 2.04),
  plot_legend(object_point("breast", "pca", "PCA", show_legend = TRUE)),
  object_heading("bulb"), arrange_compact(embedding_panels[5:8], ncol = 4, square_row_height = 2.04),
  plot_legend(object_point("bulb", "pca", "PCA", show_legend = TRUE)),
  # Left allowance keeps the first heatmap letter inside the page.
  gridExtra::arrangeGrob(nullGrob(), arrange_compact(geometry_panels, ncol = 2), ncol = 2,
                         widths = unit(c(0.22, 6.98), "in"), padding = unit(0, "pt")),
  geometry_legend,
  # The final colourbar needs room for both its title and tick labels.
  ncol = 1, heights = unit(c(0.18, 2.04, 0.32, 0.18, 2.04, 0.32, 1.90, 0.68), "in")
)
save_figure(embedding_figure, "object_embeddings", height = 7.66)

# Figure 10: PBMC embeddings are paired with all three saved diagnostics, rather
# than asking a reader to interpret attractive coordinates as a quality score.
pbmc_plots <- lapply(c("pca", "umap", "scvi"), function(view) {
  object_point("pbmc3k", view, c(pca = "PCA", umap = "UMAP", scvi = "scVI")[[view]])
})
pbmc_panels <- align_panels(lapply(pbmc_plots, ggplotGrob))
pbmc_panels <- lapply(seq_along(pbmc_panels), function(i) label_panel(pbmc_panels[[i]], letters[i]))
pbmc_record <- object_record("pbmc3k")$expression_reference
pbmc_metrics <- c(reachability_survival = "Reachability", edge_survival = "Edge survival",
                  separation_survival = "Separation")
pbmc_score_plots <- lapply(names(pbmc_metrics), function(metric) {
  values <- data.frame(method = factor(method_levels, levels = rev(method_levels),
                                       labels = rev(method_labels[method_levels])),
                       score = vapply(method_levels, function(method) pbmc_record[[method]][[metric]], numeric(1)))
  ggplot(values, aes(score, method)) +
    geom_vline(xintercept = pbmc_record$permuted_pca[[metric]], linetype = "dashed", colour = "#858C94", linewidth = 0.55) +
    geom_segment(aes(x = 0, xend = score, yend = method), colour = "#D9E2EB", linewidth = 0.7) +
    geom_point(size = 2.8, colour = "#285B8A") +
    geom_text(aes(x = 1.31, label = sprintf("%.2f", score)), hjust = 1,
              family = "Arial", size = 3.1, colour = "#172532") +
    scale_x_continuous(limits = c(-0.03, 1.35), breaks = c(0, 0.5, 1), expand = expansion(mult = 0)) +
    scale_y_discrete(expand = expansion(add = 0.65)) +
    labs(title = pbmc_metrics[[metric]], x = "Score", y = NULL) + paper_theme(11) +
    theme(axis.text = element_text(size = 9.5, face = "plain"),
          plot.title = element_text(size = 11.5, face = "plain"))
})
pbmc_score_panels <- align_panels(lapply(pbmc_score_plots, ggplotGrob))
pbmc_score_panels <- lapply(seq_along(pbmc_score_panels), function(i) label_panel(pbmc_score_panels[[i]], letters[i + 3]))
pbmc_figure <- gridExtra::arrangeGrob(
  normal_heading(paste0("PBMC | ", format(sum(views$object == "pbmc3k" & views$view == "pca"), big.mark = ","), " cells | expression-PCA reference")),
  arrange_compact(pbmc_panels, ncol = 3, square_row_height = 2.42),
  plot_legend(object_point("pbmc3k", "pca", "PCA", show_legend = TRUE)),
  arrange_compact(pbmc_score_panels, ncol = 3),
  normal_heading("Dashed line: single permutation control, not an uncertainty interval.", 10),
  ncol = 1, heights = unit(c(0.20, 2.42, 0.32, 2.10, 0.23), "in")
)
save_figure(pbmc_figure, "object_pbmc", height = 5.27)

status <- system2("python3", file.path(script_dir, "build_architecture.py"))
stopifnot(status == 0)
status <- system2(Sys.getenv("SVG_PYTHON", "/usr/bin/python3"), c(file.path(script_dir, "render_schematic.py"),
                   file.path(paper_dir, "figures/architecture.svg"),
                   file.path(paper_dir, "figures/architecture.pdf")))
stopifnot(status == 0)
dev.off()
unlink(measurement_file)
cat("Rendered nine data figures and the editable vector measurement workflow from existing records.\n")
