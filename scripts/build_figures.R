script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
paper_dir <- dirname(script_dir)
source(file.path(script_dir, "figure_style.R"))
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
candidates <- c("faithful", "shatter_0.5", "shatter_2.5", "collapse_0", "noise_0.18", "permuted")
titles <- c("Faithful + jitter", "Shatter, shift 0.5", "Shatter, shift 2.5",
            "Fork projection", "Additional noise 0.18", "Row permutation")
plots <- lapply(seq_along(candidates), function(index) {
  selected <- subset(coordinates, candidate == candidates[index])
  horizontal <- range(selected$coordinate_1)
  vertical <- range(selected$coordinate_2)
  half_extent <- max(diff(horizontal), diff(vertical)) * 0.55
  ggplot(selected,
         aes(coordinate_1, coordinate_2, colour = branch)) +
    geom_point(size = 0.65, alpha = 0.8) +
    coord_equal(xlim = mean(horizontal) + c(-half_extent, half_extent),
                ylim = mean(vertical) + c(-half_extent, half_extent), expand = FALSE) +
    scale_x_continuous(n.breaks = 3) + scale_y_continuous(n.breaks = 3) +
    scale_colour_manual(values = colours[1:3]) +
    labs(title = titles[index], x = "Coordinate 1", y = "Coordinate 2", colour = NULL) +
    guides(colour = guide_legend(nrow = 1, override.aes = list(size = 1.5, alpha = 1)))
})
save_panels(plots, output_path("calibration_perturbations"), ncol = 3, height = 4.35,
            shared_legend = TRUE, row_tags = TRUE)

metric_value <- function(seed, candidate, stack, metric) {
  selected <- Filter(function(row) row$seed == seed && row$candidate == candidate, calibration$rows)
  stopifnot(length(selected) == 1)
  selected[[1]]$stacks[[stack]]$values[[metric]]
}
metric_specs <- list(c("audit", "audit_trustworthiness", "Trustworthiness"),
                     c("zadu", "zadu_lcmc", "LCMC"),
                     c("raag", "reachability_survival", "Reachability"),
                     c("raag", "edge_survival", "Edge survival"))
plots <- lapply(metric_specs, function(specification) {
  values <- do.call(rbind, lapply(unlist(calibration$seeds), function(seed) {
    baseline <- metric_value(seed, "faithful", specification[1], specification[2])
    amplitude <- c(0, 0.5, 2.5, 9)
    measured <- vapply(c("faithful", "shatter_0.5", "shatter_2.5", "shatter_9"),
                       function(candidate) metric_value(seed, candidate, specification[1], specification[2]), numeric(1))
    data.frame(seed = paste("Seed", seed), amplitude, change = 100 * (measured - baseline))
  }))
  ggplot(values, aes(amplitude, change, colour = seed, group = seed)) +
    geom_hline(yintercept = 0, linetype = "dotted", colour = "#777777") +
    geom_line(linewidth = 0.45) + geom_point(size = 1.5) +
    scale_colour_manual(values = colours[1:3]) +
    scale_x_continuous(breaks = c(0, 0.5, 2.5, 9), guide = guide_axis(n.dodge = 2)) +
    labs(title = specification[3], x = "Arm translation amplitude", y = "Score change (pp)", colour = NULL)
})
save_panels(plots, output_path("graph_disconnection_response"), height = 4.5, shared_legend = TRUE)

variant_names <- c("sharp", "truth", "diffuse", "near_uniform", "shuffled", "constant")
variant_labels <- c("Sharp", "Smooth", "Diffuse", "Near-\nuniform", "Shuffled", "Constant")
allocation_data <- do.call(rbind, lapply(calibration$allocation_rows, function(row) {
  data.frame(seed = paste("Seed", row$seed), variant = row$variant,
             coherence = row$v2$allocation_coherence_raw,
             perplexity = row$v2$allocation_perplexity_fraction)
}))
allocation_data$variant <- factor(allocation_data$variant, levels = variant_names, labels = variant_labels)
plots <- lapply(c("coherence", "perplexity"), function(metric) {
  panel <- ggplot(allocation_data, aes(variant, .data[[metric]], colour = seed, group = seed)) +
    geom_line(linewidth = 0.45) + geom_point(size = 1.5) +
    scale_colour_manual(values = colours[1:3]) +
    scale_y_continuous(limits = c(-0.03, 1.08)) +
    scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
    labs(title = if (metric == "coherence") "Raw allocation coherence" else "Allocation dispersion",
         x = NULL, y = if (metric == "coherence") "Raw coherence" else "Mean row perplexity / dimension",
         colour = NULL)
  if (metric == "perplexity") panel <- panel +
    geom_hline(yintercept = 0.95, linetype = "dashed", colour = "#777777") +
    annotate("text", x = 5.7, y = 1.025, label = "0.95 guard", family = "Arial", size = 3, colour = "black")
  panel
})
save_panels(plots, output_path("allocation_validity_guards"), height = 2.8, shared_legend = TRUE)

statuses <- inventory$file_allocation_status_counts
inventory_data <- data.frame(category = c("Valid native", "Guarded native", "Non-native"),
                             count = c(statuses$valid, statuses$near_uniform + statuses$degenerate, statuses$non_native))
null_data <- data.frame(category = c("Native, quotable", "Coherence >\nnull maximum", "Edge + separation +\ncoherence > null maximum"),
                        count = c(null_record$n_quotable_native, null_record$n_native_coherence_above_max,
                                  null_record$n_native_joint_edge_separation_coherence_above_max))
plots <- lapply(seq_along(list(inventory_data, null_data)), function(index) {
  values <- list(inventory_data, null_data)[[index]]
  values$category <- factor(values$category, levels = rev(values$category))
  ggplot(values, aes(count, category, fill = category)) + geom_col(width = 0.65, show.legend = FALSE) +
    geom_text(aes(label = count), hjust = -0.25, family = "Arial", size = 3.1, colour = "black") +
    scale_fill_manual(values = c("#9AA1A8", colours[2], colours[1])) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(title = c("Full inventory: applicability", "Ten-permutation comparison")[index], y = NULL,
         x = c("Saved latent files (aliases retained)", "Runs in focused manifest")[index])
})
save_panels(plots, output_path("inventory_null_comparison"), height = 2.85, width = 8.4, base_size = 12)

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
    baseline <- mean(vapply(Filter(function(row) row$k == 15 && row$rank == 50, selected),
                            function(row) row[[metric]], numeric(1)))
    changes <- vapply(seq_len(nrow(settings)), function(setting_index) {
      rows <- Filter(function(row) row$k == settings$neighbour_count[setting_index] &&
                       row$rank == settings$rank[setting_index], selected)
      mean(vapply(rows, function(row) row[[metric]], numeric(1))) - baseline
    }, numeric(1))
    data.frame(group = paste(dataset_labels[dataset], if (native) "P" else "Z", sep = " / "),
               setting = settings$label, change = changes)
  }))
  values$group <- factor(values$group, levels = rev(unique(values$group)))
  values$setting <- factor(values$setting, levels = settings$label)
  colour_limit <- max(abs(values$change), 0.01)
  ggplot(values, aes(setting, group, fill = change)) + geom_tile() +
    geom_vline(xintercept = c(3.5, 6.5), colour = "black", linewidth = 0.3) +
    scale_fill_gradient2(low = "#3979AD", mid = "#F6F6F6", high = "#B94040",
                         midpoint = 0, limits = c(-colour_limit, colour_limit),
                         breaks = c(-colour_limit, 0, colour_limit),
                         labels = function(values) formatC(values, format = "f", digits = 3)) +
    scale_x_discrete(guide = guide_axis(angle = 45)) +
    labs(title = metric_labels[index], x = "Neighbour count / reference PC count", y = NULL,
         fill = "Change") +
    guides(fill = guide_colourbar(title.position = "left", barwidth = unit(1.25, "in"),
                                  barheight = unit(0.08, "in"),
                                  theme = theme(legend.title = element_text(margin = margin(r = 12)))))
})
save_panels(plots, output_path("scale_reference_sensitivity"), height = 5.5, width = 8, base_size = 12)

plots <- lapply(seq_along(metric_names), function(index) {
  metric <- metric_names[index]
  values <- do.call(rbind, lapply(sensitivity$correspondence, function(row) {
    data.frame(dataset = dataset_labels[row$dataset], model_seed = row$model_seed,
               fraction = row$requested_fraction * 100, score = row[[metric]])
  }))
  seed_means <- aggregate(score ~ dataset + model_seed + fraction, values, mean)
  means <- aggregate(score ~ dataset + fraction, values, mean)
  ggplot(means, aes(fraction, score, colour = dataset, group = dataset)) +
    geom_line(data = seed_means, aes(group = interaction(dataset, model_seed)), linewidth = 0.3, alpha = 0.35) +
    geom_line(linewidth = 0.65) + geom_point(size = 1.5) +
    scale_colour_manual(values = setNames(colours, dataset_labels[datasets])) +
    scale_y_continuous(limits = c(-0.03, 1.04)) +
    scale_x_continuous(breaks = c(0, 10, 25, 50, 100)) +
    labs(title = metric_labels[index], x = "Rows selected for cyclic reassignment (%)", y = "Score", colour = NULL) +
    guides(colour = guide_legend(nrow = 1))
})
save_panels(plots, output_path("row_correspondence_sensitivity"), height = 4.0, shared_legend = TRUE)
status <- system2("python3", file.path(script_dir, "build_architecture.py"))
stopifnot(status == 0)
status <- system2(Sys.getenv("SVG_PYTHON", "/usr/bin/python3"), c(file.path(script_dir, "render_schematic.py"),
                   file.path(paper_dir, "figures/architecture.svg"),
                   file.path(paper_dir, "figures/architecture.pdf")))
stopifnot(status == 0)
cat("Rendered six data figures and the editable vector measurement workflow.\n")
