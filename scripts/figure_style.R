library(ggplot2)
library(grid)

paper_theme <- function(base_size = 10) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      text = element_text(family = "Arial", colour = "black", face = "plain"),
      axis.text = element_text(colour = "black"),
      axis.title = element_text(colour = "black", face = "plain"),
      plot.title = element_text(hjust = 0.5, face = "plain", size = base_size + 1,
                                margin = margin(b = 5)),
      plot.subtitle = element_text(hjust = 0.5, face = "plain"),
      plot.tag = element_text(face = "bold", colour = "black", size = base_size + 2),
      plot.tag.position = c(0.015, 0.955),
      plot.background = element_rect(fill = "transparent", colour = NA),
      panel.background = element_rect(fill = "transparent", colour = NA),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.key.width = unit(15, "pt"),
      legend.key.height = unit(10, "pt"),
      legend.spacing.x = unit(4, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.box.background = element_rect(fill = "transparent", colour = NA),
      legend.key = element_rect(fill = "transparent", colour = NA),
      legend.title = element_text(face = "plain", colour = "black"),
      legend.text = element_text(colour = "black", face = "plain"),
      strip.background = element_blank(),
      strip.text = element_text(face = "plain", colour = "black"),
      plot.margin = margin(6, 5, 3, 5)
    )
}

save_panels <- function(plots, filename, ncol = 2, width = 7.2,
                        height = 2.9 * ceiling(length(plots) / ncol), tags = length(plots) > 1,
                        shared_legend = FALSE, row_tags = FALSE, base_size = 11) {
  measure_file <- tempfile(fileext = ".pdf")
  cairo_pdf(measure_file, width = width, height = height, family = "Arial")
  on.exit({ dev.off(); unlink(measure_file) }, add = TRUE)
  styled <- lapply(seq_along(plots), function(index) {
    panel <- plots[[index]] + paper_theme(base_size)
    if (tags && !row_tags) panel <- panel + labs(tag = LETTERS[index])
    panel
  })
  legend <- NULL
  if (shared_legend) {
    legend_source <- ggplotGrob(styled[[1]] + labs(tag = NULL))
    legend_indices <- which(grepl("^guide-box", legend_source$layout$name))
    legend_indices <- legend_indices[!vapply(legend_source$grobs[legend_indices], inherits,
                                            logical(1), "zeroGrob")]
    stopifnot(length(legend_indices) == 1)
    legend <- legend_source$grobs[[legend_indices]]
    styled <- lapply(styled, function(panel) panel + theme(legend.position = "none"))
  }
  panels <- lapply(styled, ggplotGrob)
  common_widths <- do.call(unit.pmax, lapply(panels, function(panel) panel$widths))
  panels <- lapply(panels, function(panel) { panel$widths <- common_widths; panel })
  row_count <- ceiling(length(panels) / ncol)
  rows <- lapply(seq_len(row_count), function(row_index) {
    indices <- seq.int((row_index - 1) * ncol + 1, min(row_index * ncol, length(panels)))
    row <- gridExtra::arrangeGrob(grobs = panels[indices], ncol = ncol)
    if (row_tags) {
      label <- textGrob(LETTERS[row_index], x = unit(0, "npc"),
                        y = unit(1, "npc") - unit(0.10, "in"), just = c("left", "top"),
                        gp = gpar(fontfamily = "Arial", fontface = "bold", fontsize = 12, col = "black"))
      row <- gridExtra::arrangeGrob(label, row, ncol = 2,
                                    widths = unit.c(unit(0.20, "in"), unit(1, "null")))
    }
    row
  })
  contents <- rows
  heights <- unit(rep(1, row_count), "null")
  if (!is.null(legend)) {
    insert_after <- if (row_count > 1) ceiling(row_count / 2) else 1
    contents <- append(rows, list(legend), after = insert_after)
    row_heights <- lapply(seq_len(row_count), function(index) unit(1, "null"))
    heights <- do.call(unit.c, append(row_heights,
                       list(grobHeight(legend) + unit(0.05, "in")), after = insert_after))
  }
  figure <- gridExtra::arrangeGrob(grobs = contents, ncol = 1, heights = heights)
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  ggsave(filename, figure, device = cairo_pdf, width = width, height = height,
         units = "in", bg = "transparent", family = "Arial", limitsize = FALSE)
  invisible(figure)
}
