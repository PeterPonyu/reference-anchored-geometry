library(ggplot2)
library(grid)

paper_theme <- function(base_size = 10) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      text = element_text(family = "Arial", colour = "black", face = "plain"),
      axis.text = element_text(colour = "black", face = "plain"),
      axis.title = element_text(colour = "black", face = "plain"),
      plot.title = element_text(hjust = 0, face = "plain", size = base_size,
                                margin = margin(b = 3)),
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
      axis.title.x = element_text(margin = margin(t = 2)),
      axis.title.y = element_text(margin = margin(r = 2)),
      # Keep titles and axis text fully inside the exported page.  This small
      # internal allowance does not reintroduce inter-panel whitespace.
      plot.margin = margin(3, 2, 2, 3)
    )
}

align_panels <- function(panels) {
  stopifnot(length(unique(vapply(panels, function(panel) length(panel$widths), integer(1)))) == 1,
            length(unique(vapply(panels, function(panel) length(panel$heights), integer(1)))) == 1)
  common_widths <- do.call(unit.pmax, lapply(panels, function(panel) panel$widths))
  common_heights <- do.call(unit.pmax, lapply(panels, function(panel) panel$heights))
  lapply(panels, function(panel) {
    # Shared rows/columns keep panel frames aligned. A shorter x-axis or narrower
    # y-axis keeps its own extent; the spare space moves to the outer margin so
    # axis titles stay next to their ticks instead of floating.
    own_heights <- panel$heights
    own_widths <- panel$widths
    panel$widths <- common_widths
    panel$heights <- common_heights
    axis_b <- panel$layout$t[panel$layout$name == "axis-b"]
    if (length(axis_b) == 1) {
      spare <- common_heights[axis_b] - own_heights[axis_b]
      panel$heights[axis_b] <- own_heights[axis_b]
      last <- length(panel$heights)
      panel$heights[last] <- panel$heights[last] + spare
    }
    axis_l <- panel$layout$l[panel$layout$name == "axis-l"]
    if (length(axis_l) == 1) {
      spare <- common_widths[axis_l] - own_widths[axis_l]
      panel$widths[axis_l] <- own_widths[axis_l]
      panel$widths[1] <- panel$widths[1] + spare
    }
    panel
  })
}

label_panel <- function(panel, label, size = 12, anchor = c("panel", "axis")) {
  anchor <- match.arg(anchor)
  stopifnot(grepl("^[a-z]$", label))
  panel_cell <- which(panel$layout$name == "panel")[1]
  stopifnot(length(panel_cell) == 1)
  # Keep the complete letter outside the upper-left axis corner.  Its right
  # edge clears the axis by 5 pt. A 9-pt anchor offset leaves about 7 pt
  # below the actual Arial glyph, including Cairo's font descent.
  # Reserve real header space so high y-ticks and the page edge stay clear.
  header_rows <- seq_len(panel$layout$t[panel_cell] - 1)
  header_height <- convertHeight(sum(panel$heights[header_rows]), "pt", valueOnly = TRUE)
  header_minimum <- size * 1.2 + 11
  if (header_height < header_minimum) {
    panel$heights[1] <- panel$heights[1] + unit(header_minimum - header_height, "pt")
  }
  # Heatmaps with long row names anchor the letter left of those names, in line
  # with the letters of the coordinate panels above them.
  left <- panel$layout$l[panel_cell]
  if (anchor == "axis") left <- panel$layout$l[panel$layout$name == "axis-l"]
  letter <- textGrob(label, x = unit(-5, "pt"),
                     y = unit(1, "npc") + unit(9, "pt"),
                     just = c("right", "bottom"),
                     gp = gpar(fontfamily = "Arial", fontface = "bold",
                               fontsize = size, col = "black"))
  gtable::gtable_add_grob(panel, letter,
                          t = panel$layout$t[panel_cell],
                          l = left,
                          b = panel$layout$b[panel_cell],
                          r = panel$layout$r[panel_cell],
                          z = Inf, clip = "off",
                          name = paste0("panel-label-", label))
}

arrange_compact <- function(grobs, ncol, square_row_height = NULL, width = 7.2) {
  stopifnot(length(grobs) > 0, ncol >= 1)
  if (!is.null(square_row_height)) {
    # Equal-aspect coordinate panels can leave unused space inside a wide
    # grid cell even with zero padding. Pack their actual square frames and
    # fixed axis/title extents, then centre the block in the figure width.
    stopifnot(length(grobs) %% ncol == 0)
    # Cairo exports whole-point page widths; use the same extent so a
    # width-limited four-column row does not grow beyond its prior frame.
    width <- floor(width * 72) / 72
    extents <- lapply(grobs, function(panel) {
      cell <- which(panel$layout$name == "panel")
      stopifnot(length(cell) == 1)
      panel_rows <- seq.int(panel$layout$t[cell], panel$layout$b[cell])
      panel_cols <- seq.int(panel$layout$l[cell], panel$layout$r[cell])
      fixed_width <- convertWidth(sum(panel$widths[-panel_cols]), "in", valueOnly = TRUE)
      fixed_height <- convertHeight(sum(panel$heights[-panel_rows]), "in", valueOnly = TRUE)
      left_width <- convertWidth(sum(panel$widths[seq_len(panel$layout$l[cell] - 1)]),
                                  "in", valueOnly = TRUE)
      title <- panel$grobs[[which(panel$layout$name == "title")]]$children[[1]]
      title_width <- left_width + convertWidth(title$x + grobWidth(title), "in", valueOnly = TRUE) + 2 / 72
      c(width = fixed_width, height = fixed_height, title = title_width)
    })
    fixed_width <- max(vapply(extents, function(x) x[["width"]], numeric(1)))
    fixed_height <- max(vapply(extents, function(x) x[["height"]], numeric(1)))
    side <- min(square_row_height - fixed_height, width / ncol - fixed_width)
    cell_width <- max(side + fixed_width,
                       max(vapply(extents, function(x) x[["title"]], numeric(1))))
    stopifnot(side > 0, cell_width <= width / ncol + 1e-6)
    block <- gridExtra::arrangeGrob(grobs = grobs, ncol = ncol,
                                    widths = unit(rep(cell_width, ncol), "in"),
                                    heights = unit(rep(square_row_height, length(grobs) / ncol), "in"),
                                    padding = unit(0, "pt"))
    frame <- gtable::gtable(widths = unit.c(unit(1, "null"),
                                           unit(ncol * cell_width, "in"), unit(1, "null")),
                             heights = unit(1, "null"))
    message(sprintf("Packed %d-column coordinates: %.3f-in cells, %.3f-in block (%.3f-in figure).",
                    ncol, cell_width, ncol * cell_width, width))
    return(gtable::gtable_add_grob(frame, block, t = 1, l = 2, clip = "off"))
  }
  gridExtra::arrangeGrob(grobs = grobs, ncol = ncol,
                         widths = unit(rep(1, ncol), "null"),
                         padding = unit(0, "pt"))
}

save_panels <- function(plots, filename, ncol = 2, width = 7.2,
                        height = 2.9 * ceiling(length(plots) / ncol), tags = length(plots) > 1,
                        shared_legend = FALSE, row_tags = FALSE, base_size = 11) {
  measure_file <- tempfile(fileext = ".pdf")
  cairo_pdf(measure_file, width = width, height = height, family = "Arial")
  on.exit({ dev.off(); unlink(measure_file) }, add = TRUE)
  # The complete paper theme would otherwise erase panel-specific settings
  # (inside legend positions, label angles); re-apply them on top of it.
  styled <- lapply(plots, function(panel) panel + paper_theme(base_size) + panel$theme)
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
  panels <- align_panels(lapply(styled, ggplotGrob))
  if (tags && !row_tags) {
    panels <- lapply(seq_along(panels), function(index) {
      label_panel(panels[[index]], letters[index], size = 12)
    })
  }
  row_count <- ceiling(length(panels) / ncol)
  rows <- lapply(seq_len(row_count), function(row_index) {
    indices <- seq.int((row_index - 1) * ncol + 1, min(row_index * ncol, length(panels)))
    row <- arrange_compact(panels[indices], ncol)
    if (row_tags) {
      label <- textGrob(letters[row_index], x = unit(0, "npc"),
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
    contents <- append(rows, list(legend))
    row_heights <- lapply(seq_len(row_count), function(index) unit(1, "null"))
    heights <- do.call(unit.c, append(row_heights,
                       list(grobHeight(legend) + unit(0.025, "in"))))
  }
  figure <- gridExtra::arrangeGrob(grobs = contents, ncol = 1, heights = heights,
                                   padding = unit(0, "pt"))
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  ggsave(filename, figure, device = cairo_pdf, width = width, height = height,
         units = "in", bg = "transparent", family = "Arial", limitsize = FALSE)
  invisible(figure)
}

# Each panel keeps its own axis width, so a long label cannot push a white
# border into the other panels. Legends sit in a band directly under the data
# rectangle they describe. Shared groups use one legend, centered on the
# combined data span of those panels.
pack_panels <- function(plots, filename, ncol = 2, width = 6.5, base_size = 11,
                        panel_height = 1.45, col_gap = 0.12, row_gap = 0.07,
                        shared = list()) {
  stopifnot(length(plots) > 0, ncol >= 1)
  if (length(panel_height) == 1) panel_height <- rep(panel_height, length(plots))
  stopifnot(length(panel_height) == length(plots))
  page_width <- width
  outer_left <- 0.08
  outer_right <- 0.10
  width <- page_width - outer_left - outer_right
  measure_file <- tempfile(fileext = ".pdf")
  cairo_pdf(measure_file, width = width, height = 12, family = "Arial")
  on.exit({ dev.off(); unlink(measure_file) }, add = TRUE)
  styled <- lapply(plots, function(panel) {
    panel + paper_theme(base_size) +
      theme(legend.box.spacing = unit(0, "pt"),
            legend.margin = margin(0, 0, 0, 0),
            legend.spacing.x = unit(7, "pt"),
            legend.spacing.y = unit(1, "pt"),
            legend.key.width = unit(14, "pt"),
            legend.key.height = unit(12, "pt"),
            legend.text = element_text(size = base_size - 1, margin = margin(0, 1, 0, 1))) +
      panel$theme
  })
  shared_owner <- rep(NA_integer_, length(styled))
  for (group in shared) {
    stopifnot(length(group) >= 2, all(group %in% seq_along(styled)))
    shared_owner[group] <- group[[1]]
  }
  legend_of <- function(plot) {
    grob <- ggplotGrob(plot)
    index <- which(grob$layout$name == "guide-box-bottom")
    if (length(index) != 1) return(NULL)
    box <- grob$grobs[[index]]
    if (inherits(box, "zeroGrob")) return(NULL)
    if (convertWidth(grobWidth(box), "in", valueOnly = TRUE) < 0.05) return(NULL)
    box
  }
  inside <- function(plot) {
    pos <- plot$theme$legend.position
    !is.null(pos) && identical(as.character(pos)[1], "inside")
  }
  legends <- vector("list", length(styled))
  bare <- vector("list", length(styled))
  for (index in seq_along(styled)) {
    owner <- shared_owner[[index]]
    if (!is.na(owner)) {
      if (index == owner) {
        extracted <- legend_of(styled[[index]])
        if (!is.null(extracted)) legends[[index]] <- extracted
      }
      bare[[index]] <- ggplotGrob(styled[[index]] + theme(legend.position = "none"))
    } else if (inside(styled[[index]])) {
      bare[[index]] <- ggplotGrob(styled[[index]])
    } else {
      extracted <- legend_of(styled[[index]])
      if (!is.null(extracted)) legends[[index]] <- extracted
      bare[[index]] <- ggplotGrob(styled[[index]] + theme(legend.position = "none"))
    }
  }
  bare <- lapply(seq_along(bare), function(index) label_panel(bare[[index]], letters[index], size = 12))
  column_width <- (width - col_gap * (ncol - 1)) / ncol
  bare <- lapply(seq_along(bare), function(index) {
    grob <- bare[[index]]
    cell <- which(grob$layout$name == "panel")[1]
    cols <- seq.int(grob$layout$l[cell], grob$layout$r[cell])
    rows <- seq.int(grob$layout$t[cell], grob$layout$b[cell])
    fixed <- sum(convertWidth(grob$widths, "in", valueOnly = TRUE))
    panel_width <- column_width - fixed
    if (!is.finite(panel_width) || panel_width < 1.05) {
      stop(sprintf("Panel %s is %.2f in wide after labels; column is %.2f in.",
                   letters[index], panel_width, column_width))
    }
    grob$widths[cols] <- unit(panel_width, "in")
    grob$heights[rows] <- unit(panel_height[[index]], "in")
    grob
  })
  row_count <- ceiling(length(bare) / ncol)
  row_heights <- panel_height
  # Panels in one row share the taller data frame so their axes line up.
  for (row_index in seq_len(row_count)) {
    indices <- seq.int((row_index - 1) * ncol + 1, min(row_index * ncol, length(bare)))
    row_heights[indices] <- max(panel_height[indices])
  }
  bare <- lapply(seq_along(bare), function(index) {
    grob <- bare[[index]]
    cell <- which(grob$layout$name == "panel")[1]
    rows <- seq.int(grob$layout$t[cell], grob$layout$b[cell])
    grob$heights[rows] <- unit(row_heights[[index]], "in")
    grob
  })
  align_plot_frames <- function(grobs) {
    info <- lapply(grobs, function(grob) {
      cell <- which(grob$layout$name == "panel")[1]
      bottom <- grob$layout$b[cell]
      heights <- convertHeight(grob$heights, "in", valueOnly = TRUE)
      above <- sum(heights[seq_len(bottom)])
      list(above = above, below = sum(heights) - above)
    })
    max_above <- max(vapply(info, function(item) item$above, numeric(1)))
    max_below <- max(vapply(info, function(item) item$below, numeric(1)))
    lapply(seq_along(grobs), function(index) {
      grob <- grobs[[index]]
      grob$heights[1] <- grob$heights[1] + unit(max_above - info[[index]]$above, "in")
      last <- length(grob$heights)
      grob$heights[last] <- grob$heights[last] + unit(max_below - info[[index]]$below, "in")
      grob
    })
  }
  panel_box <- function(grob) {
    cell <- which(grob$layout$name == "panel")[1]
    left_cols <- seq_len(grob$layout$l[cell] - 1)
    left <- if (length(left_cols)) sum(convertWidth(grob$widths[left_cols], "in", valueOnly = TRUE)) else 0
    panel_width <- sum(convertWidth(grob$widths[seq.int(grob$layout$l[cell], grob$layout$r[cell])],
                                    "in", valueOnly = TRUE))
    c(left = left, width = panel_width)
  }
  place_legend <- function(legend, left, span, limit) {
    if (is.null(legend)) return(NULL)
    legend_width <- convertWidth(grobWidth(legend), "in", valueOnly = TRUE)
    legend_height <- convertHeight(grobHeight(legend), "in", valueOnly = TRUE)
    x_left <- left + span / 2 - legend_width / 2
    x_left <- max(0, min(x_left, max(0, limit - legend_width)))
    list(legend = legend, x = x_left, width = legend_width, height = legend_height)
  }
  rows <- lapply(seq_len(row_count), function(row_index) {
    indices <- seq.int((row_index - 1) * ncol + 1, min(row_index * ncol, length(bare)))
    grobs <- align_plot_frames(bare[indices])
    plot_height <- sum(convertHeight(grobs[[1]]$heights, "in", valueOnly = TRUE))
    n <- length(grobs)
    widths <- unlist(lapply(seq_len(n), function(index) c(column_width, if (index < n) col_gap)))
    plot_row <- gtable::gtable(widths = unit(widths, "in"), heights = unit(plot_height, "in"))
    boxes <- lapply(grobs, panel_box)
    for (index in seq_len(n)) {
      plot_row <- gtable::gtable_add_grob(plot_row, grobs[[index]], t = 1, l = index * 2 - 1, clip = "off")
    }
    owners <- shared_owner[indices]
    shared_ids <- unique(owners[!is.na(owners)])
    placed <- list()
    if (length(shared_ids)) {
      for (owner in shared_ids) {
        members <- which(owners == owner)
        legend <- legends[[indices[members[[1]]]]]
        span_left <- (members[[1]] - 1) * (column_width + col_gap) + boxes[[members[[1]]]][["left"]]
        last <- members[[length(members)]]
        span_right <- (last - 1) * (column_width + col_gap) + boxes[[last]][["left"]] + boxes[[last]][["width"]]
        placed[[length(placed) + 1]] <- place_legend(legend, span_left, span_right - span_left, width)
      }
    }
    for (index in seq_len(n)) {
      if (!is.na(owners[[index]])) next
      origin <- (index - 1) * (column_width + col_gap)
      # A wide key may use the gutter, but it stops before the next data frame.
      limit <- if (index < n) origin + column_width + col_gap * 0.8 else width
      legend_index <- indices[[index]]
      this_legend <- legends[[legend_index]]
      item <- place_legend(this_legend, origin + boxes[[index]][["left"]],
                           boxes[[index]][["width"]], limit)
      if (!is.null(item)) placed[[length(placed) + 1]] <- item
    }
    legend_height <- if (length(placed)) max(vapply(placed, function(item) item$height, numeric(1))) else 0
    if (legend_height > 0) legend_height <- legend_height + 0.03
    legend_row <- nullGrob()
    if (legend_height > 0) {
      children <- lapply(placed, function(item) {
        child <- item$legend
        child$vp <- viewport(x = unit(item$x + item$width / 2, "in"),
                             y = unit(1, "npc"),
                             width = unit(item$width, "in"),
                             height = unit(item$height, "in"),
                             just = c("centre", "top"))
        grobTree(child)
      })
      legend_row <- gTree(children = do.call(gList, children))
    }
    list(plot = plot_row, legend = legend_row, plot_height = plot_height, legend_height = legend_height)
  })
  heights <- unlist(lapply(seq_len(row_count), function(index) {
    c(rows[[index]]$plot_height, rows[[index]]$legend_height, if (index < row_count) row_gap)
  }))
  grobs <- unlist(lapply(seq_len(row_count), function(index) {
    item <- list(rows[[index]]$plot)
    if (rows[[index]]$legend_height > 0) item <- append(item, list(rows[[index]]$legend))
    if (index < row_count) item <- append(item, list(nullGrob()))
    item
  }), recursive = FALSE)
  figure <- gtable::gtable(widths = unit(width, "in"), heights = unit(heights, "in"))
  for (index in seq_along(grobs)) {
    figure <- gtable::gtable_add_grob(figure, grobs[[index]], t = index, l = 1, clip = "off")
  }
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  height <- sum(heights)
  page <- gtable::gtable(widths = unit(c(outer_left, width, outer_right), "in"),
                         heights = unit(height, "in"))
  page <- gtable::gtable_add_grob(page, figure, t = 1, l = 2, clip = "off")
  message(sprintf("Packed %s at %.2f x %.2f in.", basename(filename), page_width, height))
  ggsave(filename, page, device = cairo_pdf, width = page_width, height = height,
         units = "in", bg = "transparent", family = "Arial", limitsize = FALSE)
  invisible(page)
}
