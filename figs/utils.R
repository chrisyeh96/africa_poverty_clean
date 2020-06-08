library(dplyr, quietly=T)
library(ggplot2, quietly=T)

theme_anne = function(font = "sans", size = 10) {
    ggthemes::theme_tufte(base_size = size, base_family = font) %+replace%
        theme(
            axis.line.x = element_line(color = "black", size = .2),
            axis.line.y = element_line(color = "black", size = .2),
            panel.background = element_blank(),
            plot.background = element_rect(fill = "transparent", color = NA),
            plot.title = element_text(hjust = 0.5)
        )
}