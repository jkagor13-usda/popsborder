# This code has been taken from the supplementary materials provided in the following article
# Clark, R.G., Barnes, B. & Parsa, M. Clustered and Unclustered Group Testing for Biosecurity. JABES 29, 193–211 (2024). https://doi.org/10.1007/s13253-023-00566-x

#########################################################################
# This script performs the simulations for the Supp Materials
# 27/4/2023
# to run from paperspace linux:
#  nohup Rscript simstudy_paperspace_27_04_2023.R >& stuff.txt &
#########################################################################

#########################################################################
# Define functions for calculating densities, likelihoods and other
# quantities
#########################################################################

#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Expected exactly one JSON argument")
}

# Parse JSON input
library(jsonlite)
input <- fromJSON(args[1])

# Define your R functions
function1 <- function(param1 = NULL, param2 = NULL) {
  # Your R code here
  result <- list(
    output_var1 = param1*param2,
    output_var2 = param1+param2,
    status = "success"
  )
  return(result)
}

function2 <- function(data = NULL) {
  # Your R code here
  result <- list(
    mean_value = mean(data),
    processed_data = "insert_processed_data",
    status = "success"
  )
  return(result)
}

# Route to the appropriate function
result <- switch(
  input$function,
  "function1" = do.call(function1, input$args),
  "function2" = do.call(function2, input$args),
  stop(paste("Unknown function:", input$function))
)

# Output as JSON (must be last line)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))

























