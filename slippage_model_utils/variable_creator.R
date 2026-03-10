#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Windows sometimes splits arguments - rejoin them
if (length(args) == 0) {
  stop("No arguments provided")
}

# If multiple args, assume they were split by spaces - rejoin
json_input <- paste(args, collapse = " ")

# Parse JSON input
library(jsonlite)
input <- fromJSON(json_input)

# Define your R functions
function1 <- function(param1 = NULL, param2 = NULL) {
  result <- list(
    output_var1 = param1 * param2,
    output_var2 = param1 + param2,
    status = "success"
  )
  return(result)
}

function2 <- function(data = NULL) {
  result <- list(
    mean_value = mean(data),
    processed_data = data * 2,  # Example processing
    status = "success"
  )
  return(result)
}



function3 <- function(df = NULL) {
  # Convert list-of-columns format to dataframe
  # (jsonlite sends dataframes as column-oriented objects)
  if (!is.data.frame(df)) {
    df <- as.data.frame(df)
  }

  # Example: Create a new column based on existing columns
  # Adjust this logic to your needs
  if ("col1" %in% names(df) && "col2" %in% names(df)) {
    df$new_column <- df$col1 + df$col2
  } else if ("value" %in% names(df)) {
    df$new_column <- df$value * 2
  } else {
    # Generic example: add row numbers
    df$new_column <- seq_len(nrow(df))
  }

  # Return the dataframe
  # jsonlite will convert it back to JSON in column-oriented format
  return(df)
}


function4 <- function(df = NULL, operation = "sum", multiplier = 1) {
  if (!is.data.frame(df)) {
    df <- as.data.frame(df)
  }

  # Apply different operations based on parameter
  if (operation == "sum" && all(c("col1", "col2") %in% names(df))) {
    df$new_column <- (df$col1 + df$col2) * multiplier
  } else if (operation == "product" && all(c("col1", "col2") %in% names(df))) {
    df$new_column <- (df$col1 * df$col2) * multiplier
  } else if (operation == "mean" && all(c("col1", "col2") %in% names(df))) {
    df$new_column <- ((df$col1 + df$col2) / 2) * multiplier
  } else {
    df$new_column <- seq_len(nrow(df)) * multiplier
  }

  return(df)
}

# Route to the appropriate function
result <- switch(
  input[["function"]],
  "function1" = do.call(function1, input$args),
  "function2" = do.call(function2, input$args),
  "function3" = do.call(function3, input$args),
  "function4" = do.call(function4, input$args),
  stop(paste("Unknown function:", input[["function"]]))
)


# Output as JSON (must be last line)
# For dataframes, use dataframe = "columns" to maintain structure
cat(toJSON(result, auto_unbox = TRUE, digits = NA, dataframe = "columns"))