#!/usr/bin/env Rscript

# Suppress startup messages
options(warn = -1)
suppressPackageStartupMessages({
  library(jsonlite)
  library(stringr)
  library(dplyr)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  cat(toJSON(list(
    error = "Expected exactly one JSON argument",
    status = "error"
  ), auto_unbox = TRUE))
  quit(status = 1)
}

# Parse JSON input
input <- tryCatch({
  fromJSON(args[1], simplifyVector = FALSE)
}, error = function(e) {
  cat(toJSON(list(
    error = paste("Failed to parse JSON input:", e$message),
    status = "error",
    input_received = args[1]
  ), auto_unbox = TRUE))
  quit(status = 1)
})

# Validate input structure
if (is.null(input$func_name)) {
  cat(toJSON(list(
    error = "Missing 'func_name' field in input",
    status = "error",
    input_keys = names(input)
  ), auto_unbox = TRUE))
  quit(status = 1)
}

# ===== Helper Functions =====

remove_extra_chars <- function(suffix_string, prefix_string, text) {
  text <- gsub(suffix_string, "", text)
  text <- gsub(prefix_string, "", text)
  text <- str_squish(text)
  return(text)
}

# Helper function to properly convert JSON list to data.frame
json_list_to_dataframe <- function(json_list) {
  # json_list is a named list where each element is a vector (column)

  if (!is.list(json_list)) {
    stop("Input must be a list")
  }

  if (length(json_list) == 0) {
    stop("Input list is empty")
  }

  # Get column names
  col_names <- names(json_list)

  if (is.null(col_names) || any(col_names == "")) {
    stop("All list elements must be named")
  }

  # Check all columns have same length
  lengths <- sapply(json_list, length)
  if (length(unique(lengths)) > 1) {
    stop(paste("All columns must have same length. Got:", paste(lengths, collapse=", ")))
  }

  n_rows <- lengths[1]

  # Create empty data.frame with correct number of rows
  df <- data.frame(row.names = 1:n_rows, stringsAsFactors = FALSE)

  # Add each column
  for (col_name in col_names) {
    df[[col_name]] <- json_list[[col_name]]
  }

  return(df)
}


# ===== Main Functions =====

basic_text_preproc <- function(text_field, suffix_string = NULL, prefix_string = NULL) {
  text <- text_field
  text <- tolower(text)
  text[is.na(text) | text == "not selected"] <- "missing"
  text <- gsub("[[:punct:]]+", "", text)

  # Drop end of producer names beginning with "box"
  ndx_box1 <- regexpr("box", text) - 1
  ndx_box1 <- if_else(ndx_box1 > 0, ndx_box1, nchar(text))
  text <- substr(text, 1, ndx_box1)

  # Drop stand-alone numbers
  text <- str_remove_all(text, "\\b\\d+\\b")

  # Drop leading & trailing blanks, replace strings of blanks with a single blank
  text <- str_squish(text)

  # Drop known uninformative prefix and suffix strings
  if (is.null(suffix_string)) {
    suffix_string <- " sa| s a|sociedad anonima| inc| llc| ltd| ltda| cv| rl| co| co ltd| corp| bv| b v| corporation| company| limited"
    suffix_string <- paste0(gsub("\\|", "$|", suffix_string), "$")
  }
  if (is.null(prefix_string)) {
    prefix_string <- "mr |m r "
    prefix_string <- paste0("^", gsub("\\|", "|^", prefix_string))
  }

  text <- remove_extra_chars(suffix_string = suffix_string,
                             prefix_string = prefix_string, text)

  return(list(
    processed_text = text,
    status = "success"
  ))
}




# ===== Quantity Threshold Binary Function =====
generate_quantity_binaries <- function(df,
                                      quantity_threshold = 200,
                                      group_cols = c("RISK_UNIT")) {
  if (is.null(df)) {
    stop("df argument is required")
  }

  # Properly convert to data frame
  if (!is.data.frame(df)) {
    df <- json_list_to_dataframe(df)
  }

  # Ensure QUANTITY is numeric
  df$QUANTITY <- as.numeric(df$QUANTITY)

  # Ensure group_cols is a character vector (flatten if nested list)
  if (is.list(group_cols) && !is.data.frame(group_cols)) {
    group_cols <- unlist(group_cols, recursive = TRUE)
  }
  group_cols <- as.character(group_cols)

  # Remove any NA or empty strings
  group_cols <- group_cols[!is.na(group_cols) & nchar(group_cols) > 0]

  if (length(group_cols) == 0) {
    stop("group_cols must contain at least one valid column name")
  }

  # Ensure required columns exist
  required_cols <- c("QUANTITY", group_cols)
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    available_cols <- names(df)
    stop(paste0(
      "Missing required columns: ", paste(missing_cols, collapse = ", "), "\n",
      "Available columns: ", paste(available_cols, collapse = ", ")
    ))
  }

  # Use !! and sym() for dynamic column selection
  if (length(group_cols) == 1) {
    # Single grouping column - use simpler approach
    dt <- df %>%
      group_by(.data[[group_cols[1]]]) %>%
      summarize(
        TOTAL_QTY = sum(QUANTITY, na.rm = TRUE),
        MIN_QTY = min(QUANTITY, na.rm = TRUE),
        MEDIAN_QTY = median(QUANTITY, na.rm = TRUE),
        FRAC_SMALL = mean(QUANTITY < quantity_threshold, na.rm = TRUE),
        MEDIAN_QTY_LT200 = median(QUANTITY, na.rm = TRUE) < quantity_threshold,
        FRAC_SMALL_GT07 = mean(QUANTITY < quantity_threshold, na.rm = TRUE) > 0.7,
        ANY_SMALL = as.integer(any(QUANTITY < quantity_threshold, na.rm = TRUE)),
        n_commodity_lines = n(),
        .groups = "drop"
      )
  } else {
    # Multiple grouping columns
    dt <- df %>%
      group_by(across(all_of(group_cols))) %>%
      summarize(
        TOTAL_QTY = sum(QUANTITY, na.rm = TRUE),
        MIN_QTY = min(QUANTITY, na.rm = TRUE),
        MEDIAN_QTY = median(QUANTITY, na.rm = TRUE),
        FRAC_SMALL = mean(QUANTITY < quantity_threshold, na.rm = TRUE),
        MEDIAN_QTY_LT200 = median(QUANTITY, na.rm = TRUE) < quantity_threshold,
        FRAC_SMALL_GT07 = mean(QUANTITY < quantity_threshold, na.rm = TRUE) > 0.7,
        ANY_SMALL = as.integer(any(QUANTITY < quantity_threshold, na.rm = TRUE)),
        n_commodity_lines = n(),
        .groups = "drop"
      )
  }

  # Convert to data.frame
  dt <- as.data.frame(dt, stringsAsFactors = FALSE)

  # Convert to list format for JSON transfer
  result_list <- as.list(dt)

  # Ensure character/factor columns are converted to character vectors
  for (col in names(result_list)) {
    if (is.factor(result_list[[col]])) {
      result_list[[col]] <- as.character(result_list[[col]])
    }
  }

  return(result_list)
}











function1 <- function(param1 = NULL, param2 = NULL) {
  result <- list(
    sum = if(!is.null(param1) && !is.null(param2)) param1 + param2 else NA,
    param1 = param1,
    param2 = param2,
    status = "success"
  )
  return(result)
}

function2 <- function(data = NULL) {
  result <- list(
    mean_value = if(!is.null(data)) mean(data) else NA,
    length = if(!is.null(data)) length(data) else 0,
    status = "success"
  )
  return(result)
}

function3 <- function(df = NULL) {
  if (is.null(df)) {
    stop("df argument is required")
  }

  df <- as.data.frame(df)
  numeric_cols <- sapply(df, is.numeric)
  if (any(numeric_cols)) {
    df$new_column <- rowSums(df[, numeric_cols, drop = FALSE])
  } else {
    df$new_column <- NA
  }

  return(df)
}

function4 <- function(df = NULL, operation = "sum", multiplier = 1.0) {
  if (is.null(df)) {
    stop("df argument is required")
  }

  df <- as.data.frame(df)
  numeric_cols <- sapply(df, is.numeric)

  if (!any(numeric_cols)) {
    df$result <- NA
    return(df)
  }

  if (operation == "sum") {
    df$result <- rowSums(df[, numeric_cols, drop = FALSE]) * multiplier
  } else if (operation == "product") {
    df$result <- apply(df[, numeric_cols, drop = FALSE], 1, prod) * multiplier
  } else if (operation == "mean") {
    df$result <- rowMeans(df[, numeric_cols, drop = FALSE]) * multiplier
  } else {
    stop(paste("Unknown operation:", operation))
  }

  return(df)
}

# ===== Main Execution =====

# Ensure func_name is a character string
func_name <- as.character(input$func_name)

# Execute function with error handling
result <- tryCatch({
  switch(
    func_name,
    "basic_text_preproc" = do.call(basic_text_preproc, input$args),
    "generate_quantity_binaries" = do.call(generate_quantity_binaries, input$args),
    "function1" = do.call(function1, input$args),
    "function2" = do.call(function2, input$args),
    "function3" = do.call(function3, input$args),
    "function4" = do.call(function4, input$args),
    # Default case if function name doesn't match
    {
      list(
        error = paste("Unknown function:", func_name),
        status = "error",
        available_functions = c("basic_text_preproc", "function1", "function2", "function3", "function4")
      )
    }
  )
}, error = function(e) {
  list(
    error = e$message,
    status = "error",
    function_name = func_name,
    traceback = paste(capture.output(traceback()), collapse = "\n")
  )
})

# Output as JSON (must be last line)
cat(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null"))
