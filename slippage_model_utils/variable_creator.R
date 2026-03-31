# © 2026 The Johns Hopkins University Applied Physics Laboratory LLC
# Suppress startup messages
options(warn = -1)
suppressPackageStartupMessages({
  library(jsonlite)
  library(stringr)
  library(dplyr)
  library(arrow)  # For Parquet support
})

# ===== Parse Command Line Arguments =====

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  cat(toJSON(list(
    error = "Expected exactly one argument (JSON file path)",
    status = "error"
  ), auto_unbox = TRUE))
  quit(status = 1)
}

json_path <- args[1]

# Read payload from JSON file
if (!file.exists(json_path)) {
  cat(toJSON(list(
    error = paste("JSON file not found:", json_path),
    status = "error"
  ), auto_unbox = TRUE))
  quit(status = 1)
}

# Parse JSON input from file
input <- tryCatch({
  fromJSON(json_path, simplifyVector = FALSE)
}, error = function(e) {
  cat(toJSON(list(
    error = paste("Failed to parse JSON file:", e$message),
    status = "error",
    json_path = json_path
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

# Extract components
func_name <- as.character(input$func_name)
func_args <- input$args
df_args <- input$df_args
use_parquet <- if (!is.null(input$use_parquet)) input$use_parquet else FALSE

# ===== Helper Functions for File I/O =====

# Read DataFrames from file paths (Parquet or CSV)
read_df_args <- function(df_paths, use_parquet) {
  dfs <- list()
  if (!is.null(df_paths) && length(df_paths) > 0) {
    for (arg_name in names(df_paths)) {
      file_path <- df_paths[[arg_name]]
      if (!file.exists(file_path)) {
        stop(paste("File not found:", file_path))
      }

      # Auto-detect format from extension or use_parquet flag
      if (use_parquet || grepl("\\.parquet$", file_path, ignore.case = TRUE)) {
        dfs[[arg_name]] <- as.data.frame(read_parquet(file_path))
      } else {
        dfs[[arg_name]] <- read.csv(file_path, stringsAsFactors = FALSE)
      }
    }
  }
  return(dfs)
}

# Write result DataFrames to temp files
write_result_dfs <- function(result, use_parquet, temp_dir = NULL) {
  df_results <- list()
  cleaned_result <- list()

  for (name in names(result)) {
    value <- result[[name]]
    if (is.data.frame(value)) {
      # Write DataFrame to temp file in Python's temp directory
      if (is.null(temp_dir)) {
        # Fallback to R's temp dir (but this causes the issue!)
        temp_dir <- tempdir()
      }

      if (use_parquet) {
        temp_file <- file.path(temp_dir, paste0(name, "_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".parquet"))
        write_parquet(value, temp_file)
      } else {
        temp_file <- file.path(temp_dir, paste0(name, "_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
        write.csv(value, temp_file, row.names = FALSE)
      }
      df_results[[name]] <- temp_file
    } else {
      cleaned_result[[name]] <- value
    }
  }

  # Add file paths to result
  if (length(df_results) > 0) {
    cleaned_result$df_results <- df_results
  }

  return(cleaned_result)
}

# ===== Text Processing Helper Functions =====

remove_extra_chars <- function(suffix_string, prefix_string, text) {
  text <- gsub(suffix_string, "", text)
  text <- gsub(prefix_string, "", text)
  text <- str_squish(text)
  return(text)
}

# ===== Main Functions =====

basic_text_preproc <- function(text_field, suffix_string = NULL, prefix_string = NULL) {
  text <- text_field
  text <- tolower(text)
  text[is.na(text) | text == "not selected"] <- "missing"
  text <- gsub("[[:punct:]]+", "", text)

  # Drop end of producer names beginning with "box"
  ndx_box1 <- regexpr("box", text) - 1
  ndx_box1 <- ifelse(ndx_box1 > 0, ndx_box1, nchar(text))
  text <- substring(text, 1, ndx_box1)  # << fix here

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

  # Ensure df is a data frame
  if (!is.data.frame(df)) {
    stop("df must be a data.frame")
  }

  # Ensure QUANTITY is numeric
  if (!"QUANTITY" %in% names(df)) {
    stop(paste("QUANTITY column not found. Available columns:", paste(names(df), collapse = ", ")))
  }
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

  # Perform aggregation
  if (length(group_cols) == 1) {
    # Single grouping column
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

  # Convert to data.frame and ensure proper types
  dt <- as.data.frame(dt, stringsAsFactors = FALSE)

  # Ensure character/factor columns are converted to character
  for (col in names(dt)) {
    if (is.factor(dt[[col]])) {
      dt[[col]] <- as.character(dt[[col]])
    }
  }

  # Return as DataFrame (will be written to file by wrapper)
  return(list(result_df = dt))
}

# ===== Main Execution =====

# Read DataFrames from file paths
tryCatch({
  dfs <- read_df_args(df_args, use_parquet)

  # Extract temp directory from payload (Python will provide this)
  temp_dir <- if (!is.null(input$temp_dir)) input$temp_dir else NULL

  # Merge DataFrame arguments with simple arguments
  all_args <- c(func_args, dfs)

  # Execute the requested function
  result <- switch(
    func_name,
    "basic_text_preproc" = do.call(basic_text_preproc, all_args),
    "generate_quantity_binaries" = do.call(generate_quantity_binaries, all_args),
    {
      list(
        error = paste("Unknown function:", func_name),
        status = "error",
        available_functions = c("basic_text_preproc", "generate_quantity_binaries")
      )
    }
  )

  # Check if result has an error status
  if (!is.null(result$status) && result$status == "error") {
    cat(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null"))
    cat("\n")
    flush(stdout())
    quit(save = "no", status = 1, runLast = FALSE)
  }

  # Process result - write DataFrames to files (pass temp_dir)
  final_result <- write_result_dfs(result, use_parquet, temp_dir)

  # Output JSON to stdout
  cat(toJSON(final_result, auto_unbox = TRUE, digits = NA, null = "null"))
  cat("\n")
  flush(stdout())
  flush(stderr())

  # Explicit clean exit
  quit(save = "no", status = 0, runLast = FALSE)

}, error = function(e) {
  error_result <- list(
    error = e$message,
    status = "error",
    function_name = func_name,
    traceback = paste(capture.output(traceback()), collapse = "\n")
  )
  cat(toJSON(error_result, auto_unbox = TRUE, digits = NA, null = "null"))
  cat("\n")
  flush(stdout())
  flush(stderr())
  quit(save = "no", status = 1, runLast = FALSE)
})

quit(save = "no", status = 0, runLast = FALSE)

