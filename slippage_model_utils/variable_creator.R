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
