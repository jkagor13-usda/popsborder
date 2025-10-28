# add_numbers.R
add_numbers <- function(a, b) {
  return(a+a+b)
}

# Call the function with command line arguments
args <- commandArgs(trailingOnly = TRUE)
a <- as.numeric(args[1])
b <- as.numeric(args[2])

result <- add_numbers(a, b)
cat(result, "\n")