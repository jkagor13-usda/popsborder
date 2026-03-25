import warnings


warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '2\.8\.4' or newer of 'numexpr'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '1\.3\.6' or newer of 'bottleneck'",
    category=UserWarning,
)
