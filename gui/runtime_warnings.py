import warnings


def suppress_optional_dependency_warnings() -> None:
    """Hide known pandas optional-dependency version warnings in the app runtime."""
    warnings.filterwarnings(
        "ignore",
        message=r"Pandas requires version '2\.8\.4' or newer of 'numexpr'.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Pandas requires version '1\.3\.6' or newer of 'bottleneck'.*",
        category=UserWarning,
    )
