import numpy as np


def validate_continuous_years(years: np.ndarray) -> None:
    """
    Validate that all years in a time series are present.

    Parameters
    ----------
    years : np.ndarray
        Array of years to validate.

    Raises
    ------
    ValueError
        If one or more years are missing between the first
        and last year.
    """

    continuous_years = np.arange(
        years.min(),
        years.max() + 1,
    )

    missing_years = continuous_years[~np.isin(continuous_years, years)]

    if missing_years.size != 0:
        raise ValueError(
            f"{', '.join(missing_years.astype(str))} are not present "
            f"between {years.min()} and {years.max()}"
        )
