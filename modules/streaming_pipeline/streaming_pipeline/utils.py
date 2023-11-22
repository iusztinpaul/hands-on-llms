import datetime
from typing import List, Tuple


def read_requirements(file_path: str) -> List[str]:
    """
    Reads a file containing a list of requirements and returns them as a list of strings.

    Args:
        file_path (str): The path to the file containing the requirements.

    Returns:
        List[str]: A list of requirements as strings.
    """

    with open(file_path, "r") as file:
        requirements = [line.strip() for line in file if line.strip()]

    return requirements


def split_time_range_into_intervals(
    from_datetime: datetime.datetime, to_datetime: datetime.datetime, n: int
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Splits a time range [from_datetime, to_datetime] into N equal intervals.

    Args:
        from_datetime (datetime): The starting datetime object.
        to_datetime (datetime): The ending datetime object.
        n (int): The number of intervals.

    Returns:
        List of tuples: A list where each tuple contains the start and end datetime objects for each interval.
    """

    # Calculate total duration between from_datetime and to_datetime.
    total_duration = to_datetime - from_datetime

    # Calculate the length of each interval.
    interval_length = total_duration / n

    # Generate the interval.
    intervals = []
    for i in range(n):
        interval_start = from_datetime + (i * interval_length)
        interval_end = from_datetime + ((i + 1) * interval_length)
        if i + 1 != n:
            # Subtract 1 microsecond from the end of each interval to avoid overlapping.
            interval_end = interval_end - datetime.timedelta(minutes=1)

        intervals.append((interval_start, interval_end))

    return intervals
