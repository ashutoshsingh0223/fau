from data_feeds.pictet_it.enums.pictet_file_type import PictetFileType
from datetime import date


def pictet_file_type_from_file_name(file_name: str):
    fn = file_name.lower()
    if 'p3de' in fn:
        return PictetFileType.POSITIONS
    elif 'p3tr' in fn:
        return PictetFileType.TRANSACTIONS
    else:
        raise ValueError(f'Couldn\'t map file name {file_name} to any Pictet-It file type.')


def parse_pictet_date_with_default(d: str, default=None):
    """
    Parses date to ISO format after extracting from pictet-it files
    Args:
        d: Date string from pictet-it files. Format: YYYYMMDD
        default: default date value

    Returns:
        datetime.date object
    """
    try:
        iso_date_str = f'{d[:4]}-{d[5:7]}-{d[8:10]}'
        return date.fromisoformat(iso_date_str)
    except ValueError:
        return default
