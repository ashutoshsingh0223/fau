from dotmap import DotMap

if __name__ == '__main__':
    from Brightside import setup
    setup()

import json
from pathlib import Path
import os
from datetime import date

import boto3
from django.conf import settings

from utils.aws import iterate_s3_folder
from utils.generic import save_bytes_io_to_file, get_secret, group_by

from data_feeds.pictet_it.utils.generic import parse_pictet_date_with_default


SOURCE_FOLDER_PATH = Path('./data/pictet-it/cleaned/')
DESTINATION_FOLDER_PATH = Path('./data/import/pictet-it')
DESTINATION_FOLDER_PATH.mkdir(parents=True, exist_ok=True)


def date_from_path_name(p: Path):
    d = os.path.basename(p)[6:16]
    return parse_pictet_date_with_default(d)


def retrieve_file_names():
    if settings.USE_S3:
        sftp_bucket_info = json.loads(get_secret('aws/s3/sftp_bucket_info'))

        s3 = boto3.resource('s3', region_name=sftp_bucket_info['region'])

        source_file_paths = list(iterate_s3_folder(sftp_bucket_info['bucket_name'],
                                                   prefix='pictet-it/cleaned/',
                                                   filter_func=lambda p: os.path.splitext(p.name)[1] == '.csv',
                                                   s3=s3))
    else:
        source_file_paths = [p for p in SOURCE_FOLDER_PATH.iterdir() if os.path.splitext(p.name)[1] == '.csv']

    return group_by(sorted(source_file_paths, key=date_from_path_name), date_from_path_name)


def main():
    file_paths_by_date = retrieve_file_names()

    context = DotMap()
    context.excluded_vendor_ids = set()

    for files_date, file_paths in file_paths_by_date.items():
        transformed_files = transform_files(files_date, [(p.name, p.open('r')) for p in file_paths], context)

        transactions, positions, securities, ownership_structure = transformed_files

        transactions_buffer, transactions_errors_buffer = transactions
        positions_buffer, positions_errors_buffer = positions
        securities_buffer, securities_errors_buffer = securities
        ownership_structure_buffer, ownership_structure_errors_buffer = ownership_structure

        output_folder = DESTINATION_FOLDER_PATH

        if transactions_buffer:
            save_bytes_io_to_file(transactions_buffer, output_folder / f'{files_date}_transactions.csv')
        if transactions_errors_buffer:
            save_bytes_io_to_file(transactions_errors_buffer, output_folder / f'{files_date}_transactions_errors.csv')

        # if positions_buffer:
        #     save_bytes_io_to_file(positions_buffer, output_folder / f'{files_date}_positions.csv')
        # if positions_errors_buffer:
        #     save_bytes_io_to_file(positions_errors_buffer, output_folder / f'{files_date}_positions_errors.csv')
        #
        # if securities_buffer:
        #     save_bytes_io_to_file(securities_buffer, output_folder / f'{files_date}_securities.csv')
        # if securities_errors_buffer:
        #     save_bytes_io_to_file(securities_errors_buffer, output_folder / f'{files_date}_securities_errors.csv')
        #
        # if ownership_structure_buffer:
        #     save_bytes_io_to_file(ownership_structure_buffer, output_folder / f'{files_date}_ownership_structure.csv')
        # if ownership_structure_errors_buffer:
        #     save_bytes_io_to_file(ownership_structure_errors_buffer,
        #                           output_folder / f'{files_date}_ownership_structure_errors.csv')


if __name__ == '__main__':
    main()
