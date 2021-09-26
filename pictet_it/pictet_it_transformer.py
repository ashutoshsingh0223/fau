import pandas as pd
from datetime import date
from typing import BinaryIO, Optional
from dotmap import DotMap

from utils.csv import to_csv_bytes_buffer
from utils.generic import group_by
from .transformers.transactions_transformer import transform_transactions_frame
from .enums.pictet_file_type import PictetFileType

from .utils.generic import pictet_file_type_from_file_name


def transform_frames_helper(frame_transformer, frames: [(PictetFileType, str, pd.DataFrame)], context: DotMap):
    accumulated_transformed_df: Optional[pd.DataFrame] = None
    accumulated_errors_df: Optional[pd.DataFrame] = None
    for file_type, file_name, df in frames:
        transformed_df, errors_df = frame_transformer(file_type, file_name, df, context)

        if transformed_df is not None:
            accumulated_transformed_df = transformed_df if accumulated_transformed_df is None else accumulated_transformed_df.append(
                transformed_df, ignore_index=True)

        if errors_df is not None:
            accumulated_errors_df = errors_df if accumulated_errors_df is None else accumulated_errors_df.append(
                errors_df, ignore_index=True)

    return accumulated_transformed_df, accumulated_errors_df


def transform_frames(files_date: date, frames: [(PictetFileType, str, pd.DataFrame)], context: DotMap):
    context.files_date = files_date

    print('Transforming transactions frames...')
    transformed_transactions = transform_frames_helper(transform_transactions_frame, frames, context)

    # print('Transforming positions frames...')
    # transformed_positions = transform_frames_helper(transform_positions_frame, frames, context)
    #
    # print('Transforming securities frames...')
    # transformed_securities = transform_frames_helper(transform_securities_frame, frames, context)
    #
    # print('Transforming ownership structure frames...')
    # transformed_ownership_structure = transform_frames_helper(transform_ownership_structure_frame, frames, context)

    # return transformed_transactions, transformed_positions, transformed_securities, transformed_ownership_structure
    return transformed_transactions


def transform_files(files_date: date, files: [(str, BinaryIO)], context: DotMap):
    grouped_by_pi_file_type = group_by(files, lambda f: pictet_file_type_from_file_name(f[0]))

    frames = []

    for pi_file_type, ps in grouped_by_pi_file_type.items():
        dfs = [(pi_file_type, p[0], pd.read_csv(p[1], dtype=str).fillna('')) for p in ps]
        frames.extend(dfs)

    transformed_frames = transform_frames(files_date, frames, context)

    transformed_transactions = transformed_frames

    # transformed_transactions, transformed_positions, transformed_securities, transformed_ownership_structure = transformed_frames

    transformed_transactions_export, transformed_transactions_errors = transformed_transactions
    # transformed_positions_export, transformed_positions_errors = transformed_positions
    # transformed_securities_export, transformed_securities_errors = transformed_securities
    # transformed_ownership_structure_export, transformed_ownership_structure_errors = transformed_ownership_structure

    # export to excel
    transformed_transactions_export_buffer = None
    if transformed_transactions_export is not None:
        transformed_transactions_export_buffer = to_csv_bytes_buffer(transformed_transactions_export)

    transformed_transactions_errors_buffer = None
    if transformed_transactions_errors is not None:
        transformed_transactions_errors_buffer = to_csv_bytes_buffer(transformed_transactions_errors)

    # transformed_positions_export_buffer = None
    # if transformed_positions_export is not None:
    #     transformed_positions_export_buffer = to_csv_bytes_buffer(transformed_positions_export)
    #
    # transformed_positions_errors_buffer = None
    # if transformed_positions_errors is not None:
    #     transformed_positions_errors_buffer = to_csv_bytes_buffer(transformed_positions_errors)
    #
    # transformed_securities_export_buffer = None
    # if transformed_securities_export is not None:
    #     transformed_securities_export_buffer = to_csv_bytes_buffer(transformed_securities_export)
    #
    # transformed_securities_errors_buffer = None
    # if transformed_securities_errors is not None:
    #     transformed_securities_errors_buffer = to_csv_bytes_buffer(transformed_securities_errors)
    #
    # transformed_ownership_structure_export_buffer = None
    # if transformed_ownership_structure_export is not None:
    #     transformed_ownership_structure_export_buffer = to_csv_bytes_buffer(transformed_ownership_structure_export)
    #
    # transformed_ownership_structure_errors_buffer = None
    # if transformed_ownership_structure_errors is not None:
    #     transformed_ownership_structure_errors_buffer = to_csv_bytes_buffer(transformed_ownership_structure_errors)

    return (transformed_transactions_export_buffer, transformed_transactions_errors_buffer)
           # (transformed_positions_export_buffer, transformed_positions_errors_buffer), \
           # (transformed_securities_export_buffer, transformed_securities_errors_buffer), \
           # (transformed_ownership_structure_export_buffer, transformed_ownership_structure_errors_buffer)
