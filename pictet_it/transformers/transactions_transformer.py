import pandas as pd
from dotmap import DotMap

from data_feeds.common.data_feed_error import DataFeedError
from data_feeds.common.models import Transaction
from data_feeds.pictet_it.enums.pictet_file_type import PictetFileType
from data_feeds.pictet_it.transformers.transactions import transactions_from_transactions_file_row


def transform_transactions_frame(file_type: PictetFileType, file_name: str, df: pd.DataFrame, context: DotMap):
    parsing_functions = {
        PictetFileType.TRANSACTIONS: transactions_from_transactions_file_row,
    }

    if file_type not in parsing_functions:
        return None, None

    # prepare context
    if file_type == PictetFileType.TRANSACTIONS:
        context.txs_by_tx_number = df.set_index('Transaction Number', drop=False)
        context.txs_by_account_and_tx_type = df.set_index(['PORTFOLIO', 'TRANSACTION'],
                                                          drop=False).sort_index()

    transactions = []
    errors = []
    for idx, row in df.iterrows():
        try:
            new_transactions: [Transaction] = parsing_functions[file_type](file_name, idx, row, context)
            transactions.extend([t.to_dict() for t in new_transactions])
        except DataFeedError as e:
            errors.append(e.to_dict())

    transactions_df = pd.DataFrame(transactions) if len(transactions) > 0 else None
    errors_df = pd.DataFrame(errors) if len(errors) > 0 else None

    return transactions_df, errors_df