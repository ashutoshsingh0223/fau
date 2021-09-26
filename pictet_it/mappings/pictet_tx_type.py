from DataModel.enums import TransactionType

pictet_tx_type_mapping = {
    ('CONTRIBUTION',): TransactionType.CONTRIBUTION,
    ('INTEREST',): TransactionType.INTEREST_INCOME,
    ('ACCOUNT TO ACCOUNT (INT.)',): TransactionType.TRANSFER,
    ('VARIOUS FEES', 'SUBSCRIPTION'): TransactionType.EXPENSE,
    ('SECURITIES EVENT',): TransactionType.SECURITY_CHARGE
}