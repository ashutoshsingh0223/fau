from decimal import Decimal

import pandas as pd

from DataModel.enums import TransactionType, FeeTaxExpenseType, AssetModelType, CorporateActionType
from data_feeds.common.data_feed_error import DataFeedError, DataFeedSource
from data_feeds.common.models import Transaction
from data_feeds.common.utils.generic import parse_currency_with_default
from data_feeds.jpm.enums import JPMErrorType
from data_feeds.jpm.mappings.jpm_asset_model_type import asset_model_type_mapping
from data_feeds.jpm.mappings.jpm_corporate_action_type import jpm_corporate_action_type_mapping
from data_feeds.jpm.mappings.jpm_fte_tags import jpm_fte_tags_mapping
from data_feeds.jpm.mappings.jpm_fte_tx_type import jpm_fte_tx_type_mapping
from data_feeds.jpm.mappings.jpm_fte_type import jpm_fte_mapping
from data_feeds.jpm.utils.generic import parse_jpm_date_with_default
from utils.generic import to_decimal_with_default


def transactions_from_transactions_file_row(file_path: str, line_number: int, row: pd.Series, context):
    transactions = []

    # skip excluded vendor ids
    account_number = row['PORTFOLIO']
    if any([v_id in context.excluded_vendor_ids for v_id in [account_number]]):
        return None

    tx_type_raw = row['TRANSACTION']
    type_of_transaction = row['TRANSACTION DESCRIPTION']
    instrument_type_code = row['Instrument Type Code']

    if type_of_transaction in {'FX FORWARD BUY', 'FX FORWARD SALE'}:
        # TODO implement FX Forward TXs
        return []

    if tx_type_raw in {'TPE', 'TPS', 'EDA'}:
        # skip corporate act block in/outflow TXs, technical code edr increase
        return []

    if instrument_type_code == '717':  # skip pledges
        return []

    parse_functions = {
        'VCT': parse_buy_sell_tx,
        'ACT': parse_buy_sell_tx,
        'DIV': parse_cash_distribution_tx,
        'INT': parse_interest_income_expense_tx,
        'CPS': parse_coupon_tx,
        'N': parse_buy_sell_tx,
        'L': parse_buy_sell_tx,
        'I': parse_interest_income_expense_tx,
        'VFI': parse_buy_sell_tx,
        'SFI': parse_contribution_distribution_tx,
        'TXA': parse_contribution_distribution_tx,
        'TXV': parse_contribution_distribution_tx,
        'AXA': parse_contribution_distribution_tx,
        'AXV': parse_contribution_distribution_tx,
        'ARA': parse_buy_sell_tx,
        'ARV': parse_buy_sell_tx,
        '100': parse_jpm_tx_code_100_102_103,
        '101': parse_deposit_withdrawal_tx,
        '102': parse_jpm_tx_code_100_102_103,
        '103': parse_jpm_tx_code_100_102_103,
        '110': parse_deposit_withdrawal_tx,
        '120': parse_deposit_withdrawal_tx,
        '129': parse_jpm_tx_code_129,
        '200': parse_fx_spot_tx,
        '198': parse_debit_credit_tx,
        '898': parse_debit_credit_tx,
        '702': parse_account_fee_account_expense_withholding_tax_tx,
        '804': parse_account_fee_account_expense_withholding_tax_tx,
        '805': parse_account_fee_account_expense_withholding_tax_tx,
        '807': parse_account_fee_account_expense_withholding_tax_tx,
        '811': parse_account_fee_account_expense_withholding_tax_tx,
        '812': parse_account_fee_account_expense_withholding_tax_tx,
        '816': parse_account_fee_account_expense_withholding_tax_tx,
        '817': parse_account_fee_account_expense_withholding_tax_tx,
        'TRE': parse_transfer_in_out_tx,
        'TRS': parse_transfer_in_out_tx,
        'CDE': parse_security_charge_discharge_tx,
        'SOR': parse_security_charge_discharge_tx,
        'ECE': parse_security_charge_discharge_tx,
        'ECS': parse_security_charge_discharge_tx,
        'EDE': parse_security_charge_discharge_tx,
        'EDS': parse_security_charge_discharge_tx,
        'ASE': parse_security_charge_discharge_tx,
        'ASS': parse_security_charge_discharge_tx,
        'HFE': parse_security_charge_discharge_tx,
    }

    if tx_type_raw not in parse_functions:
        raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                            JPMErrorType.TYPE_NOT_HANDLED.value,
                            f'Error parsing transactions row. TX type {tx_type_raw} has not been implemented yet.')

    parsed_txs = parse_functions[tx_type_raw](file_path, line_number, row, context)
    transactions.extend(parsed_txs)

    return transactions


# Generic Transaction
def parse_generic_tx(tx_type: TransactionType, file_path: str, line_number: int, row: pd.Series, context):
    transaction = Transaction(file_path, line_number)

    tx_type_raw = row['Olympic Transaction Code']
    transaction.uid = f'{row["Transaction Number"]}.{tx_type_raw}'  # not unique

    transaction.type_id = tx_type.value
    transaction.type_name = tx_type.name

    transaction.trade_date = parse_jpm_date_with_default(row['Trade Date'], '')
    transaction.settle_date = parse_jpm_date_with_default(row['Value Date'], '')
    transaction.insert_date = context.files_date
    transaction.received_date = context.files_date

    transaction.owner_uid = row['Account Number']
    transaction.direct_owner_1_uid = transaction.owner_uid

    type_of_transaction = row['Type Of Transaction']
    if type_of_transaction == 'CONTRACT':
        asset_type = AssetModelType.LOAN
    else:
        asset_type = asset_model_type_mapping.get((row['Instrument Type Code'],), '')
    if asset_type != '':
        transaction.asset_type = asset_type.value

    if asset_type == AssetModelType.CASH_ACCOUNT:
        transaction.asset_currency = parse_currency_with_default(row['Cash Currency'], '')
        transaction.asset_name = f'Cash Account {transaction.direct_owner_1_uid}.{transaction.asset_currency}'
    elif asset_type == AssetModelType.TIME_DEPOSIT:
        transaction.asset_currency = parse_currency_with_default(row['Cash Currency'], '')
        transaction.asset_name = f'Time Deposit {transaction.direct_owner_1_uid}.{transaction.asset_currency}.TD'
    else:
        transaction.asset_currency = parse_currency_with_default(row['Pricing Currency'], '')
        transaction.asset_name = row['Instrument Short Name']

    contract_id = row['Contract ID']
    valor = row['Valoren']

    if asset_type in {AssetModelType.CASH_ACCOUNT, AssetModelType.TIME_DEPOSIT}:
        transaction.asset_uid = f'{transaction.direct_owner_1_uid}.{transaction.asset_currency}'
        if asset_type == AssetModelType.TIME_DEPOSIT:
            transaction.asset_uid += '.TD'
    elif asset_type == AssetModelType.LOAN:
        transaction.asset_uid = f'{contract_id}.CONTRACT'
    elif valor:
        transaction.asset_uid = f'{valor}_{transaction.asset_currency}'
    else:
        raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                            JPMErrorType.MISSING_FIELD_ERROR.value,
                            f'Error parsing transactions row. Asset UID is missing.')

    transaction.description = f'{row["Transaction Description"]} - {transaction.asset_name}'
    bb_ticker_or_isin = row['Ticker'] or row['ISIN']
    if bb_ticker_or_isin:
        transaction.description += f' ({bb_ticker_or_isin})'

    transaction.comments = ''
    transaction.tags = []
    transaction.vendor_ids = [row['Transaction Number']]

    reversal_flag = row['Reversal Flag']
    reversal_reference = row['Reversal Reference']

    if reversal_flag != '0' or reversal_reference != '0':
        transaction.cancelled_tx_uid = reversal_reference

    return transaction


def get_tx_type_from_col_sign(row: pd.Series, col: str, type_positive: TransactionType, type_negative: TransactionType):
    is_cancellation = row['Reversal Flag'] != '0' or row['Reversal Reference'] != '0'
    positive_cash_amount = to_decimal_with_default(row[col]) >= 0
    if positive_cash_amount and not is_cancellation or not positive_cash_amount and is_cancellation:
        return type_positive
    else:
        return type_negative


def set_cash_impacting_fields_from_row(transaction: Transaction, row: pd.Series, context, invert_amounts=False):
    transaction.cash_amount_1 = to_decimal_with_default(row['Gross Transaction Amount'], '')
    transaction.cash_currency_1 = parse_currency_with_default(row['Cash Currency'], '')
    transaction.cash_account_1 = f'{transaction.direct_owner_1_uid}.{transaction.cash_currency_1}'
    if transaction.asset_type == AssetModelType.TIME_DEPOSIT.value:
        transaction.cash_account_1 += '.TD'
    transaction.original_value = transaction.cash_amount_1
    transaction.original_currency = parse_currency_with_default(row['Transaction Currency'], '')
    transaction.cash_fx_rate = Decimal(1)
    transaction.total_cash_amount = to_decimal_with_default(row['Cash Amount'], '')
    transaction.affects_performance = ''  # Waiting for JPM reply

    if invert_amounts:
        transaction.cash_amount_1 *= -1
        transaction.original_value *= -1


def parse_fees_taxes_expenses_from_row(parent_tx: Transaction, file_path: str, line_number: int, row: pd.Series, context):
    ftes = []

    fte_entries = {
        'BRKRG_FEE': (TransactionType.FEE,
                      FeeTaxExpenseType.BROKERAGE_FEE,
                      ['Brokerage Fee'],
                      row['Cash Currency'],
                      row['Brokerage Fees'],
                      True),
        'CNTRPRTY_FEE': (TransactionType.FEE,
                         FeeTaxExpenseType.THIRD_PARTY_FEE,
                         ['Third Party Fee'],
                         row['Cash Currency'],
                         row['Counterparty Fees'],
                         True),
        'JPM_BRKRG_FEE': (TransactionType.FEE,
                          FeeTaxExpenseType.BROKERAGE_FEE,
                          ['Brokerage Fee'],
                          row['Cash Currency'],
                          row['JPM Brokerage Fees'],
                          True),
        'JPM_FEE': (TransactionType.FEE,
                    FeeTaxExpenseType.BANK_FEE,
                    ['Bank Fee'],
                    row['Cash Currency'],
                    row['JPM Fees'],
                    True),
        'SWSS_TAX': (TransactionType.TAX,
                     FeeTaxExpenseType.LOCAL_TAX,
                     ['Swiss Tax'],
                     row['Cash Currency'],
                     row['Swiss Tax'],
                     True),
        'WTHHLDNG_TAX': (TransactionType.TAX,
                         FeeTaxExpenseType.WITHHOLDING_TAX,
                         ['Withholding Tax'],
                         row['Cash Currency'],
                         row['Withholding Tax'],
                         False),
    }

    for entry_name, (tx_type, fee_tax_expense_type, tags, currency, amount, affects_performance) in fte_entries.items():
        entry_currency = parse_currency_with_default(currency, '')
        entry_amount = to_decimal_with_default(amount, '')
        if entry_currency and entry_amount:
            fte_tx = Transaction(file_path, line_number)
            fte_tx.uid = f'{parent_tx.uid}.{entry_name}'
            fte_tx.parent_tx_uid = parent_tx.uid
            fte_tx.vendor_ids = parent_tx.vendor_ids

            fte_tx.type_id = tx_type.value
            fte_tx.type_name = tx_type.name
            fte_tx.fee_tax_expense_type_id = fee_tax_expense_type.value
            fte_tx.fee_tax_expense_type_name = fee_tax_expense_type.name

            fte_tx.trade_date = parent_tx.trade_date
            fte_tx.settle_date = parent_tx.settle_date
            fte_tx.insert_date = parent_tx.insert_date
            fte_tx.received_date = parent_tx.received_date

            fte_tx.asset_uid = parent_tx.asset_uid
            fte_tx.asset_type = parent_tx.asset_type
            fte_tx.asset_name = parent_tx.asset_name
            fte_tx.asset_currency = parent_tx.asset_currency

            fte_tx.owner_uid = parent_tx.owner_uid
            fte_tx.direct_owner_1_uid = parent_tx.direct_owner_1_uid

            fte_tx.cash_amount_1 = -entry_amount
            fte_tx.cash_currency_1 = entry_currency
            fte_tx.cash_account_1 = f'{fte_tx.direct_owner_1_uid}.{entry_currency}'
            if fte_tx.asset_type == AssetModelType.TIME_DEPOSIT.value:
                fte_tx.cash_account_1 += '.TD'
            fte_tx.original_value = fte_tx.cash_amount_1
            fte_tx.original_currency = fte_tx.cash_currency_1
            fte_tx.cash_fx_rate = Decimal(1)
            fte_tx.affects_performance = affects_performance

            fte_tx.description = f'{parent_tx.description} - {fee_tax_expense_type.name.replace("_", " ").title()}'
            fte_tx.tags = tags

            if parent_tx.cancelled_tx_uid:
                fte_tx.cancelled_tx_uid = f'{parent_tx.cancelled_tx_uid}.{entry_name}'

            ftes.append(fte_tx)
        elif entry_amount and not entry_currency:
            raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                                JPMErrorType.MISSING_FIELD_ERROR.value,
                                f'Error parsing transactions row. There is a {entry_name} entry without currency.')

    return ftes


def parse_corporate_action_from_row(corporate_action_type: CorporateActionType,
                                    file_path: str, line_number: int, row: pd.Series, context):
    tx_type = TransactionType.CORPORATE_ACTION
    corporate_action = parse_generic_tx(tx_type, file_path, line_number, row, context)

    corporate_action.uid += '.CORP_ACTION'

    corporate_action.corporate_action_type_id = corporate_action_type.value
    corporate_action.corporate_action_type_name = corporate_action_type.name

    corporate_action.ex_date = parse_jpm_date_with_default(row['Ex Date'], '')

    corporate_action_type_str = corporate_action_type.name.replace('_', ' ').title()
    corporate_action.description = f'{corporate_action.description} - {corporate_action_type_str}'
    corporate_action.tags = [corporate_action_type_str]

    return corporate_action


def parse_buy_sell_tx(file_path: str, line_number: int, row: pd.Series, context):
    tx_type = get_tx_type_from_col_sign(row, 'Cash Amount', TransactionType.SELL, TransactionType.BUY)
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context, invert_amounts=True)

    transaction.quantity_1 = to_decimal_with_default(row['Quantity'], '')
    transaction.value_per_share = to_decimal_with_default(row['Transaction Price'], '')
    if transaction.value_per_share != '' and transaction.asset_type == AssetModelType.BOND.value:
        transaction.value_per_share /= 100
    elif transaction.asset_type == AssetModelType.LOAN.value:
        transaction.value_per_share = Decimal(1)
    transaction.accrued_income = to_decimal_with_default(row['Accrued Interest'], '')
    if transaction.accrued_income != '':
        transaction.accrued_income = abs(transaction.accrued_income)
    elif transaction.asset_type == AssetModelType.LOAN.value:
        transaction.accrued_income = Decimal(0)
    transaction.multiplier = Decimal(1)

    if row['Olympic Transaction Code'] == 'VFI':
        transaction.tags.append('Redemption')

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    return transactions


def parse_interest_income_expense_tx(file_path: str, line_number: int, row: pd.Series, context):
    type_of_transaction = row['Type Of Transaction']
    if type_of_transaction == 'CONTRACT':
        asset_type = AssetModelType.LOAN
    else:
        asset_type = asset_model_type_mapping.get((row['Instrument Type Code'],), '')

    tx_type = get_tx_type_from_col_sign(row, 'Cash Amount', TransactionType.INTEREST_INCOME, TransactionType.INTEREST_EXPENSE)
    if asset_type == AssetModelType.TIME_DEPOSIT:
        tx_type = get_tx_type_from_col_sign(row, 'Cash Amount', TransactionType.INTEREST_INCOME_SHARES, TransactionType.INTEREST_EXPENSE_SHARES)

    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)

    if tx_type in {TransactionType.INTEREST_INCOME, TransactionType.INTEREST_EXPENSE}:
        set_cash_impacting_fields_from_row(transaction, row, context)
    else:
        transaction.quantity_1 = to_decimal_with_default(row['Cash Amount'], '')
        transaction.value_per_share = Decimal(1)

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    return transactions


def parse_coupon_tx(file_path: str, line_number: int, row: pd.Series, context):
    tx_type = TransactionType.COUPON
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context)

    transaction.value_per_share = to_decimal_with_default(row['Transaction Price'], '')
    if transaction.value_per_share != '' and transaction.asset_type == AssetModelType.BOND.value:
        transaction.value_per_share /= 100
    transaction.ex_date = parse_jpm_date_with_default(row['Ex Date'], '')

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    return transactions


def parse_cash_distribution_tx(file_path: str, line_number: int, row: pd.Series, context):
    tx_type = TransactionType.CASH_DISTRIBUTION
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context)

    transaction.quantity_1 = to_decimal_with_default(row['Quantity'], to_decimal_with_default(row['Current Face'], ''))
    transaction.value_per_share = to_decimal_with_default(row['Transaction Price'] or row['Dividend/Coupon Rate'], '')
    transaction.ex_date = parse_jpm_date_with_default(row['Ex Date'], '')

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    tx_type_raw = row['Olympic Transaction Code']
    if tx_type_raw == 'DIV':
        corporate_action_type = CorporateActionType.DIVIDEND
    else:
        raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                            JPMErrorType.TYPE_NOT_HANDLED.value,
                            f'Error parsing transactions row. Cash distribution of type {tx_type_raw} has no corresponding corporate action type, yet.')

    transaction.tags = [corporate_action_type.name.replace('_', ' ').title()]

    corporate_action = parse_corporate_action_from_row(corporate_action_type, file_path, line_number, row, context)
    if corporate_action is not None:
        transaction.grouping_tx_uid = corporate_action.uid
        transactions.insert(0, corporate_action)

    return transactions


def parse_deposit_withdrawal_tx(file_path: str, line_number: int, row: pd.Series, context):
    tx_type = get_tx_type_from_col_sign(row, 'Cash Amount', TransactionType.DEPOSIT, TransactionType.WITHDRAWAL)
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context)

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    return transactions


def parse_debit_credit_tx(file_path: str, line_number: int, row: pd.Series, context):
    tx_type = get_tx_type_from_col_sign(row, 'Cash Amount', TransactionType.CREDIT, TransactionType.DEBIT)
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context)

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    return transactions


def parse_cash_transfer_tx(file_path: str, line_number: int, row: pd.Series, context):
    # skip long side
    if to_decimal_with_default(row['Cash Amount']) > 0:
        return []

    tx_type = TransactionType.CASH_TRANSFER
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context)

    # find long side
    txs_by_tx_number: pd.DataFrame = context.txs_by_tx_number
    linked_txs: pd.DataFrame = txs_by_tx_number.loc[row['Transaction Number']]
    long_row = linked_txs[linked_txs['Cash Amount'] != row['Cash Amount']].iloc[0]

    transaction.direct_owner_2_uid = long_row['Account Number']
    transaction.cash_amount_2 = to_decimal_with_default(long_row['Cash Amount'], '')
    transaction.cash_currency_2 = parse_currency_with_default(long_row['Cash Currency'], '')
    transaction.cash_account_2 = f'{transaction.direct_owner_2_uid}.{transaction.cash_currency_2}'

    return [transaction]


def parse_account_fee_account_expense_withholding_tax_tx(file_path: str, line_number: int, row: pd.Series, context):
    tx_type_raw = row['Olympic Transaction Code']
    tx_type = jpm_fte_tx_type_mapping.get((tx_type_raw,))
    if tx_type is None:
        raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                            JPMErrorType.TYPE_NOT_HANDLED.value,
                            f'Error parsing transactions row. TX with code {tx_type_raw} has not been mapped to either: Account Fee, Account Expense, Withholding Tax.')

    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context)

    fte_type = jpm_fte_mapping.get((tx_type_raw,), None)
    if fte_type is None:
        raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                            JPMErrorType.TYPE_NOT_HANDLED.value,
                            f'Error parsing transactions row. {tx_type.name} TX with code {tx_type_raw} has no F/T/E mapping, yet.')

    transaction.fee_tax_expense_type_id = fte_type.value
    transaction.fee_tax_expense_type_name = fte_type.name

    fte_tags = jpm_fte_tags_mapping.get((tx_type_raw,), None)
    if fte_type is None:
        raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                            JPMErrorType.TYPE_NOT_HANDLED.value,
                            f'Error parsing transactions row. {tx_type.name} TX with code {tx_type_raw} has no F/T/E tags mapping, yet.')
    transaction.tags.extend(fte_tags)

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    return transactions


def parse_fx_spot_tx(file_path: str, line_number: int, row: pd.Series, context):
    # skip long side
    if to_decimal_with_default(row['Cash Amount']) > 0:
        return []

    tx_type = TransactionType.FX_SPOT
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context)

    # find long side
    txs_by_tx_number: pd.DataFrame = context.txs_by_tx_number
    linked_txs: pd.DataFrame = txs_by_tx_number.loc[row['Transaction Number']]
    long_row = linked_txs[linked_txs['Cash Amount'] != row['Cash Amount']].iloc[0]

    transaction.direct_owner_2_uid = long_row['Account Number']
    transaction.cash_amount_1 = to_decimal_with_default(row['Cash Amount'], '')
    transaction.cash_currency_1 = parse_currency_with_default(row['Cash Currency'], '')
    transaction.cash_amount_2 = to_decimal_with_default(long_row['Cash Amount'], '')
    transaction.cash_currency_2 = parse_currency_with_default(long_row['Cash Currency'], '')
    transaction.cash_account_2 = f'{transaction.direct_owner_2_uid}.{transaction.cash_currency_2}'
    transaction.original_value = transaction.cash_amount_1
    transaction.original_currency = transaction.cash_currency_1
    transaction.total_cash_amount = ''

    transaction.cash_fx_rate = to_decimal_with_default(transaction.cash_amount_2, Decimal(1)) / to_decimal_with_default(transaction.cash_amount_1, Decimal(1))
    transaction.cash_fx_rate = round(abs(transaction.cash_fx_rate), 10)

    transaction.asset_type = AssetModelType.FX_RATE.value
    transaction.asset_name = f'{transaction.cash_currency_2} / {transaction.cash_currency_1}'
    transaction.asset_uid = f'{transaction.cash_currency_2}_{transaction.cash_currency_1}'
    transaction.asset_currency = transaction.cash_currency_2

    return [transaction]


def parse_contribution_distribution_tx(file_path: str, line_number: int, row: pd.Series, context):
    if to_decimal_with_default(row['Transaction Price']) != 1:
        # this is probably a share based fund, parse as a normal buy/sell
        return parse_buy_sell_tx(file_path, line_number, row, context)

    tx_type = get_tx_type_from_col_sign(row, 'Cash Amount', TransactionType.DISTRIBUTION, TransactionType.CONTRIBUTION)
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)
    set_cash_impacting_fields_from_row(transaction, row, context, invert_amounts=True)

    transaction.quantity_1 = to_decimal_with_default(row['Quantity'], '')
    transaction.value_per_share = to_decimal_with_default(row['Transaction Price'], '')
    transaction.accrued_income = Decimal(0)
    transaction.multiplier = Decimal(1)

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    return transactions


def parse_transfer_in_out_tx(file_path: str, line_number: int, row: pd.Series, context):
    tx_type = get_tx_type_from_col_sign(row, 'Quantity', TransactionType.TRANSFER_IN, TransactionType.TRANSFER_OUT)
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)

    transaction.quantity_1 = to_decimal_with_default(row['Quantity'], '')
    transaction.value_per_share = to_decimal_with_default(row['Transaction Price'], '')
    if transaction.value_per_share != '' and transaction.asset_type == AssetModelType.BOND.value:
        transaction.value_per_share /= 100
    transaction.accrued_income = to_decimal_with_default(row['Accrued Interest'], '')
    if transaction.accrued_income != '':
        transaction.accrued_income = abs(transaction.accrued_income)
    transaction.multiplier = Decimal(1)

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    return transactions


def parse_security_charge_discharge_tx(file_path: str, line_number: int, row: pd.Series, context):
    tx_type = get_tx_type_from_col_sign(row, 'Quantity', TransactionType.SECURITY_CHARGE, TransactionType.SECURITY_DISCHARGE)
    transaction = parse_generic_tx(tx_type, file_path, line_number, row, context)

    if not to_decimal_with_default(row['Cash Amount']).is_zero():
        set_cash_impacting_fields_from_row(transaction, row, context, invert_amounts=True)

    transaction.quantity_1 = to_decimal_with_default(row['Quantity'], '')
    transaction.value_per_share = to_decimal_with_default(row['Transaction Price'], '')
    if transaction.value_per_share != '' and transaction.asset_type == AssetModelType.BOND.value:
        transaction.value_per_share /= 100
    transaction.accrued_income = to_decimal_with_default(row['Accrued Interest'], '')
    if transaction.accrued_income != '':
        transaction.accrued_income = abs(transaction.accrued_income)
    transaction.multiplier = Decimal(1)

    transaction.ex_date = parse_jpm_date_with_default(row['Ex Date'], '')

    transactions = [transaction]

    ftes = parse_fees_taxes_expenses_from_row(transaction, file_path, line_number, row, context)
    transactions.extend(ftes)

    # create grouping TX if necessary
    tx_type_raw = row['Olympic Transaction Code']
    corporate_action_type = jpm_corporate_action_type_mapping.get((tx_type_raw,))
    if corporate_action_type is None:
        raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                            JPMErrorType.TYPE_NOT_HANDLED.value,
                            f'Error parsing transactions row. {tx_type.name} TX with code {tx_type_raw} has not been mapped to a corporate action type, yet.')

    should_create_grouping_tx = tx_type_raw not in {'ECS', 'EDS', 'ASS'}
    if should_create_grouping_tx:
        corporate_action = parse_corporate_action_from_row(corporate_action_type, file_path, line_number, row, context)
        transaction.grouping_tx_uid = corporate_action.uid
        transaction.tags.extend(corporate_action.tags)
        transactions.insert(0, corporate_action)
    else:
        # find the main mechanic TX of the corporate action and get the grouping TX from there
        txs_by_account_and_tx_type: pd.DataFrame = context.txs_by_account_and_tx_type

        mechanic_counterpart_mapping = {'ECS': 'ECE', 'EDS': 'EDE', 'ASS': 'ASE'}
        mechanic_counterpart = mechanic_counterpart_mapping.get(tx_type_raw)
        if mechanic_counterpart is not None:
            linked_txs: pd.DataFrame = txs_by_account_and_tx_type.loc[(row['Account Number'], mechanic_counterpart)]

            if isinstance(linked_txs, pd.Series):
                linked_txs = pd.DataFrame([linked_txs])

            if linked_txs.shape[0] == 1:
                other_row = linked_txs.iloc[0]
                transaction.grouping_tx_uid = f'{other_row["Transaction Number"]}.{other_row["Olympic Transaction Code"]}.CORP_ACTION'
            else:
                # there is more than one mechanic, sort by distance from tx id, to find the closest
                # if there are still more then one TX, leave it blank
                # these assumes that they won't be interleaved with other corporate actions
                linked_txs = linked_txs.copy()
                tx_as_decimal = to_decimal_with_default(transaction.uid)
                linked_txs['TX ID Distance'] = linked_txs.apply(lambda r: abs(to_decimal_with_default(r['Transaction Number']) - tx_as_decimal), axis=1)
                linked_txs.sort_values('TX ID Distance', inplace=True)

                if linked_txs.iloc[0]['TX ID Distance'] != linked_txs.iloc[1]['TX ID Distance']:
                    transaction.grouping_tx_uid = f'{linked_txs.iloc[0]["Transaction Number"]}.{linked_txs.iloc[0]["Olympic Transaction Code"]}.CORP_ACTION'

    return transactions


def parse_jpm_tx_code_100_102_103(file_path: str, line_number: int, row: pd.Series, context):
    # these TXs could have two opposite legs to represent a giro/cash transfer
    # when they do, they usually have the same Transaction Number
    # some times it seems that they don't, but we'll leave those as separate Withdrawal / Deposit TXs for now
    txs_by_tx_number: pd.DataFrame = context.txs_by_tx_number

    linked_txs: pd.DataFrame = txs_by_tx_number.loc[row['Transaction Number']]

    if isinstance(linked_txs, pd.Series):
        return parse_deposit_withdrawal_tx(file_path, line_number, row, context)
    elif linked_txs['Cash Currency'].unique().shape[0] > 1:
        return parse_fx_spot_tx(file_path, line_number, row, context)
    else:
        return parse_cash_transfer_tx(file_path, line_number, row, context)


def parse_jpm_tx_code_129(file_path: str, line_number: int, row: pd.Series, context):
    # these TXs put together more than one cash debit/credit with the same TX ID
    # we separate them by appending a number to each debit/credit TX UID
    txs_by_tx_number: pd.DataFrame = context.txs_by_tx_number

    linked_txs: pd.DataFrame = txs_by_tx_number.loc[row['Transaction Number']]

    if isinstance(linked_txs, pd.Series):
        linked_txs = pd.DataFrame([linked_txs])

    if linked_txs.shape[0] == 1:
        return parse_deposit_withdrawal_tx(file_path, line_number, row, context)
    else:
        # append the line_number to distinguish the single TXs
        txs = parse_deposit_withdrawal_tx(file_path, line_number, row, context)
        for tx in txs:
            if tx.parent_tx_uid == '':
                tx.uid = f'{tx.uid}.{line_number}'
            else:
                tx.parent_tx_uid = f'{tx.parent_tx_uid}.{line_number}'
        return txs
