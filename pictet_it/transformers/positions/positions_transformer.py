import pandas as pd
import pycountry

from data_feeds.common.data_feed_error import DataFeedError
from data_feeds.common.models import Position
from data_feeds.jpm.mappings.jpm_asset_model_type import asset_model_type_mapping
from utils.generic import default_if_none
from decimal import Decimal
from DataModel.enums import PositionType, AssetModelType, DataFeedSource
from data_feeds.common.utils.generic import parse_currency_with_default, parse_country_with_default
from data_feeds.pictet_it.utils.generic import parse_pictet_date_with_default
from utils.generic import to_decimal_with_default

position_type_mapping = {
    'CASH': PositionType.CASH.value,
    'LOAN': PositionType.CASH.value,
    'FX': PositionType.FORWARD.value,
    'HOLDING': PositionType.HOLDING.value,
}


def position_from_p3det_file_row(file_path: str, line_number: int, row: pd.Series, context):
    position = Position(file_path, line_number)

    # TODO Handling type of holding
    type_of_holding = row['Type Of Holding'] # Cash or security
    if type_of_holding == 'FX':
        # TODO implement FX Forward positions
        return None

    instrument_type_code = row['FINANCIAL INSTR. CODE']
    if instrument_type_code == '717':  # skip pledges
        return None

    position.received_date = context.files_date
    position.owner_uid = row['CONTAINER NO']
    position.direct_owner_uid = position.owner_uid
    position.type = position_type_mapping.get(type_of_holding, PositionType.HOLDING.value)
    position.asset_name = row['SECURITY DESCRIPTION']
    position.quantity = to_decimal_with_default(row['Quantity'], '')
    position.position_date = parse_pictet_date_with_default(row['MARKET PRICE DATE'], '')
    position.fx_rate_date = parse_pictet_date_with_default(row['VALUATION DATE'], position.position_date)

    position.total_value_reference = to_decimal_with_default(row['VALUATION SECURITY CCY'], '')
    if position.total_value_reference != '':
        position.total_value_reference += to_decimal_with_default(row['ACCRUED INT. SECURITY CCY'])
    position.reference_currency = parse_currency_with_default(row['NOMINAL CURRENCY'], '')

    position.total_value_asset = to_decimal_with_default(row['MARKET VAL WITH ACC. INT.'], '')
    position.accrued_income = to_decimal_with_default(row['ACCRUED INT.'], '')
    position.asset_currency = parse_currency_with_default(row['VALUATION CURRENCY'], '')

    # TODO From here
    if type_of_holding == 'CONTRACT':
        asset_type = AssetModelType.LOAN
    else:
        asset_type = asset_model_type_mapping.get((instrument_type_code,))

    if asset_type is None:
        raise DataFeedError(DataFeedSource.JPMLUX, file_path, line_number, row,
                            JPMErrorType.TYPE_NOT_HANDLED.value,
                            f'Error parsing holdings positions row. Asset type {row["Instrument Type Code"]} has not been mapped, yet.')

    account_number = row['Account Number']
    valor = row['Valoren']
    if asset_type in {AssetModelType.CASH_ACCOUNT, AssetModelType.TIME_DEPOSIT}:
        position.asset_uid = f'{account_number}.{position.asset_currency}'
        if asset_type == AssetModelType.TIME_DEPOSIT:
            position.asset_uid += '.TD'
    elif asset_type == AssetModelType.LOAN:
        position.asset_uid = f'{account_number}.{position.asset_currency}.{parse_jpm_date_with_default(row["Start Date"])}.{position.quantity.normalize()}'
    else:
        position.asset_uid = f'{valor}_{position.asset_currency}'

    position.value_per_share = to_decimal_with_default(row['Price'], Decimal(1))
    if asset_type == AssetModelType.BOND:
        position.value_per_share /= 100
    position.price_date = parse_jpm_date_with_default(row['Date of Price'], position.position_date)

    position.fx_rate = to_decimal_with_default(row['Position To Base FX Rate'], Decimal(1))
    if position.fx_rate.is_zero():
        position.fx_rate = Decimal(1)

    position.cost_per_share = to_decimal_with_default(row['Average Cost Price'], '')

    # skip excluded vendor ids
    if any([v_id in context.excluded_vendor_ids for v_id in [account_number, valor]]):
        return None

    # if row['Base Accrued Interest'] and row['Quantity']:
    #     position.accrued_income_per_share = Decimal(row['Accrued Interest'])/Decimal(row['Quantity'])  # I GUESS

    asset_value_checksum = abs(
        to_decimal_with_default(position.value_per_share) * to_decimal_with_default(position.quantity)
        + to_decimal_with_default(position.accrued_income) - to_decimal_with_default(position.total_value_asset))

    if asset_value_checksum >= 0.01:
        print("something wrong in our total asset value")
        print(position.asset_name, "error =", asset_value_checksum)

    reference_value_checksum = abs(
        to_decimal_with_default(position.total_value_asset) * to_decimal_with_default(position.fx_rate, Decimal(1))
        - to_decimal_with_default(position.total_value_reference))

    if reference_value_checksum >= 10:
        # try inverting the fx rate
        position.fx_rate = Decimal(1) / position.fx_rate

        old_checksum = reference_value_checksum
        reference_value_checksum = abs(
            to_decimal_with_default(position.total_value_asset) * to_decimal_with_default(position.fx_rate, Decimal(1))
            - to_decimal_with_default(position.total_value_reference))

        if reference_value_checksum > old_checksum:
            # inverting the fx rate didn't improve things, use the original fx rate
            position.fx_rate = to_decimal_with_default(row['Position To Base FX Rate'], Decimal(1))

            print("something wrong in our total reference value")
            print(position.asset_name, "error =", old_checksum)
        elif reference_value_checksum >= 10:
            # still not enough
            print("something wrong in our total reference value")
            print(position.asset_name, "error =", reference_value_checksum)

    return position