from DataModel.enums import FeeTaxExpenseType

jpm_fte_mapping = {
    ('BROKERAGE FEES',): FeeTaxExpenseType.BROKERAGE_FEE,
    ('CORRESPONDENT FEES',): FeeTaxExpenseType.MANAGEMENT_FEE,
    ('FOREIGN TAXS',): FeeTaxExpenseType.FOREIGN_TAX,
}