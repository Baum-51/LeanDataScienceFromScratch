import datetime

stock_price = {'closing_price': 102.06,
               'date': datetime.date(2014, 8, 29),
               'symbol': 'APPL'}

prices: dict['datetime.date', float] = {}

from collections import namedtuple

StockPrice = namedtuple('StockPrice',  ['symbol', 'date', 'closing_price'])
price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol =='MSFT'
assert price.closing_price == 106.03

from typing import NamedTuple

class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float
    
    def is_high_tech(self) -> bool:
        """これはクラスなので、メソッドの追加も可能"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN&', 'AAPL']
    
price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03
assert price.is_high_tech()


from dataclasses import dataclass

@dataclass
class StockPrice2:
    symbol: str
    date: str
    closing_price: float
    
    def is_high_tech(self) -> bool:
        """これはクラスなので、メソッドの追加も可能"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN&', 'AAPL']
    
price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price2.symbol == 'MSFT'
assert price2.closing_price == 106.03
assert price2.is_high_tech()

# stockデータを半分にする
price2.closing_price /= 2
assert price2.closing_price == 53.015

from dateutil.parser import parse

def parse_row(row: list[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol=symbol,
                      date=parse(date).date(),
                      closing_price=float(closing_price))
# parse_rowのテスト
stock = parse_row(['MSFT', "2018-12-14", "106.03"])

assert stock.symbol == 'MSFT'
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03

from typing import Optional
import re

def try_parse_row(row: list[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price = row
    
    # 銘柄シンボルは、すべて大文字
    if not re.match(r"^[A-Z]+$", symbol):
        return None
    
    try:
        date = parse(date_).date()
    except ValueError:
        return None
    
    try:
        closing_price = float(closing_price)
    except ValueError:
        return None
    
    return StockPrice(symbol, date, closing_price)

# エラーの場合はNoneが返るべき
assert try_parse_row(['MSFT0', "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None
# データが良好である場合は、以前と同じデータを返すべき
assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock

import csv

data: list[StockPrice] = []

with open('other_text/comma_delimited_stock_prices.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        maybe_stock = try_parse_row(row)
        if maybe_stock is None:
            print(f"skipping invalid row: {row}")
        else:
            data.append(maybe_stock)