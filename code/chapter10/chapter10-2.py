import os
import sys
sys.path.append(os.getcwd())

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

data: list[StockPrice] = [] 
with open("other_text/stocks.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = [[row['Symbol'], row['Date'], row['Close']]
            for row in reader]
    for row in rows:
        maybe_stock = try_parse_row(row)
        if maybe_stock is None:
            print(f"skipping invalid row: {row}")
        else:
            data.append(maybe_stock)


max_appl_price = max(stock_price.closing_price
                     for stock_price in data
                     if stock_price.symbol == "AAPL")
print(max_appl_price)

from collections import defaultdict

max_prices: dict[str, float] = defaultdict(lambda: float('-inf'))

for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price
print(max_prices)

# 銘柄ごとに値を集める
prices: dict[str, list[StockPrice]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)

# 日付の順に価格を並べる
prices = {symbol: sorted(symbol_prices)
          for symbol, symbol_prices in prices.items()}

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1

class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float

def day_over_day_changes(prices: list[StockPrice]) -> list[DailyChange]:
    """
    同じ銘柄の株価が日付順に並び変えると想定する
    """
    return [DailyChange(symbol=today.symbol,
                        date=today.date,
                        pct_change=pct_change(yesterday, today))
                    for yesterday, today in zip(prices, prices[1:])]
    
all_changes = [change
               for symbol_prices in prices.values()
               for change in day_over_day_changes(symbol_prices)]

max_change = max(all_changes, key=lambda change: change.pct_change)
# http://news.cnet.com/2100-1001-202143.htmlを参照（アクセスできない）
assert max_change.symbol == 'AAPL'
assert max_change.date   == datetime.date(1997, 8, 6)
assert 0.33 < max_change.pct_change < 0.34

min_change = min(all_changes, key=lambda change: change.pct_change)
# http://money.cnn.com/200/09/29/markets/techwrap/を参照
assert min_change.symbol == 'AAPL'
assert min_change.date   == datetime.date(2000, 9, 29)
assert -0.52 < min_change.pct_change < -0.51

changes_by_month: dict[DailyChange] = {month: [] for month in range(1, 13)}
print(changes_by_month)
for change in all_changes:
    changes_by_month[change.date.month].append(change)
    
avg_daily_change = {
    month: sum(change.pct_change for change in changes) / len(changes)
    for month, changes in changes_by_month.items()
}

assert avg_daily_change[10] == max(avg_daily_change.values())

from scratch.linear_algebra import distance

a_to_b = distance([63, 150], [67, 160])
a_to_c = distance([63, 150], [70, 171])
b_to_c = distance([67, 160], [70, 171])

print(f'a_to_b {a_to_b}')
print(f'a_to_c {a_to_c}')
print(f'b_to_c {b_to_c}')

a_to_b = distance([160, 150],   [170.2, 160])
a_to_c = distance([160, 150],   [177.8, 171])
b_to_c = distance([170.2, 160], [177.8, 171])

print(f'a_to_b {a_to_b}')
print(f'a_to_c {a_to_c}')
print(f'b_to_c {b_to_c}')

from scratch.linear_algebra import vector_mean
from scratch.linear_algebra import Matrix, Vector, make_matrix
from scratch.statistics import standard_deviation


def scale(data: list[Vector]) -> tuple([Vector, Vector]):
    """各次元の平均と標準偏差を返す"""
    dim = len(data[0])
    
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data]) for i in range(dim)]
    
    return means, stdevs

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)
assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]

def rescale(data: list[Vector]) -> list[Vector]:
    """
    各次元が平均0、標準偏差1となるように入力データのスケールを修正する
    (標準偏差が0の場合は変更しない)
    """
    dim = len(data[0])
    means, stdevs = scale(data)
    
    # 各ベクトルのコピーを作る
    rescaled = [v[:] for v in data]
    
    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
    return rescaled

means, stdevs = scale(rescale(vectors))
assert means == [0, 0, 1]
assert stdevs == [1, 1, 0]

import tqdm
import random

for i in tqdm.tqdm(range(100)):
    " 何か時間のかかる処理を行う"
    _=[random.random() for _ in range(1000000)]
    
def primes_up_to(n: int) -> list[int]:
    primes = [2]
    with tqdm.trange(3, n) as t:
        for i in t:
            # 自分より小さい素数で割り切らなければ、iは素数である
            i_is_prime = not any(i % p == 0 for p in primes)
            if i_is_prime:
                primes.append(i)
            t.set_description(f"{len(primes)} primes")
            
my_primes = primes_up_to(100_000)

from scratch.linear_algebra import subtract

def de_mean(data: list[Vector]) -> list[Vector]:
    """すべての次元で平均0となるように、データを再センタリングする"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]

from scratch.linear_algebra import magnitude

def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]

from scratch.linear_algebra import dot
def directional_variance(data: list[Vector], w: Vector) -> float:
    """
    wが示す方向に対する、xの分散を求める
    """
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)

def directional_variance_gradient(data: list[Vector], w: Vector):
    """
    w方向の分散に対する勾配
    """
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data) for i in range(len(w))]

from scratch.gradient_descent import gradient_step
def first_principal_component(data: list[Vector],
                              n: int=100,
                              step_size: float=0.1) -> Vector:
    # ランダムな推測地で開始する
    guess = [1.0 for _ in data[0]]
    
    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")
    return direction(guess)

from scratch.linear_algebra import scalar_multiply

def project(v: Vector, w: Vector) -> Vector:
    """vをw方向に射影したベクトルを返す"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """vをw方向へ射影した結果をvから取り除く"""
    return subtract(v, project(v, w))

def remove_projection(data: list[Vector], w: Vector) -> list[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]

def pca(data: list[Vector], num_components: int) -> list[Vector]:
    components: list[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)
    return components

def transform_vector(v: Vector, components: list[Vector]) -> list[Vector]:
    return [dot(v, w) for w in components]

def transform(data: list[Vector], components: list[Vector]) -> list[Vector]:
    return [transform_vector(v, components) for v in data]