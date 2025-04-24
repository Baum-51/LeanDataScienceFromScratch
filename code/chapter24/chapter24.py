import os, sys
sys.path.append(os.getcwd())

users = [[0, "Hero", 0],
         [1, "Dunn", 2],
         [2, "Sue", 3],
         [3, "Chi", 3]]

from typing import Sequence, Any, Callable, Iterator
from collections import defaultdict

# 後で使用する型エイリアス
Row = dict[str, Any]                       # データベースの行(row)
WhereClause = Callable[[Row], bool]        # 単一行に対する述語
HavingClause = Callable[[list[Row]], bool] # 複数行に対する述語

class Table:
    def __init__(self, columns: list[str], types: list[type]) -> None:
        assert len(columns) == len(types), "# 列の数と型の数は等しく無ければならない"
        
        self.columns = columns    # 列名
        self.types = types        # 列の型
        self.rows: list[Row] = [] # (まだデータはない)
    
    def col2type(self, col: str) -> type:
        idx = self.columns.index(col)
        return self.types[idx]
    
    def insert(self, values: list) -> None:
        # 値の数が正しいことを確認
        if len(values) != len(self.types):
            raise ValueError(f"You need to provide {len(self.types)} values")
        
        # 値の型が正しいことを確認
        for value, typ3 in zip(values, self.types):
            if not isinstance(value, typ3) and value is not None:
                raise TypeError(f"Expected type {typ3} but got {value}")
        
        # 行として辞書を追加する
        self.rows.append(dict(zip(self.columns, values)))
    
    def __getitem__(self, idx: int) -> Row:
        return self.rows[idx]
    
    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def __repr__(self):
        """表用のプリティープリント：列に続いて行"""
        rows = "\n".join(str(row) for row in self.rows)
        
        return f"{self.columns}\n{rows}"
    
    def update(self,
               updates: dict[str, Any],
               predicate: WhereClause=lambda row: True):
        # 最初に、名前と型が有効であることを確認する
        for column, new_value in updates.items():
            if column not in self.columns:
                raise ValueError(f"invalid column: {column}")
            
            typ3 = self.col2type(column)
            if not isinstance(new_value, typ3) and new_value is not None:
                raise TypeError(f"expected type {typ3}, but got {new_value}")
        
        # 更新を行う
        for row in self.rows:
            if predicate(row):
                for column, new_value in updates.items():
                    row[column] = new_value


if __name__ == '__main__':
    # コンストラクタには列の名前と型が必要
    users = Table(['user_id', 'name', 'num_friends'], [int, str, int])
    users.insert([0, "Hero", 0])
    users.insert([1, "Dunn", 2])
    users.insert([2, "Sue", 3])
    users.insert([3, "Chi", 3])
    users.insert([4, "Thor", 3])
    users.insert([5, "Clive", 2])
    users.insert([6, "Hicks", 3])
    users.insert([7, "Devin", 2])
    users.insert([8, "Kate", 2])
    users.insert([9, "Klein", 3])
    users.insert([10, "Jen", 1])
    
    print(users)
    print('='*100)
    
    assert len(users) == 11
    assert users[1]['name'] == 'Dunn'
    
    assert users[1]['num_friends'] == 2 # 元の値
    users.update({'num_friends': 3},    # num_friends = 3にする
                 lambda row: row['user_id'] == 1)
    assert users[1]['num_friends'] == 3 # 更新後の値
    
    