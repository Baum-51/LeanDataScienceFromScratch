import math

def entropy(class_probabilities: list[float]) -> float:
    """各分類の確率リストから、エントロピーを計算する"""
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p > 0)

assert entropy([1.0]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.25])

from typing import Any
from collections import Counter

def class_probabilities(labels: list[Any]) -> list[float]:
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labels: list[Any]) -> float:
    return entropy(class_probabilities(labels))

assert data_entropy(['a']) == 0
assert data_entropy([True, False]) == 1
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])

def partition_entropy(subsets: list[list[Any]]) -> float:
    """データを部分集合に分割した場合のエントロピーを返す"""
    total_count = sum(len(subset) for subset in subsets)
    
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

from typing import NamedTuple, Optional

class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None # 値なしを許容する
    
                  #  level     lang     tweets  phd  did_well
inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)
         ]

from typing import TypeVar
from collections import defaultdict

T = TypeVar('T') # 入力を表す汎用型

def partition_by(inputs: list[T], attribute: str) -> dict[Any, list[T]]:
    """指定された属性に基づいて、入力をリストに分割する"""
    partitions: dict[Any, list[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute) # 指定した属性の値
        partitions[key].append(input) # inputを正しい分類に追加する
    return partitions

def partition_entropy_by(inputs: list[Any],
                         attribute: str,
                         label_attribute: str) -> float:
    """与えられた分割におけるエントロピーを計算する"""
    # partitionsはinputで構成される
    partitions = partition_by(inputs, attribute)
    
    # ただし、partition_entropyにはクラスラベルのみが必要
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]
    
    return partition_entropy(labels)

for key in ['level', 'lang', 'tweets', 'phd']:
    print(key, partition_entropy_by(inputs, key, 'did_well'))
    
assert 0.69 < partition_entropy_by(inputs, 'level', 'did_well') < 0.70
assert 0.86 < partition_entropy_by(inputs, 'lang', 'did_well') < 0.87
assert 0.78 < partition_entropy_by(inputs, 'tweets', 'did_well') < 0.79
assert 0.89 < partition_entropy_by(inputs, 'phd', 'did_well') < 0.90

# 最もエントロピーの低かったlevel(経験)での分割を行う。
# levelがmidのものはdid_wellがすべてTrueになるため次の分割はいらない。
# 他２つは分割が必要となる。ここではseniorを例次の分割をする特徴量を選ぶ(エントロピーが最も低いもの)
senior_inputs = [input for input in inputs if input.level == 'Senior']

assert 0.4 == partition_entropy_by(senior_inputs, 'lang', 'did_well')
assert 0.0 == partition_entropy_by(senior_inputs, 'tweets', 'did_well')
assert 0.95 < partition_entropy_by(senior_inputs, 'phd', 'did_well') < 0.96

from typing import NamedTuple, Union, Any

class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None
    
DecisionTree = Union[Leaf, Split]

# LeafとSplitを使うとlevelの決定木を以下のように表せる
hiring_tree = Split('level', {  # 最初に経験値（level）を考慮する
    'Junior': Split('phd', {    # 経験値が低であれば、次にphdを考慮する
        False: Leaf(True),      # phdがFalseであれば、採用（True）と予測する
        True: Leaf(False)       # phdがTrueであれば、不採用（False）と予測する
    }),
    'Mid': Leaf(True),          # 経験値が中の場合、単に採用（True）と予測する
    'Senior': Split('tweets', { # 経験値が高の場合、次にツイートを考慮する
        False: Leaf(False),     # ツイートがFalseであれば不採用（False）と予測する
        True: Leaf(True)        # ツイートがTrueであれば採用（True）と予測する
    })
})

def classify(tree: DecisionTree, input: Any) -> Any:
    """入力を与えられた決定木に従い分類する"""
    
    # 末端ノードであれば、その値を返す
    if isinstance(tree, Leaf):
        return tree.value
    
    # そうでなければ、決定木は分類を行う属性と、
    # その属性の値と次に適用する決定木の辞書の
    # タプルである
    subtree_key = getattr(input, tree.attribute)
    if subtree_key not in tree.subtrees:
        return tree.default_value
    
    subtree = tree.subtrees[subtree_key]
    return classify(subtree, input)

def build_tree_id3(inputs: list[Any],
                   split_attributes: list[str],
                   target_attribute: str) -> DecisionTree:
    # ターゲットラベルをカウント
    label_counts = Counter(getattr(input, target_attribute) for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]
    
    # ユニークなラベルであれば、予測を行う
    if len(label_counts) == 1:
        return Leaf(most_common_label)
    # 分岐が残ってなければ、最も多いラベルを結果として返す
    if not split_attributes:
        return Leaf(most_common_label)
    
    # そうでなければ、最良の属性を使って分割する
    def split_entropy(attribute: str) -> float:
        """最良の属性を見つけるヘルパー関数"""
        return partition_entropy_by(inputs, attribute, target_attribute)
    best_attribute = min(split_attributes, key=split_entropy)
    
    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]
    
    # 再帰的に部分木を生成する
    subtrees = {attribute_value : build_tree_id3(subset,
                                                 new_attributes,
                                                 target_attribute)
                for attribute_value, subset in partitions.items()}
    return Split(best_attribute, subtrees, default_value=most_common_label)

tree = build_tree_id3(inputs, ['level', 'lang', 'tweets', 'phd'], 'did_well')

# Trueと予測する
assert classify(tree, Candidate('Junior', 'Java', True, False))
# Falseと予測する
assert not classify(tree, Candidate('Junior', 'Java', True, True))

# Trueと予測する
assert classify(tree, Candidate('Intern', 'Java', 'True', 'True'))