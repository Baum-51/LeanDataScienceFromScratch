import os, sys
sys.path.append(os.getcwd())
import random

random.seed(0)

data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]

from matplotlib import pyplot as plt

def text_size(total: int) -> float:
    """
    totalが0なら8、200なら28nisuru
    """
    return 8 + total / 200 * 20


def fix_unicode(text: str) -> str:
    return text.replace(u"\u2019", "'")


import re
from bs4 import BeautifulSoup
import requests
    
url = "https://www.oreilly.com/ideas/what-is-data-science"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

#content = soup.find('div', 'article-body') # article-body divセクションを探す
content = soup.find('div', 'main-post-radar-content') # main-post-radar-content divセクションを探す
regex = r"[\w']+|[\.]"                       # 単語か、ピリオドにマッチする正規表現

document = []

for paragraph in content('p'):
    words = re.findall(regex, fix_unicode(paragraph.text))
    document.extend(words)
        
from collections import defaultdict

transitions = defaultdict(list)
for prev, current in zip(document, document[1:]):
    transitions[prev].append(current)

def generate_using_bigrams() -> str:
    current = "." # 次の単語が文の開始を示す
    result = []
    while True:
        next_word_candidates = transitions[current]   # bi-gramsでcurrentの次にくる単語
        
        current = random.choice(next_word_candidates) # そこから無作為に1つ選ぶ
        result.append(current)                        # 結果のリストに追加
        if current == ".": return " ".join(result)    # "."だったら終了


trigram_transitions = defaultdict(list)
starts = []
    
for prev, current, next in zip(document, document[1:], document[2:]):
    if prev == ".":            # 直前の単語がピリオドなら
        starts.append(current) # ここが文の開始
        
    trigram_transitions[(prev, current)].append(next)

def generate_using_trigrams() -> str:
    current = random.choice(starts) # 開始単語を無作為に選択する
    prev = "."
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next_word = random.choice(next_word_candidates)
        
        prev, current = current, next_word
        result.append(current)
        
        if current == ".":
            return " ".join(result)

# あとで参照するGrammerの型クラス
Grammar = dict[str, list[str]]

grammar = {
    "_S"  : ["_NP _VP"],
    "_NP" : ["_N",
             "_A _NP _P _A _N"],
    "_VP" : ["_V",
             "_V _NP"],
    "_N"  : ["data science", "Python", "regression"],
    "_A"  : ["big", "linear", "logistic"],
    "_P"  : ["about", "near"],
    "_V"  : ["learns", "trains", "tests", "is"]
}

def is_terminal(token: str) -> bool:
    return token[0] != "_"

def expand(grammar: Grammar, tokens: list[str]) -> list[str]:
    for i, token in enumerate(tokens):
        # 終端記号であれば、スキップする
        if is_terminal(token): continue
        
        # そうでなければ、非終端記号のためランダムに生成ルールのいずれかと置き換える
        replacement = random.choice(grammar[token])
    
        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            # 置き換えれば、例えば"_NP _VP"になる可能性があるため、
            # スペースで分割する必要がある
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
        
        # 新しいトークンのリストにexpandを適用する
        return expand(grammar, tokens)
    # ここに到達した際には、すべて終端記号に置き換え完了している
    return tokens

def generate_sentence(grammar: Grammar) -> list[str]:
    return expand(grammar, ["_S"])

def roll_a_die() -> int:
    return random.choice([1, 2, 3, 4, 5, 6])

def direct_sample() -> tuple[int, int]:
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

def random_y_given_x(x: int) -> int:
    """x + 1, x + 2, ... x + 6と同じ"""
    return x + roll_a_die()

def random_x_given_y(y: int) -> int:
    if y <= 7:
        # 合計が7以下であれば、最初の目は
        # 1, 2, ..., (合計 - 1)のいずれかである
        return random.randrange(1, y)
    else:
        # 合計がそれ以上なら、最初の目は
        # (合計 - 6), (合計 - 5), ..., 6のいずれかである
        return random.randrange(y - 6, 7)

def gibbs_sample(num_iters: int=100) -> tuple[int, int]:
    x, y = 1, 2 # 初期値は重要ではない
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

def compare_distributions(num_samples: int=1000) -> dict[int, list[int]]:
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts

def sample_from(weights: list[float]) -> int:
    """weights[i] / sum(weights)となるiを返す"""
    total = sum(weights)
    rnd = total * random.random()   # 0から重みの合計の間の一様乱数
    for i, w in enumerate(weights):
        rnd -= w
        if rnd <= 0: return i

from collections import Counter

# 1,000回生成して割合を調べる
draws = Counter(sample_from([0.1, 0.1, 0.8]) for _ in range(1000))
assert  10 < draws[0] < 190  # ~10%でなければならない。非常にざっくりしたテスト。
assert  10 < draws[1] < 190  # ~10%でなければならない。非常にざっくりしたテスト。
assert 650 < draws[2] < 950  # ~10%でなければならない。非常にざっくりしたテスト。
assert draws[0] + draws[1] + draws[2] == 1000


documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

K = 4
# 文章ごとのカウンターをリスト
document_topic_counts = [Counter() for _ in documents]
# トピックごとのカウンターをリスト
topic_word_counts = [Counter() for _ in range(K)]
# トピックごとの総数のリスト
topic_counts = [0 for _ in range(K)]
# 文章ごとの総数のリスト
document_lengths = [len(document) for document in documents]

distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

D = len(documents)


def p_topic_given_document(topic: int, d: int, alpha: float=0.1) -> float:
    """
    文章'd'の中で、トピック'topic'に割り当てられた単語の割合
    (スムージング項が含まれる)
    """
    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))

def p_word_given_topic(word: str, topic: int, beta: float=0.1) -> float:
    """
    トピック'topic'に割り当てられた単語'word'の割合
    (スムージング項が含まれる)
    """
    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))
    
def topic_weight(d: int, word: str, k: int) -> float:
    """
    与えられた文書とその中の単語に対する、k番目トピックの重みを返す
    """
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

def choose_new_topic(d: int, word: str) -> int:
    return sample_from([topic_weight(d, word, k) for k in range(K)])

random.seed(0)
document_topics = [[random.randrange(K) for word in document]
                   for document in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1
        
import tqdm

for iter in tqdm.trange(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):
            # 重みに影響を与えないように、まずこの単語 / トピックを除外する
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1
            
            # 重みに従い、新しいトピックを割り当てる
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic
            
            # 新しいトピックでカウンターを更新する
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1

for k, word_counts in enumerate(topic_word_counts):
    for word, count in word_counts.most_common():
        if count > 0:
            print(k, word, count)
            
topic_names = ["Big Data and programming languages",
               "Python and statistics",
               "databases",
               "machine learning"]

for document, topic_counts in zip(documents, document_topic_counts):
    print(document)
    for topic, count in topic_counts.most_common():
        if count > 0:
            print(topic_names[topic], count)
    print()

from scratch.linear_algebra import dot, Vector
import math

def cosine_similarity(v1: Vector, v2: Vector) -> float:
    return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))

assert cosine_similarity([1., 1, 1], [2., 2, 2]) == 1 # 同じ方向
assert cosine_similarity([-1., -1], [2., 2]) == -1    # 逆の方向
assert cosine_similarity([1., 0], [0, 1.]) == 0       # 直行

colors = ["red", "green", "blue", "yellow", "black", ""]
nouns = ["bed", "car", "boat", "cat"]
verbs = ["is", "was", "seems"]
adverbs = ["very", "quite", "extremely", ""]
adjectives = ["slow", "fast", "soft", "hard"]

def make_sentence() -> str:
    return " ".join([
        "The",
        random.choice(colors),
        random.choice(nouns),
        random.choice(verbs),
        random.choice(adverbs),
        random.choice(adjectives),
        "."
    ])

NUM_SENTENCES = 50
random.seed(0)
sentences = [make_sentence() for _ in range(NUM_SENTENCES)]



class Vocabulary:
    def __init__(self, words: list[str]=None) -> None:
        self.w2i: dict[str, int] = {} # word -> word_idのマッピング
        self.i2w: dict[int, str] = {} # word_id -> wordのマッピング
        
        for word in (words or []):
            self.add(word)
    
    @property
    def size(self) -> int:
        """Vocabularyクラスに属する単語の数"""
        return len(self.w2i)
    
    def add(self, word: str) -> None:
        if word not in self.w2i:     # 単語が新しければ
            word_id = len(self.w2i)  # idを割り当て
            self.w2i[word] = word_id # word -> word_idマッピングに追加
            self.i2w[word_id] = word
            
    def get_id(self, word: str) -> int:
        """与えられた単語のid(またはNone)を返す"""
        return self.w2i.get(word)
    
    def get_word(self, word_id: int) -> str:
        """与えられたidの単語(またはNone)を返す"""
        return self.i2w.get(word_id)
    
    def one_hot_encode(self, word: str):
        word_id = self.get_id(word)
        assert word_id is not None, f"unknown word {word}"
        
        return [1.0 if i == word_id else 0.0 for i in range(self.size)]
    
import json
def save_vocab(vocab: Vocabulary, filename: str) -> None:
    with open(filename, mode='w') as f:
        json.dump(vocab.w2i, f) # 保存の必要があるのはw2iのみ
    
def load_vocab(filename: str) -> Vocabulary:
    vocab = Vocabulary()
    with open(filename, mode='r') as f:
        # w2iを読み込んで、i2wを生成する
        vocab.w2i = json.load(f)
        vocab.i2w = {id: word for word, id in vocab.w2i.items()}
    return vocab

from typing import Iterable
from chapter19.chapter19 import Layer, Tensor, random_tensor, zeros_like

class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 使用する埋め込みごとに、サイズembedding_dimのベクトル
        self.embeddings = random_tensor(num_embeddings, embedding_dim)
        self.grad = zeros_like(self.embeddings)
        
        # 最後のidを保存する
        self.last_input_id = None
        
    def forward(self, input_id: int) -> Tensor:
        """入力のIDに対応する埋め込みベクトルを選択する"""
        self.input_id = input_id # 逆伝播のために覚えとく
        
        return self.embeddings[input_id]
    
    def backward(self, gradient: Tensor) -> None:
        # 最後の入力に対応する勾配をゼロにする
        # 毎回すべてゼロのテンソルを新しく作成するよりもはるかに簡単
        if self.last_input_id is not None:
            zero_row = [0 for _ in range(self.embedding_dim)]
            self.grad[self.last_input_id] = zero_row
            
        self.last_input_id = self.input_id
        self.grad[self.input_id] = gradient
        
    def params(self) -> Iterable[Tensor]:
        return [self.embeddings]
    
    def grads(self) -> Iterable[Tensor]:
        return [self.grad]

class TextEmbedding(Embedding):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        super().__init__(vocab.size, embedding_dim)
        
        # vocabularyを覚えておく
        self.vocab = vocab
    
    def __getitem__(self, word: str) -> Tensor:
        word_id = self.vocab.get_id(word)
        if word_id is not None:
            return self.embeddings[word_id]
        else:
            return None
        

    def closest(self, word: str, n: int=5) -> list[tuple[float, str]]:
        """コサイン類似度に基づいて周辺のn個の単語を返す"""
        vector = self[word]
        
        # ペア(similarity, other_word)を作成し、類似度の高い順にソートする
        scores = [(cosine_similarity(vector, self.embeddings[i]), other_word)
                   for other_word, i in self.vocab.w2i.items()]
        scores.sort(reverse=True)

from scratch.deep_learning import tensor_apply, tanh

class SimpleRnn(Layer):
    """可能な限り最もシンプルな再帰層"""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.w = random_tensor(hidden_dim, input_dim, init='xavier')
        self.u = random_tensor(hidden_dim, hidden_dim, init='xavier')
        self.b = random_tensor(hidden_dim)
        
        self.reset_hidden_state()
    
    def reset_hidden_state(self) -> None:
        self.hidden = [0 for _ in range(self.hidden_dim)]
        
    def forward(self, input: Tensor) -> Tensor:
        self.input = input             # inputと１つ前の隠れ層の状態を
        self.prev_hidden = self.hidden # 逆伝播のために保存する
        
        a = [(dot(self.w[h], input) +       # 重みとinputのドット積
              dot(self.u[h], self.hidden) + # 重みと隠れ層のドット積
              self.b[h])                    # バイアス
             for h in range(self.hidden_dim)]
        
        self.hidden = tensor_apply(tanh, a) # tanh活性化関数を適用して
        return self.hidden                  # 結果を返す
    
    def backward(self, gradient: Tensor):
        # tanhを逆伝播させる
        a_grad = [gradient[h] * (1 - self.hidden[h] ** 2)
                  for h in range(self.hidden_dim)]
        
        # bはaと同じ勾配を持つ
        self.b_grad = a_grad
        
        # w[h][i]はinput[i]で乗算し、a[h]を加えるため
        # w_grad[h][i] = a_grad[h] * input[i]
        self.w_grad = [[a_grad[h] * input[i]
                        for i in range(self.input_dim)]
                       for h in range(self.hidden_dim)]
        
        # 各u[h][h2]はhidden[h2]で乗算し、a[h]を加えるため
        # u_grad[h][h2] = a_grad[h] * prev_hidden[h2]
        self.u_grad = [[a_grad[h] * self.prev_hidden[h2]
                        for h2 in range(self.hidden_dim)]
                       for h in range(self.hidden_dim)]
        
        # input[i]は、各w[h][i]で乗算され、a[h]を加えるため
        # input_grad[i] = sum(a_grad[h] * w[h][i] for h in ...)
        return [sum(a_grad[h] * self.w[h][i] for h in range(self.hidden_dim))
                for i in range(self.input_dim)]
    
    def params(self) -> Iterable[Tensor]:
        return [self.w, self.u, self.b]
    
    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.u_grad, self.b_grad]
    
from bs4 import BeautifulSoup
import requests

# 内容が動的のためrequestでは取れない
url = "https://www.ycombinator.com/topcompanies/"
soup = BeautifulSoup(requests.get(url).text, 'html5lib')

# 社名を2回取得してしまうので、集合内包表記を使って重複排除する
"""
companies = list({b.text
                  for b in soup('b')
                  if "h4" in b.get("class", ('_coName_i9oky_470'))})
"""
# class="_coName_i9oky_470"
companies =  [b.text for b in soup.find_all('span', '_coName_i9oky_470')]
print(len(companies))
assert len(companies) == 101





if __name__ == '__main__':
    for word, job_popularity, resume_popularity in data:
        plt.text(job_popularity, resume_popularity, word,
                 ha='center', va='center',
                 size=text_size(job_popularity + resume_popularity))
        plt.xlabel("Popularity on Job Postings")
        plt.ylabel("Popularity on Resumes")
        plt.axis([0, 100, 0, 100])
        plt.xticks([])
        plt.yticks([])
        plt.savefig('../picture/chap21-1.png')
        
    
    
    print(generate_using_bigrams())
    print('='*100)
    print(generate_using_trigrams())
    print('='*100)
    print(generate_sentence(grammar))
    
    
    import re

    # 簡単な正規表現だが、少なくともここで使用するデータに対しては機能する
    tokenized_sentences = [re.findall("[a-z]+|[.]", sentence.lower())
                        for sentence in sentences]
    

    # テキストに基づいてVocabulary(word -> word_id)マッピングを作成する
    vocab = Vocabulary(word
                    for sentence_words in tokenized_sentences
                    for word in sentence_words)

    from chapter19.chapter19 import Tensor, one_hot_encode

    inputs: list[int] = []
    targets: list[Tensor] = []

    for sentence in tokenized_sentences:
        for i, word in enumerate(sentence):
            for j in [i - 2, i -1, i + 1, i + 2]:
                if 0 <= j < len(sentence):
                    nearby_word = sentence[j]
                    
                    # inputにword_idを追加する
                    inputs.append(vocab.get_id(word))
                    
                    # targetに、周囲の単語のOne-Hotエンコーディングを追加する
                    targets.append(vocab.one_hot_encode(nearby_word))
                    
    from chapter19.chapter19 import Sequential, Linear

    random.seed(0)
    EMBEDDING_DIM = 5 # 問題ないと思われる大きさ

    # 埋め込み層を参照できるように別に定義する
    embedding = TextEmbedding(vocab=vocab, embedding_dim=EMBEDDING_DIM)

    model = Sequential([
        # (word_idのベクトルとして)単語が与えられたら、その埋め込み層を調べる
        # 周囲の単語のスコアを計算するために、線形層を使う
        embedding,
        Linear(input_dim=EMBEDDING_DIM, output_dim=vocab.size)
    ])
    
    from chapter19.chapter19 import SoftmaxCrossEntropy, Momentum, GradientDescent

    loss = SoftmaxCrossEntropy()
    optimizer = GradientDescent(learning_rate=0.01)

    for epoch in range(100):
        epoch_loss = 0.0
        for input, target in zip(inputs, targets):
            predicted = model.forward(input)
            epoch_loss = loss.loss(predicted, target)
            gradient = loss.gradient(predicted, target)
            model.backward(gradient)
            optimizer.step(model)
        print(epoch, epoch_loss)          # 損失を表示する
        print(embedding.closest("black")) # 加えていくつか周辺の単語を表示して
        print(embedding.closest("slow"))  # 何が学習されているかを
        print(embedding.closest("car"))   # 確認する
    
    pairs = [(cosine_similarity(embedding[w1], embedding[w2]), w1, w2)
             for w1 in vocab.w2i
             for w2 in vocab.w2i
             if w1 < w2]
    pairs.sort(reverse=True)
    print(pairs[:5])
    
    from scratch.working_with_data import pca, transform
    import matplotlib.pyplot as plt

    components = pca(embedding.embeddings, 2)
    transformed = transform(embedding.embeddings, components)

    # 散布図をプロットする（点が見えないように白にする）
    fig, ax = plt.subplots()
    ax.scatter(*zip(*transformed), marker='.', color='w')

    # 各位置にアノテーションとして単語を配置する
    for word, idx in vocab.w2i.items():
        ax.annotate(word, transformed[idx])
        
    # 軸を非表示にする
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.savefig('../picture/chap21-2.png')