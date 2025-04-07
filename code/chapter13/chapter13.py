import re

def tokenize(text: str) -> set[str]:
    text = text.lower()                       # 小文字に変換する
    all_words = re.findall("[a-z0-9]+", text) # 単語を識別する（日本語ではできない）
    return set(all_words)                     # 重複を取り除く

assert tokenize("Data Science is science") == {"data", "science", "is"}

from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool
    
from typing import Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float=0.5):
        self.k = k # スムージング
        
        self.tokens: set[str] = set()
        self.token_spam_counts: dict[str, int] = defaultdict(int)
        self.token_ham_counts:  dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0
        
    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # メッセージの数をインクリメント
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
                
            # 単語の数をインクリメント
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1
    
    def _probabilities(self, token: str) -> tuple[float, float]:
        """tokenに対するP(token|spam)とP(toke|ham)を返す"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham  = (ham  + self.k) / (self.ham_messages  + 2 * self.k)
        
        return p_token_spam, p_token_ham
    
    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0
        
        # 語彙リストの単語を順に適用する
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            
            # その単語がメッセージ中に現れた場合、
            # その確率の対数を加算する
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham  += math.log(prob_if_ham)
            # メッセージに現れなかった場合には、単語にを含まない場合の確率の対数、
            # つまりlog(1-含む場合の確率を加算する)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham  += math.log(1.0 - prob_if_ham)
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham  = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)

messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages  == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts  == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"

probs_if_spam = [
        (1 + 0.5) / (1 + 2 * 0.5),     # "spam"(存在する)
    1 - (0 + 0.5) / (1 + 2 * 0.5), # "ham"(存在しない)
    1 - (1 + 0.5) / (1 + 2 * 0.5), # "rules"(存在しない)
        (0 + 0.5) / (1 + 2 * 0.5),     # "hello"(存在する)
]

probs_if_ham = [
        (0 + 0.5) / (2 + 2 * 0.5),     # "spam"(存在する)
    1 - (2 + 0.5) / (2 + 2 * 0.5), # "ham"(存在しない)
    1 - (1 + 0.5) / (2 + 2 * 0.5), # "rules"(存在しない)
        (1 + 0.5) / (2 + 2 * 0.5),     # "hello"(存在する)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)