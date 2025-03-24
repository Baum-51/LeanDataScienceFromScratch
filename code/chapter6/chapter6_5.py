def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x < 1 else 0
def uniform_cdf(x: float) -> float:
    """一葉確率変数がx以下となる確率を返す"""
    if   x < 0: return 0 # 一様分布は0を下回らない
    elif x < 1: return x # 例えば、P(X <= 0.4) = 0.4となる
    else      : return 1 # 一様分布は最大で1
    