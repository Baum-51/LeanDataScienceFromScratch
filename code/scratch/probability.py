import math

def normal_cdf(x: float, mu: float=0, sigma: float=1) -> float:
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p:         float,
                       mu:        float=0,
                       sigma:     float=1,
                       tolerance: float=0.00001) -> float:
    """二部探索を用いて、逆関数の近似値を計算する"""
    # 標準正規分布でない場合、標準正規分布からの差分を求める
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z = -10.0
    hi_z  =  10.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = (normal_cdf(mid_z))
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
    return mid_z