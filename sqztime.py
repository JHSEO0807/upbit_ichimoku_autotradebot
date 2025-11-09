import pandas as pd
import numpy as np
import pyupbit

# ================= 설정 =================
TICKER       = "KRW-SOL"
INTERVAL     = "minute5"       # minute1, minute3, minute5, minute10, minute15, minute30, minute60, minute240
MAX_BARS     = 1000            # 충분한 히스토리(약 2주치 5분봉) 확보

# Pine 입력값(너가 올린 스크립트와 동일)
BB_LENGTH    = 20              # length
BB_MULT      = 2.0             # mult (주의: 아래에서 dev는 multKC를 사용함 — Pine 코드 그대로)
KC_LENGTH    = 20              # lengthKC
KC_MULT      = 1.5             # multKC
USE_TR       = True            # useTrueRange

PRINT_LAST   = 30              # 콘솔에 최근 신호 몇 건까지 표시할지
# =======================================


def fetch_ohlcv_paginated(ticker: str, interval: str, max_bars: int) -> pd.DataFrame:
    """
    Upbit OHLCV 페이지네이션 수집.
    - pyupbit.get_ohlcv는 호출 시점부터 과거로 200개씩 끊어 가져옴
    - 타임존을 Asia/Seoul로 통일하고, 최종 인덱스는 naive(KST)로 정규화
    """
    dfs, to = [], None
    remaining = max_bars
    while remaining > 0:
        cnt = min(200, remaining)
        df = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=cnt, to=to)
        if df is None or df.empty:
            break

        # 인덱스 KST로 통일
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Seoul")
        else:
            df.index = df.index.tz_convert("Asia/Seoul")

        dfs.append(df)
        remaining -= len(df)

        # 다음 페이지 기준(가장 오래된 봉 직전 1초, UTC)
        oldest_kst = df.index.min()
        to_utc = oldest_kst.tz_convert("UTC") - pd.Timedelta(seconds=1)
        to = to_utc.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")

        if len(df) < cnt:
            break

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    # 최종 인덱스를 KST naive로
    out.index = out.index.tz_convert("Asia/Seoul").tz_localize(None)
    return out.tail(max_bars)


# --- Pine helper equivalents ---
def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length, min_periods=length).mean()

def stdev_pine(s: pd.Series, length: int) -> pd.Series:
    # ta.stdev: sqrt( SMA(x^2, n) - SMA(x, n)^2 ), 음수 오차 0 클립
    m1 = sma(s, length)
    m2 = sma(s * s, length)
    var = (m2 - m1 * m1).clip(lower=0.0)
    return np.sqrt(var)

def true_range_pine(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    tr1 = h - l
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def linreg_last_pine(y: pd.Series, length: int) -> pd.Series:
    """
    ta.linreg(y, length, 0): 윈도우 x=0..length-1 에서 회귀직선의 마지막 포인트 예측값.
    """
    def _apply(arr: np.ndarray) -> float:
        n = len(arr)
        x = np.arange(n, dtype=float)
        yv = arr.astype(float)
        xm = x.mean(); ym = yv.mean()
        vx = ((x - xm) ** 2).sum()
        if vx == 0:
            return yv[-1]
        cov = ((x - xm) * (yv - ym)).sum()
        slope = cov / vx
        inter = ym - slope * xm
        return inter + slope * (n - 1)
    return y.rolling(length, min_periods=length).apply(_apply, raw=True)


def compute_sqzmom_buy(df: pd.DataFrame) -> pd.DataFrame:
    """
    질문에 올린 LazyBear 변형 스크립트와 완전 동일한 수식으로 계산.
    - dev = multKC * stdev(source, length)  ← (의도적으로 동일하게 구현)
    - sqzOn/sqzOff 원본 부등호
    - buySignal = (sqzOn.shift(1) & sqzOff) & (val>0 & val>val.shift(1, nz=0))
    """
    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]

    # ===== Calculate BB =====
    source = c
    basis  = sma(source, BB_LENGTH)
    dev    = KC_MULT * stdev_pine(source, BB_LENGTH)   # ← Pine 코드 그대로(multKC 사용)
    upperBB = basis + dev
    lowerBB = basis - dev

    # ===== Calculate KC =====
    ma      = sma(source, KC_LENGTH)
    rng     = true_range_pine(h, l, c) if USE_TR else (h - l)
    rangema = sma(rng, KC_LENGTH)
    upperKC = ma + rangema * KC_MULT
    lowerKC = ma - rangema * KC_MULT

    # ===== Squeeze States =====
    sqzOn  = (lowerBB > lowerKC) & (upperBB <  upperKC)   # black
    sqzOff = (lowerBB < lowerKC) & (upperBB >  upperKC)   # gray
    noSqz  = ~(sqzOn | sqzOff)                            # blue

    # ===== Momentum Value =====
    hh = h.rolling(KC_LENGTH, min_periods=KC_LENGTH).max()
    ll = l.rolling(KC_LENGTH, min_periods=KC_LENGTH).min()
    midHL = (hh + ll) / 2.0
    center = (midHL + sma(c, KC_LENGTH)) / 2.0
    val = linreg_last_pine(source - center, KC_LENGTH)
    val_prev = val.shift(1)
    nz_val_prev = val_prev.fillna(0.0)

    # ===== BUY SIGNAL =====
    isLime = (val > 0) & (val > nz_val_prev)
    squeezeReleaseUp = sqzOn.shift(1, fill_value=False) & sqzOff
    buySignal = squeezeReleaseUp & isLime

    out = pd.DataFrame(index=df.index)
    out["open"] = o; out["high"] = h; out["low"] = l; out["close"] = c
    out["upperBB"] = upperBB; out["lowerBB"] = lowerBB
    out["upperKC"] = upperKC; out["lowerKC"] = lowerKC
    out["sqzOn"] = sqzOn; out["sqzOff"] = sqzOff; out["noSqz"] = noSqz
    out["val"] = val
    out["isLime"] = isLime
    out["squeezeReleaseUp"] = squeezeReleaseUp
    out["buySignal"] = buySignal

    # 보기 좋은 출력용 시간 컬럼(KST)
    out["time"] = out.index.strftime("%Y-%m-%d %H:%M:%S")
    return out


def main():
    df = fetch_ohlcv_paginated(TICKER, INTERVAL, MAX_BARS)
    if df.empty:
        raise RuntimeError("OHLCV 데이터를 가져오지 못했습니다.")

    sig = compute_sqzmom_buy(df)

    buys = sig[sig["buySignal"]].copy()
    cols_show = ["time","open","high","low","close","val","squeezeReleaseUp","isLime","sqzOn","sqzOff"]
    print(f"\n[요약] 히스토리 봉 수={len(sig)}, 매수 신호 수={int(buys.shape[0])}")
    if buys.empty:
        print("최근 구간에 매수 신호가 없습니다. (타임프레임/기간을 늘리거나 파라미터를 조정해보세요.)")
        return

    print(f"\n=== 최근 매수 신호 {min(PRINT_LAST, len(buys))}건 (KST) ===")
    print(buys[cols_show].tail(PRINT_LAST).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    main()
