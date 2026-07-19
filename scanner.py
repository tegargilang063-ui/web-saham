"""
Logika scanner SUPERKETAT — diadaptasi dari notebook Superketat.ipynb
(metodologi CIA - Chart Investor Academy @tradingdiary2)
"""
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf


def hitung_semua_indikator(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung semua indikator CIA style dari data OHLCV."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']
    open_ = df['Open']

    for p in [3, 5, 10, 20, 50, 100, 200]:
        df[f'ma{p}'] = close.rolling(p).mean()

    for p in [3, 5, 10, 20, 50, 150, 200]:
        df[f'ema{p}'] = close.ewm(span=p, adjust=False).mean()

    def wma(series, period):
        w = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

    df['wma20'] = wma(close, 20)
    df['wma50'] = wma(close, 50)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_mid'] = bb_mid
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid * 100
    df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

    k_period = 15
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    raw_k = (close - low_min) / (high_max - low_min) * 100
    df['stoch_k'] = raw_k.rolling(3).mean()
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    tp = (high + low + close) / 3
    rmf = tp * vol
    pos_flow = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_flow = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    df['mfi'] = 100 - (100 / (1 + mfr))

    guro_raw = (df['rsi'] + df['mfi'] + df['stoch_k']) / 3 / 10
    df['guro_ii'] = guro_raw.round(2)
    df['guro_ma5'] = guro_raw.rolling(5).mean().round(2)

    obv = [0]
    close_vals = close.values
    vol_vals = vol.values
    for i in range(1, len(close_vals)):
        if close_vals[i] > close_vals[i - 1]:
            obv.append(obv[-1] + vol_vals[i])
        elif close_vals[i] < close_vals[i - 1]:
            obv.append(obv[-1] - vol_vals[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_ma5'] = pd.Series(obv, index=df.index).rolling(5).mean()

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / close * 100

    df['vol_ma5'] = vol.rolling(5).mean()
    df['vol_ma20'] = vol.rolling(20).mean()
    df['vol_spike'] = vol / df['vol_ma5'].replace(0, np.nan)
    df['value'] = close * vol
    df['value_miliar'] = df['value'] / 1e9

    df['body'] = (close - open_).abs()
    df['candle_dir'] = np.where(close >= open_, 'HIJAU', 'MERAH')

    return df


def hitung_superketat(row, threshold=3.0):
    close = row.get('Close')
    ma3 = row.get('ma3')
    ma5 = row.get('ma5')
    ma10 = row.get('ma10')
    ma20 = row.get('ma20')

    if any(pd.isna([close, ma3, ma5, ma10, ma20])):
        return None, None, None

    mas = [ma3, ma5, ma10, ma20]
    ma_spread = (max(mas) - min(mas)) / close * 100
    in_zone = min(mas) * 0.98 <= close <= max(mas) * 1.02
    score = max(0, min(100, 100 - ma_spread * 25))
    return round(ma_spread, 2), round(score, 1), in_zone


def label_sinyal(row):
    s = {}
    close = row.get('Close')
    ma3, ma5, ma10, ma20, ma50 = (row.get(k) for k in ['ma3', 'ma5', 'ma10', 'ma20', 'ma50'])
    if not any(pd.isna([close, ma3, ma5, ma10, ma20])):
        if close > ma3 and ma3 > ma5 and ma5 > ma10 and ma10 > ma20:
            s['Trend'] = 'UPTREND'
        elif close < ma3 and ma3 < ma5 and ma5 < ma10 and ma10 < ma20:
            s['Trend'] = 'DOWNTREND'
        else:
            s['Trend'] = 'SIDEWAYS'
        s['vs_MA20'] = f"{(close - ma20) / ma20 * 100:+.1f}%"
        s['vs_MA50'] = f"{(close - ma50) / ma50 * 100:+.1f}%" if not pd.isna(ma50) else 'N/A'

    k, d = row.get('stoch_k'), row.get('stoch_d')
    if not any(pd.isna([k, d])):
        s['Stoch'] = 'OVERBOUGHT' if k > 80 else ('OVERSOLD' if k < 20 else 'NETRAL')
        s['Stoch_cross'] = 'K>D' if k > d else 'K<D'

    bb_pct, bb_w = row.get('bb_pct'), row.get('bb_width')
    if not pd.isna(bb_pct):
        if bb_pct > 95:
            s['BB'] = 'ATAS BB'
        elif bb_pct > 70:
            s['BB'] = 'Upper Zone'
        elif bb_pct < 5:
            s['BB'] = 'BAWAH BB'
        elif bb_pct < 30:
            s['BB'] = 'Lower Zone'
        else:
            s['BB'] = 'Mid Zone'
        s['BB_Squeeze'] = 'SQUEEZE' if (not pd.isna(bb_w) and bb_w < 5) else '-'

    macd, msig = row.get('macd'), row.get('macd_sig')
    if not any(pd.isna([macd, msig])):
        pos = 'ATAS_0' if macd > 0 else 'BAWAH_0'
        cross = 'GOLDEN' if macd > msig else 'DEATH'
        s['MACD'] = f"{pos}|{cross}"

    rsi = row.get('rsi')
    if not pd.isna(rsi):
        s['RSI'] = f"{rsi:.0f} " + ('OVERBOUGHT' if rsi > 70 else ('OVERSOLD' if rsi < 30 else 'NETRAL'))

    g = row.get('guro_ii')
    if not pd.isna(g):
        lbl = 'STRONG BULL' if g > 7 else ('BULLISH' if g > 5 else ('NETRAL' if g > 3 else 'BEARISH'))
        s['Gurumology'] = f"{g:.1f} {lbl}"

    spk = row.get('vol_spike')
    if not pd.isna(spk):
        s['Vol_Spike'] = f"{spk:.1f}x {'SPIKE!' if spk > 2 else ''}".strip()

    obv, obv_ma5 = row.get('obv'), row.get('obv_ma5')
    if not any(pd.isna([obv, obv_ma5])):
        s['OBV'] = 'OBV>MA5 (akumulasi)' if obv > obv_ma5 else 'OBV<MA5 (distribusi)'

    return s


def hitung_ms_signal(row):
    macd, msig = row.get('macd'), row.get('macd_sig')
    k, d = row.get('stoch_k'), row.get('stoch_d')

    macd_sign = None
    if not pd.isna(macd) and not pd.isna(msig):
        macd_sign = '+' if macd > msig else '-'
    stoch_sign = None
    if not pd.isna(k) and not pd.isna(d):
        stoch_sign = '+' if k > d else '-'

    if macd_sign is None and stoch_sign is None:
        return ''
    if macd_sign is None:
        return f'S{stoch_sign}'
    if stoch_sign is None:
        return f'M{macd_sign}'
    if macd_sign == stoch_sign:
        return f'MS{macd_sign}'
    return f'M{macd_sign} S{stoch_sign}'


def download_dengan_retry(ticker, period, max_retry=2, delay_retry=2.0):
    last_exception = None
    for attempt in range(1, max_retry + 1):
        try:
            raw = yf.download(ticker, period=period, interval='1d',
                               progress=False, auto_adjust=True, threads=False)
            if raw is not None and not raw.empty and len(raw) >= 30:
                return raw, None
            last_exception = 'kosong/data<30baris'
        except Exception as e:
            last_exception = str(e)[:80]
        if attempt < max_retry:
            time.sleep(delay_retry + random.uniform(0, 1.0))
    return None, last_exception


def scan_satu(ticker, cfg):
    kode = ticker.replace('.JK', '')
    try:
        time.sleep(random.uniform(0, cfg['delay_antar_saham']))
        raw, err = download_dengan_retry(ticker, cfg['periode_data'],
                                          cfg['max_retry'], cfg['delay_retry'])
        if raw is None:
            return None, kode, f'{kode}({err})'

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume']).copy()
        if len(raw) < 30:
            return None, kode, f'{kode}(<30 baris)'

        df = hitung_semua_indikator(raw)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        close = float(last['Close'])
        change_pct = (close - float(prev['Close'])) / float(prev['Close']) * 100

        value_ma20 = float(df['value_miliar'].rolling(20).mean().iloc[-1])
        if pd.isna(value_ma20) or value_ma20 < cfg['min_value_miliar']:
            return None, kode, f'{kode}(val_avg={value_ma20:.2f}M<{cfg["min_value_miliar"]}M)'

        ma_spread, score, in_zone = hitung_superketat(last, cfg['superketat_threshold'])
        if score is None or score < cfg['min_score']:
            return None, kode, None

        ma20_val = float(last['ma20'])
        batas_atas = ma20_val * (1 + cfg['max_close_above_ma20'] / 100)
        dist_ma20_pct = (close - ma20_val) / ma20_val * 100

        if cfg['require_close_above_ma20']:
            if close <= ma20_val:
                return None, kode, f'{kode}(close<=MA20)'
            if close > batas_atas:
                return None, kode, f'{kode}(close terlalu jauh di atas MA20)'

        sinyal = label_sinyal(last)
        tgl = df.index[-1].strftime('%Y-%m-%d')

        hasil = {
            'Kode': kode, 'Tanggal': tgl, 'Close': int(close), 'Chg%': round(change_pct, 2),
            'Volume': int(last['Volume']), 'Value_Miliar': round(float(last['value_miliar']), 2),
            'MA3': round(float(last['ma3']), 0), 'MA5': round(float(last['ma5']), 0),
            'MA10': round(float(last['ma10']), 0), 'MA20': round(float(last['ma20']), 0),
            'MA_Spread%': ma_spread, 'SK_Score': score, 'InZone': 'Y' if in_zone else 'N',
            'vs_MA50': sinyal.get('vs_MA50', 'N/A'), 'vs_MA20%': round(dist_ma20_pct, 2),
            'Trend': sinyal.get('Trend', 'N/A'), 'BB': sinyal.get('BB', 'N/A'),
            'BB_Width%': round(float(last['bb_width']), 1) if not pd.isna(last['bb_width']) else None,
            'BB_Squeeze': sinyal.get('BB_Squeeze', '-'),
            'Stoch_K': round(float(last['stoch_k']), 1) if not pd.isna(last['stoch_k']) else None,
            'Stoch_D': round(float(last['stoch_d']), 1) if not pd.isna(last['stoch_d']) else None,
            'Stoch_Zone': sinyal.get('Stoch', 'N/A'), 'Stoch_Cross': sinyal.get('Stoch_cross', 'N/A'),
            'MACD_Status': sinyal.get('MACD', 'N/A'), 'MS_Signal': hitung_ms_signal(last),
            'RSI': sinyal.get('RSI', 'N/A'), 'Gurumology_II_EST': sinyal.get('Gurumology', 'N/A'),
            'Vol_Spike': sinyal.get('Vol_Spike', 'N/A'), 'OBV_Signal': sinyal.get('OBV', 'N/A'),
            'ATR%': round(float(last['atr_pct']), 2) if not pd.isna(last.get('atr_pct')) else None,
        }
        return hasil, kode, None
    except Exception as e:
        return None, kode, f'{kode}({str(e)[:60]})'


def run_scan(tickers, cfg, progress_cb=None):
    """Jalankan scan paralel. progress_cb(done, total, kode) dipanggil tiap saham selesai."""
    hasil_scan, gagal = [], []
    total = len(tickers)
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=cfg['max_workers']) as executor:
        futures = {executor.submit(scan_satu, t, cfg): t for t in tickers}
        for future in as_completed(futures):
            hasil, kode, err_msg = future.result()
            with lock:
                done += 1
                if hasil:
                    hasil_scan.append(hasil)
                if err_msg:
                    gagal.append(err_msg)
                if progress_cb:
                    progress_cb(done, total, kode)

    return hasil_scan, gagal
