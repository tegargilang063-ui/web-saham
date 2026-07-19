"""
Star Bottom Finder — saham yang sudah lama didiskon habis-habisan (downtrend
panjang, deep drawdown dari ATH), lalu basing/konsolidasi ketat di area
terendah, dan sekarang baru mulai breakout tajam dari dasar itu (dengan
konfirmasi volume + momentum).

CATATAN: ini kriteria baru yang dirancang untuk IDX (bukan port dari notebook
lama), berdasarkan pola yang terlihat di chart contoh (mis. KJEN): downtrend
multi-tahun -> basing ketat -> reversal candle besar + volume spike.
Semua threshold bisa diatur di UI.
"""
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf


def download_dengan_retry(ticker, period='5y', max_retry=2, delay_retry=2.0):
    last_exception = None
    for attempt in range(1, max_retry + 1):
        try:
            raw = yf.download(ticker, period=period, interval='1d',
                               progress=False, auto_adjust=True, threads=False)
            if raw is not None and not raw.empty and len(raw) >= 250:
                return raw, None
            last_exception = 'kosong/data<250baris'
        except Exception as e:
            last_exception = str(e)[:80]
        if attempt < max_retry:
            time.sleep(delay_retry + random.uniform(0, 1.0))
    return None, last_exception


def analisa_star_bottom(df: pd.DataFrame, cfg: dict):
    """Hitung metrik & skor Star Bottom dari satu saham. Return None kalau
    tidak memenuhi kriteria hard-filter (bukan kandidat star bottom sama sekali)."""
    basing_window = cfg['basing_window_hari']
    breakout_window = cfg['breakout_window_hari']

    if len(df) < basing_window + breakout_window + 30:
        return None

    close = df['Close']
    vol = df['Volume']

    # -- 1. Drawdown dari ATH (dalam periode data yang diambil, mis. 5 tahun) --
    ath_val = float(close.max())
    harga_now = float(close.iloc[-1])
    if ath_val <= 0:
        return None
    drawdown_pct = (harga_now - ath_val) / ath_val * 100  # nilai negatif

    if drawdown_pct > -cfg['min_drawdown_pct']:
        return None  # belum cukup "didiskon" dari ATH

    # -- 2. Basing: window sebelum fase breakout --
    basing_slice = close.iloc[-(basing_window + breakout_window):-breakout_window]
    breakout_slice_close = close.iloc[-breakout_window:]
    vol_basing_slice = vol.iloc[-(basing_window + breakout_window):-breakout_window]
    vol_breakout_slice = vol.iloc[-breakout_window:]

    basing_low = float(basing_slice.min())
    basing_high = float(basing_slice.max())
    if basing_low <= 0:
        return None
    basing_range_pct = (basing_high - basing_low) / basing_low * 100

    if basing_range_pct > cfg['max_basing_range_pct']:
        return None  # basing-nya tidak cukup ketat/sideways

    # -- 3. Breakout: kenaikan dari basing low --
    breakout_gain_pct = (harga_now - basing_low) / basing_low * 100
    if breakout_gain_pct < cfg['min_breakout_gain_pct']:
        return None  # belum breakout signifikan

    # -- 4. Volume spike saat breakout vs saat basing --
    vol_ma_basing = float(vol_basing_slice.mean())
    vol_ma_breakout = float(vol_breakout_slice.mean())
    vol_spike_ratio = (vol_ma_breakout / vol_ma_basing) if vol_ma_basing > 0 else None

    # -- 5. Momentum: RSI(14) & MACD(12,26,9) dari seluruh histori --
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss != 0, 100.0)  # tidak ada hari merah dalam 14 hari -> RSI = 100
    rsi = rsi.where(~((loss == 0) & (gain == 0)), 50.0)  # flat total -> netral
    rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    rsi_basing = rsi.iloc[-(basing_window + breakout_window):-breakout_window]
    rsi_min_basing = float(rsi_basing.min()) if not rsi_basing.empty else None

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_golden = bool(macd.iloc[-1] > macd_sig.iloc[-1])

    # -- Scoring 0-100 (rata-rata 5 komponen, masing2 dinormalisasi) --
    score_drawdown = min(100, abs(drawdown_pct) / cfg['min_drawdown_pct'] * 50)
    score_basing = max(0, 100 - basing_range_pct / cfg['max_basing_range_pct'] * 100)
    score_breakout = min(100, breakout_gain_pct / cfg['min_breakout_gain_pct'] * 50)
    score_volume = min(100, (vol_spike_ratio or 1) * 30) if vol_spike_ratio else 0
    score_momentum = ((30 if macd_golden else 0)
                       + (40 if (rsi_min_basing is not None and rsi_min_basing < 35) else 0)
                       + (30 if (rsi_now is not None and rsi_now > 40) else 0))

    composite = round(float(np.mean([score_drawdown, score_basing, score_breakout,
                                      score_volume, score_momentum])), 1)

    return {
        'ATH': round(ath_val, 0),
        'Harga_Now': round(harga_now, 0),
        'Drawdown_ATH%': round(drawdown_pct, 1),
        'Basing_Low': round(basing_low, 0),
        'Basing_Range%': round(basing_range_pct, 1),
        'Breakout_Gain%': round(breakout_gain_pct, 1),
        'Vol_Spike_Breakout': round(vol_spike_ratio, 2) if vol_spike_ratio else None,
        'RSI_Now': round(rsi_now, 1) if rsi_now is not None else None,
        'RSI_Min_Basing': round(rsi_min_basing, 1) if rsi_min_basing is not None else None,
        'MACD_Golden': 'Y' if macd_golden else 'N',
        'StarBottom_Score': composite,
    }


def scan_satu(ticker, cfg):
    kode = ticker.replace('.JK', '')
    try:
        time.sleep(random.uniform(0, cfg['delay_antar_saham']))
        raw, err = download_dengan_retry(ticker, cfg['periode_download'], cfg['max_retry'], cfg['delay_retry'])
        if raw is None:
            return None, kode, f'{kode}({err})'

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume']).copy()

        metrik = analisa_star_bottom(raw, cfg)
        if metrik is None:
            return None, kode, None  # bukan kandidat, bukan error

        close = raw['Close']
        value_ma20 = float((close * raw['Volume']).rolling(20).mean().iloc[-1] / 1e9)
        if pd.isna(value_ma20) or value_ma20 < cfg['min_value_miliar']:
            return None, kode, f'{kode}(likuiditas rendah)'

        hasil = {'Kode': kode, **metrik, 'Value_Miliar': round(value_ma20, 2)}
        return hasil, kode, None
    except Exception as e:
        return None, kode, f'{kode}({str(e)[:60]})'


def run_scan(tickers, cfg, progress_cb=None):
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
