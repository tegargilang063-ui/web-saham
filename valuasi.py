"""
Logika Analisa Valuasi Saham — diadaptasi dari notebook Analisa_Valuasi_Saham.ipynb
Persentil historis PBV/PE/PS/ROE/NPM dsb vs rentang historis.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

TAHUN_HISTORIS = 10


def get_percentile_rank(value, hist_values):
    clean = [v for v in hist_values if v is not None and not np.isnan(v)]
    if not clean or len(clean) < 2:
        return None, None, None, None
    min_v, max_v = min(clean), max(clean)
    if max_v == min_v:
        return 50.0, 50.0, min_v, max_v
    pct_from_min = ((value - min_v) / (max_v - min_v)) * 100
    pct_from_min = max(0, min(100, pct_from_min))
    return round(pct_from_min, 2), round(100 - pct_from_min, 2), round(min_v, 2), round(max_v, 2)


def _get_quarterly_series(df, row_names):
    if df is None or df.empty:
        return {}
    for name in row_names:
        if name in df.index:
            s = df.loc[name].dropna()
            return dict(sorted(s.items(), key=lambda kv: kv[0]))
    return {}


def _shares_outstanding(stk, info):
    so = info.get('sharesOutstanding')
    if so:
        return so
    try:
        fast = stk.fast_info
        return fast.get('shares') or fast.get('shares_outstanding')
    except Exception:
        return None


def _detect_scale_factor(market_cap, harga, shares, equity_terbaru):
    if not market_cap or not equity_terbaru or equity_terbaru == 0:
        return 1.0
    implied_pbv = market_cap / equity_terbaru
    if implied_pbv <= 0:
        return 1.0
    if 0.01 <= implied_pbv <= 50:
        return 1.0
    for kurs in [15000, 15500, 16000]:
        adj_pbv = market_cap / (equity_terbaru * kurs)
        if 0.01 <= adj_pbv <= 50:
            return kurs
    return 1.0


def _is_wajar(metrik, val):
    if val is None:
        return False
    batas = {'PBV': (-100, 1000), 'PE': (-1000, 1000), 'PS': (-100, 1000),
             'NPM': (-500, 100), 'ROE': (-500, 500)}
    if metrik in batas:
        lo, hi = batas[metrik]
        return lo <= val <= hi
    return True


def fetch_full_data(kode, tahun_historis=TAHUN_HISTORIS):
    """Ambil data current + historis dari Yahoo Finance untuk satu saham."""
    ticker_jk = kode.upper() + '.JK'
    result = {'kode': kode.upper(), 'error': None}

    try:
        stk = yf.Ticker(ticker_jk)
        info = stk.info

        result['harga'] = info.get('currentPrice') or info.get('regularMarketPrice')
        result['nama'] = info.get('longName', kode.upper())
        result['sektor'] = info.get('sector', '-')
        result['group'] = info.get('industry', '-')
        market_cap = info.get('marketCap')
        result['mkt_cap_b'] = round(market_cap / 1e9, 1) if market_cap else None

        shares = _shares_outstanding(stk, info)
        harga = result['harga']

        result['cakupan_data'] = {}
        result['peringatan'] = []

        try:
            inc_q = stk.quarterly_financials
            bal_q = stk.quarterly_balance_sheet
        except Exception:
            inc_q, bal_q = None, None

        rev_q = _get_quarterly_series(inc_q, ['Total Revenue', 'TotalRevenue'])
        ni_q = _get_quarterly_series(inc_q, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'])
        eq_q = _get_quarterly_series(bal_q, ['Stockholders Equity', 'Total Stockholder Equity',
                                             'Total Equity Gross Minority Interest'])

        result['diagnostik_kuartal'] = {
            'jumlah_kuartal_revenue': len(rev_q),
            'jumlah_kuartal_net_income': len(ni_q),
            'jumlah_kuartal_equity': len(eq_q),
            'tanggal_kuartal_revenue': [d.strftime('%Y-%m-%d') for d in sorted(rev_q.keys())],
        }
        if len(rev_q) < 4 or len(ni_q) < 4:
            result['peringatan'].append(
                f"Yahoo Finance baru punya {len(rev_q)} kuartal Revenue & {len(ni_q)} kuartal "
                f"Net Income (butuh min 4 utk TTM/PE/PS). Wajar untuk saham baru IPO.")

        scale = 1.0
        if eq_q and market_cap:
            eq_terbaru = eq_q[max(eq_q.keys())]
            scale = _detect_scale_factor(market_cap, harga, shares, eq_terbaru)
            if scale != 1.0:
                result['peringatan'].append(
                    f"Terdeteksi mismatch satuan mata uang, dikoreksi otomatis dengan faktor {scale:,.0f}x.")
                rev_q = {k: v * scale for k, v in rev_q.items()}
                ni_q = {k: v * scale for k, v in ni_q.items()}
                eq_q = {k: v * scale for k, v in eq_q.items()}

        rev_terbaru = rev_q[max(rev_q.keys())] if rev_q else None
        ni_terbaru = ni_q[max(ni_q.keys())] if ni_q else None
        eq_terbaru = eq_q[max(eq_q.keys())] if eq_q else None
        bvps_terbaru = (eq_terbaru / shares) if (eq_terbaru and shares) else None

        ni_dates_sorted_all = sorted(ni_q.keys())
        rev_dates_sorted_all = sorted(rev_q.keys())
        eps_ttm_terbaru = None
        if shares and len(ni_dates_sorted_all) >= 4:
            eps_ttm_terbaru = sum(ni_q[d] for d in ni_dates_sorted_all[-4:]) / shares
        rev_ttm_terbaru = None
        if shares and len(rev_dates_sorted_all) >= 4:
            rev_ttm_terbaru = sum(rev_q[d] for d in rev_dates_sorted_all[-4:]) / shares

        def _rolling_ttm_total(qdict, dates_sorted):
            out = {}
            for i, tgl in enumerate(dates_sorted):
                if i >= 3:
                    out[tgl] = sum(qdict[d] for d in dates_sorted[i - 3:i + 1])
            return out

        ni_ttm_total_q = _rolling_ttm_total(ni_q, ni_dates_sorted_all)
        rev_ttm_total_q = _rolling_ttm_total(rev_q, rev_dates_sorted_all)
        ni_ttm_terbaru_total = ni_ttm_total_q[max(ni_ttm_total_q.keys())] if ni_ttm_total_q else None
        rev_ttm_terbaru_total = rev_ttm_total_q[max(rev_ttm_total_q.keys())] if rev_ttm_total_q else None

        cur = {
            'PBV': round(harga / bvps_terbaru, 2) if (harga and bvps_terbaru) else None,
            'PE': round(harga / eps_ttm_terbaru, 2) if (harga and eps_ttm_terbaru) else None,
            'PS': round(harga / rev_ttm_terbaru, 2) if (harga and rev_ttm_terbaru) else None,
            'NPM': round(ni_terbaru / rev_terbaru * 100, 2) if (ni_terbaru is not None and rev_terbaru) else None,
            'ROE': round(ni_terbaru / eq_terbaru * 100, 2) if (ni_terbaru is not None and eq_terbaru) else None,
            'EPS': round(eps_ttm_terbaru, 2) if eps_ttm_terbaru else None,
            'Revenue_B': round(rev_terbaru / 1e9, 1) if rev_terbaru else None,
            'NetIncome_B': round(ni_terbaru / 1e9, 1) if ni_terbaru is not None else None,
            'DivYield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 0.0,
            'Price10Y': harga,
            'NetIncome_TTM_B': round(ni_ttm_terbaru_total / 1e9, 1) if ni_ttm_terbaru_total is not None else None,
            'Revenue_TTM_B': round(rev_ttm_terbaru_total / 1e9, 1) if rev_ttm_terbaru_total else None,
        }

        for k in ['PBV', 'PE', 'PS', 'NPM', 'ROE']:
            if cur.get(k) is not None and not _is_wajar(k, cur[k]):
                result['peringatan'].append(f"{k}: nilai {cur[k]} di luar rentang wajar -> ditandai N/A")
                cur[k] = None

        result['current'] = cur
        hist = {k: [] for k in cur.keys()}

        for tgl in sorted(set(rev_q) & set(ni_q)):
            rev, ni = rev_q.get(tgl), ni_q.get(tgl)
            if rev and ni is not None and rev != 0:
                v = round(ni / rev * 100, 2)
                if _is_wajar('NPM', v):
                    hist['NPM'].append(v)
        for tgl in sorted(set(ni_q) & set(eq_q)):
            ni, eq = ni_q.get(tgl), eq_q.get(tgl)
            if ni is not None and eq and eq != 0:
                v = round(ni / eq * 100, 2)
                if _is_wajar('ROE', v):
                    hist['ROE'].append(v)
        if rev_q:
            hist['Revenue_B'] = [round(v / 1e9, 1) for v in rev_q.values()]
        if ni_q:
            hist['NetIncome_B'] = [round(v / 1e9, 1) for v in ni_q.values()]
        if ni_ttm_total_q:
            hist['NetIncome_TTM_B'] = [round(v / 1e9, 1) for v in ni_ttm_total_q.values()]
        if rev_ttm_total_q:
            hist['Revenue_TTM_B'] = [round(v / 1e9, 1) for v in rev_ttm_total_q.values()]

        bvps_q = {}
        if shares and shares != 0:
            for tgl, eq in eq_q.items():
                bvps_q[tgl] = eq / shares

        eps_ttm_q, rev_ttm_q = {}, {}
        if shares and shares != 0:
            for i, tgl in enumerate(ni_dates_sorted_all):
                if i >= 3:
                    eps_ttm_q[tgl] = sum(ni_q[d] for d in ni_dates_sorted_all[i - 3:i + 1]) / shares
            for i, tgl in enumerate(rev_dates_sorted_all):
                if i >= 3:
                    rev_ttm_q[tgl] = sum(rev_q[d] for d in rev_dates_sorted_all[i - 3:i + 1]) / shares

        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365 * tahun_historis)
            hist_price = stk.history(start=start_date.strftime('%Y-%m-%d'),
                                      end=end_date.strftime('%Y-%m-%d'),
                                      interval='1mo', auto_adjust=False)
            if hist_price.empty:
                hist_price = stk.history(period='max', interval='1mo', auto_adjust=False)

            if not hist_price.empty:
                closes = hist_price['Close'].dropna()
                hist['Price10Y'] = closes.tolist()
                result['cakupan_data']['Price'] = (closes.index.min().strftime('%b %Y'),
                                                    closes.index.max().strftime('%b %Y'))

                def _nearest_quarter_value(qdict, tanggal):
                    valid = [d for d in qdict if d <= tanggal]
                    return qdict[max(valid)] if valid else None

                pbv_real, pe_real, ps_real = [], [], []
                for tgl, harga_bulan in closes.items():
                    tgl_naive = tgl.tz_localize(None) if tgl.tzinfo else tgl
                    bvps = _nearest_quarter_value(bvps_q, tgl_naive)
                    epst = _nearest_quarter_value(eps_ttm_q, tgl_naive)
                    revps = _nearest_quarter_value(rev_ttm_q, tgl_naive)
                    if bvps and bvps != 0:
                        v = round(harga_bulan / bvps, 2)
                        if _is_wajar('PBV', v):
                            pbv_real.append(v)
                    if epst and epst != 0:
                        v = round(harga_bulan / epst, 2)
                        if _is_wajar('PE', v):
                            pe_real.append(v)
                    if revps and revps != 0:
                        v = round(harga_bulan / revps, 2)
                        if _is_wajar('PS', v):
                            ps_real.append(v)

                if len(pbv_real) >= 2:
                    hist['PBV'] = pbv_real
                    result['cakupan_data']['PBV'] = f'{len(pbv_real)} bulan (riil, dari Book Value aktual)'
                if len(pe_real) >= 2:
                    hist['PE'] = pe_real
                    result['cakupan_data']['PE'] = f'{len(pe_real)} bulan (riil, dari EPS TTM aktual)'
                if len(ps_real) >= 2:
                    hist['PS'] = ps_real
                    result['cakupan_data']['PS'] = f'{len(ps_real)} bulan (riil, dari Revenue TTM aktual)'
                if eps_ttm_q:
                    hist['EPS'] = [round(v, 2) for v in eps_ttm_q.values()]
        except Exception:
            pass

        for k in hist:
            if cur.get(k) is not None:
                hist[k].append(cur[k])

        result['hist'] = hist

        pct = {}
        for k in cur:
            val = cur.get(k)
            h = hist.get(k, [])
            if val is not None and len(h) >= 2:
                lo, hi, mn, mx = get_percentile_rank(val, h)
                pct[k] = {'val': val, 'pct_lo': lo, 'pct_hi': hi, 'min': mn, 'max': mx}
            else:
                pct[k] = {'val': val, 'pct_lo': None, 'pct_hi': None, 'min': None, 'max': None}
        result['pct'] = pct

        pbv_pct = pct.get('PBV', {})
        pbv_min, pbv_max = pbv_pct.get('min'), pbv_pct.get('max')
        if bvps_terbaru and pbv_min is not None and pbv_max is not None:
            result['harga_implied_pbv_min'] = round(pbv_min * bvps_terbaru)
            result['harga_implied_pbv_max'] = round(pbv_max * bvps_terbaru)
        else:
            result['harga_implied_pbv_min'] = None
            result['harga_implied_pbv_max'] = None

        def _growth(qdict, dates_sorted, lag):
            if len(dates_sorted) <= lag:
                return None
            terbaru = qdict[dates_sorted[-1]]
            pembanding = qdict[dates_sorted[-1 - lag]]
            if pembanding in (0, None) or terbaru is None:
                return None
            return round((terbaru - pembanding) / abs(pembanding) * 100, 1)

        result['growth'] = {
            'revenue_qoq': _growth(rev_q, rev_dates_sorted_all, 1),
            'revenue_yoy': _growth(rev_q, rev_dates_sorted_all, 4),
            'netincome_qoq': _growth(ni_q, ni_dates_sorted_all, 1),
            'netincome_yoy': _growth(ni_q, ni_dates_sorted_all, 4),
        }

    except Exception as e:
        result['error'] = str(e)

    return result


def bar_html(pct, cakupan, key, label, fmt_val='{:.2f}', unit='', color_reverse=False, extra_html=''):
    """Buat bar persentil HTML (gaya Stockbit/RTI)."""
    d = pct.get(key, {})
    v, pct_lo, pct_hi, mn, mx = d.get('val'), d.get('pct_lo'), d.get('pct_hi'), d.get('min'), d.get('max')

    if v is None:
        na_badge = ' <span style="color:#e74c3c;font-weight:bold;font-size:10px">ATL</span>' if color_reverse else ''
        return f'''<div style="margin-bottom:18px">
            <div style="text-align:center;font-size:13px;color:#aaa;margin-bottom:4px">{label}{na_badge}</div>
            <div style="text-align:center;color:#555;font-size:12px">N/A</div></div>'''

    val_fmt = fmt_val.format(v) + unit

    if pct_lo is not None:
        if color_reverse:
            bar_color = '#27ae60' if pct_lo >= 50 else '#c0392b'
        else:
            bar_color = '#27ae60' if pct_lo <= 40 else ('#e67e22' if pct_lo <= 70 else '#c0392b')
        bar_fill = pct_lo

        def fmt_num(n):
            if n is None:
                return 'N/A'
            if abs(n) >= 1000:
                return f'{n:,.0f}' + unit
            return fmt_val.format(n) + unit

        header = f'{pct_lo:.2f}% &lt;&lt; {val_fmt} &gt;&gt; {pct_hi:.2f}%'
        if pct_hi is not None and pct_hi <= 0.05:
            header += ' <span style="color:#27ae60;font-weight:bold;font-size:10px">ATH</span>'
        elif pct_lo is not None and pct_lo <= 0.05:
            header += ' <span style="color:#e74c3c;font-weight:bold;font-size:10px">ATL</span>'

        info_cakupan = cakupan.get(key, '')
        cakupan_html = (f'<div style="text-align:center;font-size:9px;color:#666;margin-top:2px">'
                         f'{info_cakupan}</div>') if info_cakupan else ''

        return f'''<div style="margin-bottom:20px">
            <div style="text-align:center;font-size:13px;color:#ccc;margin-bottom:5px;font-weight:bold">{label}</div>
            <div style="text-align:center;font-size:11px;color:#aaa;margin-bottom:6px">{header}</div>
            <div style="position:relative;background:#2c2c2c;border-radius:4px;height:28px;overflow:hidden">
                <div style="position:absolute;left:0;top:0;height:100%;width:{bar_fill}%;background:{bar_color};border-radius:4px"></div>
                <div style="position:absolute;left:8px;top:50%;transform:translateY(-50%);font-size:12px;font-weight:bold;color:white;z-index:2">{fmt_num(mn)}</div>
                <div style="position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:12px;font-weight:bold;color:white;z-index:2">{fmt_num(mx)}</div>
            </div>
            {cakupan_html}
            {extra_html}
        </div>'''
    return f'''<div style="margin-bottom:20px">
        <div style="text-align:center;font-size:13px;color:#ccc;margin-bottom:5px;font-weight:bold">{label}</div>
        <div style="text-align:center;font-size:11px;color:#aaa;margin-bottom:6px">Data historis tidak tersedia &mdash; nilai saat ini: {val_fmt}</div>
        <div style="background:#2c2c2c;border-radius:4px;height:28px;display:flex;align-items:center;justify-content:center">
            <span style="color:#aaa;font-size:12px">{val_fmt}</span>
        </div>
        {extra_html}
    </div>'''


def render_detail_card(data):
    """Render kartu detail lengkap 1 saham (mirip tampilan Stockbit/RTI) sebagai HTML."""
    harga = data.get('harga', 0) or 0
    pct = data.get('pct', {})
    cakupan = data.get('cakupan_data', {})
    now_str = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

    h_min, h_max = data.get('harga_implied_pbv_min'), data.get('harga_implied_pbv_max')
    if h_min is not None and h_max is not None:
        extra_pbv = f'''<div style="text-align:center;font-size:9px;color:#5599ff;margin-bottom:2px">Harga implied (PBV min&ndash;max &times; Book Value sekarang)</div>
            <div style="position:relative;height:16px">
                <span style="position:absolute;left:0;font-size:12px;font-weight:bold;color:#5599ff">Rp {h_min:,}</span>
                <span style="position:absolute;right:0;font-size:12px;font-weight:bold;color:#5599ff">Rp {h_max:,}</span>
            </div>'''
    else:
        extra_pbv = ''

    rentang_price = cakupan.get('Price')
    label_historis = f'Historis: {rentang_price[0]} – {rentang_price[1]}' if rentang_price else f'Historis: {TAHUN_HISTORIS} Tahun'

    mkt_cap = data.get('mkt_cap_b')
    mkt_cap_str = f"{mkt_cap:,}" if mkt_cap is not None else '?'

    html = f'''
    <div style="font-family:'Segoe UI',sans-serif; background:#1a1a2e; color:#eee;
                border-radius:12px; padding:20px 24px; max-width:820px; margin:10px 0">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
            <div>
                <span style="font-size:20px;font-weight:bold;color:#00d4ff">{data['kode']} {harga:,.0f}</span>
                <div style="font-size:11px;color:#888;margin-top:2px">{data.get('group', '-')}</div>
            </div>
            <div style="font-size:11px;color:#888;text-align:right">{now_str}<br>{label_historis}</div>
        </div>
        <hr style="border-color:#333;margin:10px 0 18px 0">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px 32px">
            <div>
                {bar_html(pct, cakupan, 'PBV', 'PBV', '{:.2f}', 'x', False, extra_pbv)}
                {bar_html(pct, cakupan, 'PE', 'PE', '{:.2f}', 'x', False)}
                {bar_html(pct, cakupan, 'PS', 'PS', '{:.2f}', 'x', False)}
                {bar_html(pct, cakupan, 'DivYield', 'Dividend Yield', '{:.2f}', '%', True)}
                {bar_html(pct, cakupan, 'Price10Y', 'Price (Histori)', '{:,.0f}', '', False)}
            </div>
            <div>
                {bar_html(pct, cakupan, 'NPM', 'NPM', '{:.2f}', '%', True)}
                {bar_html(pct, cakupan, 'NetIncome_B', 'Net Income (Kuartal)', '{:,.1f}', 'B', True)}
                {bar_html(pct, cakupan, 'NetIncome_TTM_B', 'Net Income TTM', '{:,.1f}', 'B', True)}
                {bar_html(pct, cakupan, 'Revenue_B', 'Revenue (Kuartal)', '{:,.1f}', 'B', True)}
                {bar_html(pct, cakupan, 'Revenue_TTM_B', 'Revenue TTM', '{:,.1f}', 'B', True)}
                {bar_html(pct, cakupan, 'EPS', 'EPS (TTM)', '{:,.2f}', '', True)}
                {bar_html(pct, cakupan, 'ROE', 'ROE', '{:.2f}', '%', True)}
            </div>
        </div>
        <div style="margin-top:10px;padding:10px;background:#16213e;border-radius:8px;
                    display:flex;justify-content:space-around;font-size:11px;color:#aaa">
            <div style="text-align:center"><div style="color:#888;margin-bottom:2px">Market Cap</div>
                <div style="color:#eee;font-weight:bold">Rp {mkt_cap_str}B</div></div>
            <div style="text-align:center"><div style="color:#888;margin-bottom:2px">Sektor</div>
                <div style="color:#eee;font-weight:bold">{data.get('sektor', '-')}</div></div>
            <div style="text-align:center"><div style="color:#888;margin-bottom:2px">🟢 Bar Hijau</div>
                <div style="color:#27ae60;font-weight:bold">Valuasi Murah / Profit Tinggi</div></div>
            <div style="text-align:center"><div style="color:#888;margin-bottom:2px">🔴 Bar Merah</div>
                <div style="color:#c0392b;font-weight:bold">Valuasi Mahal / Profit Rendah</div></div>
        </div>
    </div>'''
    return html


def hitung_composite_score(d):
    pct = d.get('pct', {})

    def safe(key, field):
        return pct.get(key, {}).get(field)

    pbv_pct, pe_pct = safe('PBV', 'pct_lo'), safe('PE', 'pct_lo')
    roe_pct, npm_pct = safe('ROE', 'pct_lo'), safe('NPM', 'pct_lo')
    return {
        'pbv': (100 - pbv_pct) if pbv_pct is not None else None,
        'pe': (100 - pe_pct) if pe_pct is not None else None,
        'roe': roe_pct, 'npm': npm_pct,
        'ni_raw': safe('NetIncome_B', 'val'), 'rev_raw': safe('Revenue_B', 'val'),
    }


def minmax_normalize(kv_list):
    if not kv_list:
        return {}
    vals = [v for _, v in kv_list]
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 50.0 for k, _ in kv_list}
    return {k: round((v - mn) / (mx - mn) * 100, 2) for k, v in kv_list}


WEIGHTS = {'pbv': 0.30, 'pe': 0.10, 'roe': 0.20, 'npm': 0.20, 'ni': 0.10, 'rev': 0.10}


def compute_top_n(all_data, n=15):
    """Hitung composite score & kembalikan top-N saham + df ringkasan semua saham."""
    scored = []
    for d in all_data:
        s = hitung_composite_score(d)
        s['kode'] = d['kode']
        s['data'] = d
        scored.append(s)

    ni_list = [(s['kode'], s['ni_raw']) for s in scored if s['ni_raw'] is not None]
    rev_list = [(s['kode'], s['rev_raw']) for s in scored if s['rev_raw'] is not None]
    ni_norm = minmax_normalize(ni_list)
    rev_norm = minmax_normalize(rev_list)

    final_scores = []
    for s in scored:
        kode = s['kode']
        komponen = {'pbv': s['pbv'], 'pe': s['pe'], 'roe': s['roe'], 'npm': s['npm'],
                    'ni': ni_norm.get(kode), 'rev': rev_norm.get(kode)}
        total_w, total_sc = 0.0, 0.0
        for k, w in WEIGHTS.items():
            if komponen[k] is not None:
                total_sc += komponen[k] * w
                total_w += w
        composite = round(total_sc / total_w, 2) if total_w > 0 else 0.0
        final_scores.append({'kode': kode, 'composite': composite, 'komponen': komponen, 'data': s['data']})

    final_scores.sort(key=lambda x: x['composite'], reverse=True)
    top = final_scores[:n]
    return top, final_scores


def _medal(rank):
    return {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f'#{rank}')


def _warna_score(s):
    if s >= 70:
        return '#27ae60'
    if s >= 50:
        return '#e67e22'
    return '#c0392b'


def render_top_cards(top_list):
    """Render kartu Top-N rekomendasi (HTML), mirip tampilan notebook asli."""

    def fv(v, fmt='{:.2f}', suffix=''):
        return fmt.format(v) + suffix if v is not None else 'N/A'

    card_html = ''
    for rank, item in enumerate(top_list, 1):
        kode = item['kode']
        score = item['composite']
        d = item['data']
        pct = d.get('pct', {})
        k = item['komponen']

        harga = d.get('harga') or 0
        sektor = d.get('sektor', '-')
        mkt_cap = d.get('mkt_cap_b')
        ni_val = pct.get('NetIncome_B', {}).get('val')
        rev_val = pct.get('Revenue_B', {}).get('val')
        pbv_val = pct.get('PBV', {}).get('val')
        pe_val = pct.get('PE', {}).get('val')
        roe_val = pct.get('ROE', {}).get('val')
        npm_val = pct.get('NPM', {}).get('val')
        sc = _warna_score(score)

        card_html += f'''
        <div style="background:#1e1e2e;border:1px solid #2a2a3e;border-radius:12px;
                    padding:16px 20px;margin-bottom:14px;position:relative;
                    box-shadow:0 2px 8px rgba(0,0,0,0.4)">
            <div style="position:absolute;top:14px;right:16px;
                        background:{sc};color:white;border-radius:20px;
                        padding:4px 12px;font-size:13px;font-weight:bold">
                {score:.1f} / 100
            </div>
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
                <div style="font-size:22px">{_medal(rank)}</div>
                <div>
                    <div style="font-size:16px;font-weight:bold;color:#00d4ff">{kode}</div>
                    <div style="font-size:11px;color:#888">{sektor}</div>
                </div>
                <div style="margin-left:auto;text-align:right;padding-right:90px">
                    <div style="font-size:15px;font-weight:bold;color:#eee">Rp {harga:,.0f}</div>
                    <div style="font-size:11px;color:#888">Mkt Cap: Rp{fv(mkt_cap, '{:,.1f}')}B</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px 20px;
                        font-size:12px;border-top:1px solid #2a2a3e;padding-top:10px">
                <div><span style="color:#888">PBV:</span>
                    <b style="color:#eee">{fv(pbv_val, '{:.2f}', 'x')}</b>
                    <span style="color:#aaa;font-size:10px"> ({fv(k['pbv'], '{:.0f}', 'pt')})</span>
                </div>
                <div><span style="color:#888">PE:</span>
                    <b style="color:#eee">{fv(pe_val, '{:.2f}', 'x')}</b>
                    <span style="color:#aaa;font-size:10px"> ({fv(k['pe'], '{:.0f}', 'pt')})</span>
                </div>
                <div><span style="color:#888">Net Income:</span>
                    <b style="color:#27ae60">{fv(ni_val, '{:,.1f}', 'B')}</b>
                </div>
                <div><span style="color:#888">ROE:</span>
                    <b style="color:#eee">{fv(roe_val, '{:.1f}', '%')}</b>
                    <span style="color:#aaa;font-size:10px"> ({fv(k['roe'], '{:.0f}', 'pt')})</span>
                </div>
                <div><span style="color:#888">NPM:</span>
                    <b style="color:#eee">{fv(npm_val, '{:.1f}', '%')}</b>
                    <span style="color:#aaa;font-size:10px"> ({fv(k['npm'], '{:.0f}', 'pt')})</span>
                </div>
                <div><span style="color:#888">Revenue:</span>
                    <b style="color:#eee">{fv(rev_val, '{:,.1f}', 'B')}</b>
                </div>
            </div>
            <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;align-items:center">
                <span style="font-size:10px;color:#666">Skor komponen:</span>
                <span style="font-size:10px;color:#aaa">PBV {fv(k['pbv'], '{:.0f}')}</span>
                <span style="color:#444">|</span>
                <span style="font-size:10px;color:#aaa">PE {fv(k['pe'], '{:.0f}')}</span>
                <span style="color:#444">|</span>
                <span style="font-size:10px;color:#aaa">ROE {fv(k['roe'], '{:.0f}')}</span>
                <span style="color:#444">|</span>
                <span style="font-size:10px;color:#aaa">NPM {fv(k['npm'], '{:.0f}')}</span>
            </div>
        </div>'''

    header_html = f'''
    <div style="font-family:'Segoe UI',sans-serif;max-width:900px;margin:10px 0">
        <div style="background:#12122a;border-radius:12px;padding:16px 20px;margin-bottom:16px">
            <h3 style="color:#00d4ff;margin:0 0 8px 0">🏆 Top {len(top_list)} Rekomendasi Saham IDX</h3>
            <p style="color:#888;font-size:12px;margin:0">
                Composite Score = PBV (30%) + PE (10%) + ROE (20%) + NPM (20%) + Net Income (10%) + Revenue (10%)<br>
                <span style="color:#27ae60">■</span> Skor ≥70 &nbsp;
                <span style="color:#e67e22">■</span> Skor 50-70 &nbsp;
                <span style="color:#c0392b">■</span> Skor &lt;50 &nbsp;&nbsp;
                <i>Angka dalam kurung = skor komponen (0-100)</i>
            </p>
        </div>
        {card_html}
        <div style="background:#12122a;border-radius:8px;padding:10px 16px;font-size:11px;color:#666;margin-top:8px">
            ⚠️ Bukan rekomendasi investasi. Gunakan sebagai alat bantu analisis saja. Selalu lakukan riset mandiri.
        </div>
    </div>'''
    return header_html
