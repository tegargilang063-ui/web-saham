"""
IDX Toolkit — SUPERKETAT Scanner + Analisa Valuasi Saham
Gabungan 2 notebook Tegar jadi 1 web app (Streamlit).
"""
import time
import io

import pandas as pd
import streamlit as st

from stocks_list import STOCKS_IDX_ALL
import scanner
import valuasi

st.set_page_config(page_title="IDX Toolkit — Scanner & Valuasi", layout="wide", page_icon="📈")


def render_html(html: str):
    """st.markdown menganggap baris yang diawali banyak spasi sebagai code block,
    jadi HTML-nya tidak ter-render (muncul sebagai teks mentah). Hapus indentasi
    di awal tiap baris dulu sebelum dikirim ke markdown."""
    cleaned = "\n".join(line.lstrip() for line in html.split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)

st.title("📈 IDX Toolkit")
st.caption("SUPERKETAT Scanner (CIA style) + Analisa Valuasi Saham — data via Yahoo Finance. "
           "Bukan rekomendasi investasi, selalu DYOR.")

tab_scanner, tab_valuasi = st.tabs(["🔍 SUPERKETAT Scanner", "📊 Analisa Valuasi"])

# ============================================================
# TAB 1 — SUPERKETAT SCANNER
# ============================================================
with tab_scanner:
    st.subheader("🔍 IDX Stock Scanner — CIA Style (SUPERKETAT)")
    st.markdown(
        "Mencari saham IDX dimana MA3/5/10/20 berdekatan (konsolidasi sebelum breakout), "
        "dengan closing di atas MA20 tapi tidak terlalu jauh (zona ideal entry CIA style)."
    )

    with st.expander("⚙️ Konfigurasi", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            sk_threshold = st.slider("SUPERKETAT Threshold (%)", 1.0, 10.0, 3.0, 0.5)
            min_score = st.slider("Min SK Score", 0, 100, 40, 5)
        with c2:
            require_ma20 = st.checkbox("Wajib Close > MA20", value=True)
            max_above_ma20 = st.slider("Maks Close di atas MA20 (%)", 1.0, 10.0, 3.0, 0.5)
        with c3:
            periode = st.selectbox("Periode data historis", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
            min_value = st.number_input("Min nilai transaksi (Rp miliar/hari)", 0.1, 50.0, 1.0, 0.1)

        c4, c5 = st.columns(2)
        with c4:
            max_workers = st.slider("Jumlah thread paralel", 1, 8, 4)
        with c5:
            mode_universe = st.radio("Universe saham", ["Semua saham (±800)", "Custom / subset"], horizontal=True)

        if mode_universe == "Custom / subset":
            custom_list = st.text_area(
                "Kode saham (pisahkan koma, tanpa .JK)",
                value="BBCA, BBRI, BMRI, TLKM, ASII, UNVR, ICBP, KLBF",
            )
            tickers_sel = [f"{s.strip().upper()}.JK" for s in custom_list.split(",") if s.strip()]
        else:
            tickers_sel = [f"{s}.JK" for s in STOCKS_IDX_ALL]

        st.caption(f"Total saham akan di-scan: **{len(tickers_sel)}**. "
                   f"Estimasi waktu: ~{len(tickers_sel)*0.5/max_workers/60:.1f}-{len(tickers_sel)*1.5/max_workers/60:.1f} menit "
                   f"(tergantung koneksi & rate-limit Yahoo Finance).")

    run_btn = st.button("🚀 Jalankan Scan", type="primary", key="scan_btn")

    if run_btn:
        cfg = {
            "superketat_threshold": sk_threshold,
            "min_score": min_score,
            "require_close_above_ma20": require_ma20,
            "max_close_above_ma20": max_above_ma20,
            "periode_data": periode,
            "min_value_miliar": min_value,
            "max_workers": max_workers,
            "delay_antar_saham": 0.3,
            "max_retry": 2,
            "delay_retry": 2.0,
        }

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_cb(done, total, kode):
            progress_bar.progress(done / total)
            status_text.text(f"Scanning... {done}/{total} ({kode})")

        t0 = time.time()
        with st.spinner("Menjalankan scan..."):
            hasil_scan, gagal = scanner.run_scan(tickers_sel, cfg, progress_cb)
        elapsed = time.time() - t0

        progress_bar.progress(1.0)
        status_text.text(f"Selesai dalam {elapsed:.0f} detik.")

        st.session_state["sk_hasil"] = hasil_scan
        st.session_state["sk_gagal"] = gagal

    if "sk_hasil" in st.session_state:
        hasil_scan = st.session_state["sk_hasil"]
        gagal = st.session_state["sk_gagal"]

        if not hasil_scan:
            st.warning("❌ Tidak ada saham yang lolos kriteria. Coba naikkan threshold atau turunkan min score.")
        else:
            df_hasil = pd.DataFrame(hasil_scan).sort_values("SK_Score", ascending=False).reset_index(drop=True)
            st.success(f"✅ {len(df_hasil)} saham lolos kriteria SUPERKETAT (dari {len(gagal) + len(df_hasil)} diproses).")

            kolom_ringkas = ["Kode", "Close", "Chg%", "Value_Miliar", "SK_Score",
                              "Trend", "MS_Signal", "Vol_Spike"]
            df_ringkas = df_hasil[[c for c in kolom_ringkas if c in df_hasil.columns]]
            st.dataframe(df_ringkas, use_container_width=True, height=450)

            with st.expander("📋 Tabel lengkap (semua kolom indikator)"):
                st.dataframe(df_hasil, use_container_width=True, height=500)

            csv = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download CSV (semua kolom)", csv, "superketat_scan.csv", "text/csv")

            with st.expander("📖 Cara membaca kolom"):
                st.markdown("""
                | Kolom | Arti | Kondisi ideal |
                |---|---|---|
                | `MA_Spread%` | Jarak antar MA5-MA10-MA20 | < 2% = SUPERKETAT |
                | `SK_Score` | Skor kerapatan MA | > 80 = sangat ketat |
                | `BB_Squeeze` | BB Width < 5% | SQUEEZE = siap breakout |
                | `Stoch_K/D` | Stochastic (15,3,3) | K<20 oversold, K>D bullish |
                | `MACD_Status` | Posisi & cross MACD | ATAS_0\\|GOLDEN = terkuat |
                | `Gurumology_II_EST` | Indikator custom CIA | >5 bullish, <3 bearish |
                | `Vol_Spike` | Volume vs rata-rata 5 hari | >2x = konfirmasi kuat |
                """)
            st.caption("Inspired by: CIA - Chart Investor Academy (@tradingdiary2). Selalu DYOR.")

# ============================================================
# TAB 2 — ANALISA VALUASI
# ============================================================
with tab_valuasi:
    st.subheader("📊 IDX Stock Analyzer — Persentil Historis")
    st.markdown(
        "Posisi valuasi saat ini (PBV/PE/PS/ROE/NPM dll) dibanding rentang historisnya — "
        "mirip tampilan Stockbit/RTI."
    )

    sub_detail, sub_screen = st.tabs(["🔎 Detail Satu Saham", "🏆 Screening & Top Rekomendasi"])

    # ---- Sub-tab: Detail satu saham ----
    with sub_detail:
        c1, c2 = st.columns([2, 1])
        with c1:
            kode_input = st.text_input("Kode saham (tanpa .JK)", value="BBNI").strip().upper()
        with c2:
            tahun_hist = st.selectbox("Rentang historis (tahun)", [5, 10], index=1, key="tahun_detail")

        if st.button("🔄 Ambil Data", type="primary", key="detail_btn"):
            with st.spinner(f"Mengambil data {kode_input}... (mungkin 5-10 detik)"):
                data = valuasi.fetch_full_data(kode_input, tahun_historis=tahun_hist)
            st.session_state["val_detail"] = data

        if "val_detail" in st.session_state:
            data = st.session_state["val_detail"]
            if data.get("error"):
                st.error(f"❌ Error: {data['error']}")
            else:
                render_html(valuasi.render_detail_card(data))

                peringatan = data.get("peringatan", [])
                if peringatan:
                    with st.expander("⚠️ Peringatan / koreksi data otomatis"):
                        for p in peringatan:
                            st.write(f"- {p}")

                with st.expander("📌 Cakupan & diagnostik data"):
                    st.write("**Cakupan data riil per metrik:**")
                    for k, v in data.get("cakupan_data", {}).items():
                        st.write(f"- {k}: {v}")
                    diag = data.get("diagnostik_kuartal", {})
                    if diag:
                        st.write("**Diagnostik laporan keuangan:**")
                        st.write(f"- Kuartal Revenue tersedia: {diag.get('jumlah_kuartal_revenue')}")
                        st.write(f"- Kuartal Net Income tersedia: {diag.get('jumlah_kuartal_net_income')}")
                        st.write(f"- Kuartal Equity tersedia: {diag.get('jumlah_kuartal_equity')}")

    # ---- Sub-tab: Screening & Top rekomendasi ----
    with sub_screen:
        mode_universe_v = st.radio("Universe saham", ["Semua saham (±800, lambat)", "Custom / subset"],
                                    horizontal=True, key="val_universe_mode")
        if mode_universe_v == "Custom / subset":
            custom_list_v = st.text_area(
                "Kode saham (pisahkan koma, tanpa .JK)",
                value="BBCA, BBRI, BMRI, TLKM, ASII, UNVR, ICBP, KLBF, ADRO, PTBA",
                key="val_custom_list",
            )
            daftar_saham_v = [s.strip().upper() for s in custom_list_v.split(",") if s.strip()]
        else:
            daftar_saham_v = STOCKS_IDX_ALL

        top_n = st.slider("Jumlah Top Rekomendasi", 5, 30, 15, key="val_topn")
        st.caption(f"Total saham akan di-scan: **{len(daftar_saham_v)}**. "
                   f"Estimasi ~5-10 detik/saham (ambil data historis kuartalan + bulanan).")

        if st.button("🔄 Jalankan Screening", type="primary", key="screen_btn"):
            progress_bar_v = st.progress(0.0)
            status_v = st.empty()
            all_data = []
            total = len(daftar_saham_v)
            for i, kode in enumerate(daftar_saham_v):
                status_v.text(f"[{i+1}/{total}] Mengambil {kode}...")
                d = valuasi.fetch_full_data(kode)
                if not d.get("error"):
                    all_data.append(d)
                progress_bar_v.progress((i + 1) / total)
                time.sleep(0.3)
            status_v.text(f"Selesai: {len(all_data)}/{total} saham berhasil diambil.")
            st.session_state["val_all_data"] = all_data

        if "val_all_data" in st.session_state:
            all_data = st.session_state["val_all_data"]

            if not all_data:
                st.warning("Tidak ada data berhasil diambil.")
            else:
                top, final_scores = valuasi.compute_top_n(all_data, n=top_n)

                render_html(valuasi.render_top_cards(top))

                st.markdown("---")
                st.markdown("#### 📋 Tabel Ringkasan Semua Saham")

                rows = []
                for d in all_data:
                    pct = d.get("pct", {})

                    def pct_str(key):
                        p = pct.get(key, {})
                        lo = p.get("pct_lo")
                        return f"{lo:.0f}%" if lo is not None else "N/A"

                    rows.append({
                        "Kode": d["kode"], "Harga": d.get("harga"),
                        "PBV": pct.get("PBV", {}).get("val"), "PBV %hist": pct_str("PBV"),
                        "PE": pct.get("PE", {}).get("val"), "PE %hist": pct_str("PE"),
                        "PS": pct.get("PS", {}).get("val"),
                        "ROE (%)": pct.get("ROE", {}).get("val"), "ROE %hist": pct_str("ROE"),
                        "NPM (%)": pct.get("NPM", {}).get("val"), "NPM %hist": pct_str("NPM"),
                        "EPS TTM": pct.get("EPS", {}).get("val"),
                        "Revenue (B)": pct.get("Revenue_B", {}).get("val"),
                        "Net Income (B)": pct.get("NetIncome_B", {}).get("val"),
                        "Div Yield (%)": pct.get("DivYield", {}).get("val"),
                    })
                df_sum = pd.DataFrame(rows)
                st.dataframe(df_sum, use_container_width=True, height=400)

                # Screening: murah + profit bagus
                st.markdown("---")
                st.markdown("#### 🔍 Screening: Murah + Profit Bagus")
                st.caption("Kriteria: PBV/PE ≤ persentil 40% historis (murah) + ROE/NPM ≥ persentil 50% "
                           "historis (profit bagus) + Net Income positif")
                screen_rows = []
                for d in all_data:
                    pct = d.get("pct", {})
                    pbv_pct = pct.get("PBV", {}).get("pct_lo")
                    pe_pct = pct.get("PE", {}).get("pct_lo")
                    roe_pct = pct.get("ROE", {}).get("pct_lo")
                    npm_pct = pct.get("NPM", {}).get("pct_lo")
                    ni_val = pct.get("NetIncome_B", {}).get("val")

                    valuasi_murah = (pbv_pct is not None and pbv_pct <= 40) or (pe_pct is not None and pe_pct <= 40)
                    profit_bagus = (roe_pct is not None and roe_pct >= 50) or (npm_pct is not None and npm_pct >= 50)
                    laba_positif = ni_val is not None and ni_val > 0

                    if valuasi_murah and profit_bagus and laba_positif:
                        screen_rows.append({
                            "Kode": d["kode"], "Harga": d.get("harga"),
                            "PBV": pct.get("PBV", {}).get("val"),
                            "PE": pct.get("PE", {}).get("val"),
                            "ROE (%)": pct.get("ROE", {}).get("val"),
                            "NPM (%)": pct.get("NPM", {}).get("val"),
                            "Net Inc (B)": ni_val,
                            "Revenue (B)": pct.get("Revenue_B", {}).get("val"),
                        })

                if screen_rows:
                    df_screen = pd.DataFrame(screen_rows)
                    st.success(f"{len(df_screen)} saham lolos screening.")
                    st.dataframe(df_screen, use_container_width=True)
                else:
                    st.info("Tidak ada saham yang lolos kriteria screening di universe ini.")

                # Export Excel
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df_sum.to_excel(writer, sheet_name="Semua Saham", index=False)
                    if screen_rows:
                        pd.DataFrame(screen_rows).to_excel(writer, sheet_name="Hasil Screening", index=False)
                    pd.DataFrame([{
                        "Rank": i + 1, "Kode": t["kode"], "Composite Score": t["composite"],
                        "Harga": t["data"].get("harga"),
                    } for i, t in enumerate(top)]).to_excel(writer, sheet_name="Top Rekomendasi", index=False)
                st.download_button("💾 Download Excel (semua sheet)", buf.getvalue(),
                                    "IDX_Analyzer.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("⚠️ Bukan rekomendasi investasi. Data dari Yahoo Finance, bisa delay/tidak sepenuhnya akurat. "
           "Selalu lakukan riset mandiri sebelum mengambil keputusan investasi.")
