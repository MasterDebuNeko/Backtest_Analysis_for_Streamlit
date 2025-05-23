import pandas as pd
import numpy as np
import streamlit as st # Import Streamlit

# 📌 Utility Functions (คงเดิมจากไฟล์ของท่านพี่)
def clean_number(val):
    try:
        return float(str(val).replace(',', '').replace(' ', ''))
    except Exception:
        return np.nan

def validate_stop_loss(stop_loss_pct):
    try:
        pct = float(stop_loss_pct)
        if not (0 < pct < 1):
            raise ValueError()
        return pct
    except Exception:
        raise ValueError("stop_loss_pct ต้องเป็นตัวเลข 0 < x < 1 เช่น 0.002 (0.2%)")

def safe_divide(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((denominator == 0) | pd.isnull(denominator), np.nan, numerator / denominator)
    return result

# 📌 Core Function: Calculate R-Multiple and Risk
def calc_r_multiple_and_risk(xls_path_or_buffer, stop_loss_pct): # Modified to accept path or buffer
    stop_loss_pct = validate_stop_loss(stop_loss_pct)
    # --- Load Data ---
    try:
        # If xls_path_or_buffer is a string (path), use it. Otherwise, assume it's a buffer (UploadedFile object).
        df_trades = pd.read_excel(xls_path_or_buffer, sheet_name='List of trades')
        df_props  = pd.read_excel(xls_path_or_buffer, sheet_name='Properties')
    except Exception as e:
        raise RuntimeError(f"โหลดไฟล์ผิดพลาด: {e}")

    # ... (ส่วนที่เหลือของฟังก์ชัน calc_r_multiple_and_risk ของท่านพี่)
    # ... (ตรวจสอบให้แน่ใจว่า print statements ถูกเปลี่ยนเป็น st.info, st.warning หรือ st.error ตามความเหมาะสม)
    # ตัวอย่างการเปลี่ยน print เป็น st.warning
    # if df_entry_orig.empty:
    #     st.warning("⚠️ ไม่พบรายการ Entry trades ในไฟล์ Excel.") # เปลี่ยนจาก print
    # if df_exit_orig.empty:
    #     st.warning("⚠️ ไม่พบรายการ Exit trades ในไฟล์ Excel.")  # เปลี่ยนจาก print

    # --- Extract Point Value ---
    try:
        point_value_row = df_props[df_props.iloc[:, 0].astype(str).str.contains("point value", case=False, na=False)]
        if point_value_row.empty:
            raise ValueError("ไม่พบ Point Value ใน properties sheet")
        point_value = clean_number(point_value_row.iloc[0, 1])
        if np.isnan(point_value) or point_value <= 0:
            raise ValueError("Point Value ผิดปกติ")
    except Exception as e:
         raise ValueError(f"ข้อผิดพลาดในการดึง Point Value: {e}")

    # --- Prepare Entry & Exit DataFrames ---
    try:
        df_entry_orig = df_trades[df_trades['Type'].str.contains("Entry", case=False, na=False)].copy()
        df_exit_orig  = df_trades[df_trades['Type'].str.contains("Exit", case=False, na=False)].copy()
        if df_entry_orig.empty:
             st.warning("⚠️ ไม่พบรายการ Entry trades ในไฟล์ Excel.")
        if df_exit_orig.empty:
             st.warning("⚠️ ไม่พบรายการ Exit trades ในไฟล์ Excel.")
    except KeyError as e:
        raise KeyError(f"ไม่พบคอลัมน์ Type ใน trades: {e}")
    except Exception as e:
        raise RuntimeError(f"ข้อผิดพลาดในการกรอง Entry/Exit trades: {e}")

    # ✅ Convert Date/Time columns
    try:
        df_entry = df_entry_orig.copy()
        df_exit = df_exit_orig.copy()
        df_entry['Date/Time'] = pd.to_datetime(df_entry['Date/Time'])
        df_exit['Date/Time'] = pd.to_datetime(df_exit['Date/Time'])
    except KeyError as e:
        raise KeyError(f"ไม่พบคอลัมน์ Date/Time: {e}")
    except Exception as e:
        raise ValueError(f"รูปแบบ Date/Time ไม่ถูกต้อง: {e}")

    for col in ['Price USD', 'Quantity']:
        if col not in df_entry.columns:
            raise KeyError(f"ไม่พบคอลัมน์ {col} ใน Entry")
        df_entry[col] = df_entry[col].map(clean_number)
        if col in df_exit.columns:
            df_exit[col] = df_exit[col].map(clean_number)

    # --- Calculate Risk USD ---
    if not df_entry.empty:
        df_entry['Risk USD'] = (
            df_entry['Price USD'] *
            stop_loss_pct *
            df_entry['Quantity'] *
            point_value
        )
    else:
         st.warning("⚠️ df_entry ว่างเปล่า ไม่สามารถคำนวณ Risk USD ได้")
         df_entry['Risk USD'] = np.nan

    # --- Check for Duplicate Trade Numbers ---
    if 'Trade #' not in df_entry.columns or 'Trade #' not in df_exit.columns:
        raise KeyError("ไม่พบคอลัมน์ 'Trade #'")
    if not df_entry.empty and df_entry['Trade #'].duplicated().any():
        raise ValueError("Trade # ใน Entry มีค่าซ้ำ")
    if not df_exit.empty and df_exit['Trade #'].duplicated().any():
        raise ValueError("Trade # ใน Exit มีค่าซ้ำ")

    # --- Map Risk USD to Exit Trades ---
    n_missing_risk = 0
    if not df_exit.empty and not df_entry.empty:
        risk_map = df_entry.set_index('Trade #')['Risk USD']
        df_exit['Risk USD'] = df_exit['Trade #'].map(risk_map)
        n_missing_risk = df_exit['Risk USD'].isnull().sum()
        if n_missing_risk > 0:
            st.warning(f"⚠️ พบ Exit {n_missing_risk} รายการ ที่หา Risk USD ไม่เจอ (Trade # ไม่ match หรือ entry ขาด)")
    elif not df_exit.empty:
         st.warning("⚠️ df_entry ว่างเปล่า ไม่สามารถ map Risk USD ไปยัง df_exit ได้")
         df_exit['Risk USD'] = np.nan
    elif df_exit.empty:
         st.info("ℹ️ df_exit ว่างเปล่า ไม่จำเป็นต้อง map Risk USD")


    # --- Clean and Calculate R-Multiples ---
    calc_fields = [
        ('Profit(R)', 'P&L USD'),
        ('MFE(R)',    'Run-up USD'),
        ('MAE(R)',    'Drawdown USD'),
    ]
    if not df_exit.empty:
        for r_col, src_col in calc_fields:
            if src_col not in df_exit.columns:
                raise KeyError(f"ไม่พบคอลัมน์ {src_col} ใน Exit")
            df_exit[src_col] = df_exit[src_col].map(clean_number)
            if 'Risk USD' not in df_exit.columns:
                 df_exit['Risk USD'] = np.nan
            df_exit[r_col] = safe_divide(df_exit[src_col], df_exit['Risk USD'])

        # --- Outlier Check ---
        for col in ['Profit(R)', 'MFE(R)', 'MAE(R)']:
            if col in df_exit.columns and not df_exit[col].isnull().all():
                outliers = df_exit[col].abs() > 20
                if outliers.any():
                    st.warning(f"⚠️ พบ outlier {col} > 20R ทั้งหมด {outliers.sum()} trade")
            elif col in df_exit.columns:
                 st.info(f"ℹ️ คอลัมน์ {col} ว่างเปล่าหรือไม่สามารถคำนวณได้ สำหรับ Outlier Check")
            else:
                st.info(f"ℹ️ ไม่พบคอลัมน์ {col} สำหรับ Outlier Check")
    else:
        st.info("ℹ️ df_exit ว่างเปล่า ไม่สามารถคำนวณ R-Multiples หรือตรวจสอบ Outlier ได้")
        for r_col, src_col in calc_fields:
             df_exit[r_col] = np.nan
        if 'Risk USD' not in df_exit.columns:
            df_exit['Risk USD'] = np.nan

    # ✅ === Add Detailed Entry/Exit Time Information ===
    df_result = df_exit.copy()
    if not df_exit.empty and not df_entry.empty:
        entry_time_map   = df_entry.set_index('Trade #')['Date/Time']
        if 'Signal' in df_entry.columns:
            entry_signal_map = df_entry.set_index('Trade #')['Signal']
            df_result['Entry Signal'] = df_result['Trade #'].map(entry_signal_map)
        else:
             st.info("ℹ️ ไม่พบคอลัมน์ 'Signal' ใน Entry trades. ไม่สามารถ map 'Entry Signal' ได้.")
             df_result['Entry Signal'] = np.nan

        df_result['Entry Time']   = df_result['Trade #'].map(entry_time_map)
        df_result['Entry Day']    = df_result['Entry Time'].apply(lambda x: x.day_name() if pd.notnull(x) else np.nan)
        df_result['Entry HH:MM']  = df_result['Entry Time'].apply(lambda x: x.strftime('%H:%M') if pd.notnull(x) else np.nan)

        rename_cols = {'Date/Time': 'Exit Time'}
        if 'Signal' in df_exit.columns:
             rename_cols['Signal'] = 'Exit Type'
        else:
             st.info("ℹ️ ไม่พบคอลัมน์ 'Signal' ใน Exit trades. ไม่สามารถตั้งชื่อ Exit Type ได้.")
        df_result.rename(columns=rename_cols, inplace=True)
        if 'Exit Type' not in df_result.columns:
             df_result['Exit Type'] = np.nan
    elif not df_exit.empty:
        st.warning("⚠️ df_entry ว่างเปล่า ไม่สามารถ map Entry Time/Signal ได้. จะมีเฉพาะข้อมูล Exit.")
        df_result['Entry Time'] = np.nan
        df_result['Entry Signal'] = np.nan
        df_result['Entry Day'] = np.nan
        df_result['Entry HH:MM'] = np.nan
        rename_cols = {'Date/Time': 'Exit Time'}
        if 'Signal' in df_exit.columns:
             rename_cols['Signal'] = 'Exit Type'
        df_result.rename(columns=rename_cols, inplace=True)
        if 'Exit Type' not in df_result.columns:
             df_result['Exit Type'] = np.nan
    else:
        st.info("ℹ️ df_exit ว่างเปล่า ไม่สามารถสร้างตารางผลลัพธ์ได้.")
        possible_columns = [
            'Trade #', 'Entry Day', 'Entry HH:MM', 'Entry Time', 'Entry Signal',
            'Exit Time', 'Exit Type', 'P&L USD', 'Run-up USD', 'Drawdown USD',
            'Risk USD', 'Profit(R)', 'MFE(R)', 'MAE(R)'
        ]
        return pd.DataFrame(columns=possible_columns) # Return empty DF here

    desired_columns = [
        'Trade #', 'Entry Day', 'Entry HH:MM', 'Entry Time', 'Entry Signal',
        'Exit Time', 'Exit Type', 'P&L USD', 'Run-up USD', 'Drawdown USD',
        'Risk USD', 'Profit(R)', 'MFE(R)', 'MAE(R)'
    ]
    final_columns = [col for col in desired_columns if col in df_result.columns]
    df_result = df_result[final_columns]

    # --- Quick Summary Print (to Streamlit) ---
    # st.subheader("--- SUMMARY (from calc_r_multiple_and_risk) ---") # Already displayed in app.py
    # st.write(f"Total Exit trades processed: {len(df_exit)}")
    # if 'Risk USD' in df_exit.columns:
    #      st.write(f"Risk USD mapping missing: {n_missing_risk}")
    # else:
    #      st.info("ℹ️ Risk USD column ไม่พบใน df_exit")
    # if 'Profit(R)' in df_exit.columns:
    #     st.write(f"NaN Profit(R): {df_exit['Profit(R)'].isnull().sum()}")
    #     if not df_exit['Profit(R)'].isnull().all():
    #         st.write(f"Profit(R) min/max: {df_exit['Profit(R)'].min():.4f} / {df_exit['Profit(R)'].max():.4f}")
    #     else:
    #          st.info("ℹ️ Profit(R) ว่างเปล่าหรือมีแต่ NaN")
    # else:
    #     st.info(f"ℹ️ ไม่พบคอลัมน์ Profit(R) สำหรับ Quick Summary")

    return df_result

# 📌 Summary Function: R-Multiple Basic Statistics (คงเดิมจากไฟล์ของท่านพี่)
def summarize_r_multiple_stats(df_result):
    df = df_result.copy()
    if 'Exit Time' not in df.columns:
        st.warning("⚠️ ไม่พบคอลัมน์ 'Exit Time' ใน DataFrame สำหรับการสรุปสถิติ")
        return {col: np.nan for col in [
            "Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)",
            "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)",
            "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades",
            "Win %", "BE %", "Win+BE %"]}

    if pd.api.types.is_datetime64_any_dtype(df['Exit Time']):
         pass
    else:
        try:
            df['Exit Time'] = pd.to_datetime(df['Exit Time'])
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถแปลง 'Exit Time' เป็น datetime ได้: {e}")
            return {col: np.nan for col in [
                "Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)",
                "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)",
                "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades",
                "Win %", "BE %", "Win+BE %"]}

    if 'Profit(R)' not in df.columns:
        st.warning("⚠️ ไม่พบคอลัมน์ 'Profit(R)' ใน DataFrame สำหรับการสรุปสถิติ")
        return {col: np.nan for col in [
            "Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)",
            "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)",
            "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades",
            "Win %", "BE %", "Win+BE %"]}

    df_valid = df.dropna(subset=['Profit(R)']).copy()
    n_total = len(df_valid)
    if n_total == 0:
        st.info("ℹ️ ไม่มีเทรดที่มี Profit(R) ที่ถูกต้อง ไม่สามารถคำนวณสถิติได้")
        return {col: 0 if col.startswith("Total") else np.nan for col in [
            "Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)",
            "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)",
            "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades",
            "Win %", "BE %", "Win+BE %"]}

    n_win   = (df_valid['Profit(R)'] > 0).sum()
    n_loss  = (df_valid['Profit(R)'] < 0).sum()
    n_be    = (df_valid['Profit(R)'] == 0).sum()
    win_sum  = df_valid.loc[df_valid['Profit(R)'] > 0, 'Profit(R)'].sum()
    loss_sum = df_valid.loc[df_valid['Profit(R)'] < 0, 'Profit(R)'].sum()
    profit_factor = safe_divide(win_sum, abs(loss_sum))
    net_profit_r = df_valid['Profit(R)'].sum()
    df_valid = df_valid.sort_values(by='Exit Time').reset_index(drop=True)
    equity_curve = df_valid['Profit(R)'].cumsum()
    equity_high  = equity_curve.cummax()
    dd_curve     = equity_curve - equity_high
    max_drawdown = dd_curve.min() if not dd_curve.empty else 0
    np_dd_ratio  = safe_divide(net_profit_r, abs(max_drawdown))

    dd_periods = []
    if not df_valid.empty and 'Exit Time' in df_valid.columns:
        in_drawdown_flag = (df_valid['Profit(R)'].cumsum() - df_valid['Profit(R)'].cumsum().cummax() < -1e-9).astype(int)
        period_start_idx = None
        for i, dd_flag in enumerate(in_drawdown_flag):
            if dd_flag == 1 and period_start_idx is None: period_start_idx = i
            elif dd_flag == 0 and period_start_idx is not None:
                start_date = df_valid.iloc[period_start_idx]['Exit Time'].date()
                end_date = df_valid.iloc[i-1]['Exit Time'].date()
                days = (end_date - start_date).days + 1
                dd_periods.append(days)
                period_start_idx = None
        if period_start_idx is not None:
            start_date = df_valid.iloc[period_start_idx]['Exit Time'].date()
            end_date = df_valid.iloc[len(df_valid)-1]['Exit Time'].date()
            days = (end_date - start_date).days + 1
            dd_periods.append(days)
    max_dd_period_days = max(dd_periods) if dd_periods else 0

    win_pct   = 100 * safe_divide(n_win, n_total)
    be_pct    = 100 * safe_divide(n_be, n_total)
    winbe_pct = 100 * safe_divide((n_win + n_be), n_total)
    stats = {
        "Profit Factor": profit_factor, "Net Profit (R)": net_profit_r,
        "Maximum Equity DD (R)": max_drawdown, "Net Profit to Max Drawdown Ratio": np_dd_ratio,
        "Drawdown Period (Days)": max_dd_period_days, "Total Trades": n_total,
        "Winning Trades": n_win, "Losing Trades": n_loss, "Breakeven Trades": n_be,
        "Win %": win_pct, "BE %": be_pct, "Win+BE %": winbe_pct,
    }
    return stats

# --- Main processing function for Streamlit ---
def process_data(excel_file_path_or_buffer, desired_stop_loss):
    """
    Main function to be called by Streamlit app.
    It calculates R-Multiples, risk, and summary statistics.
    """
    # IPython.display.display can be removed or replaced with st.dataframe/st.write
    # For example, display(trade_results_df.head()) becomes st.dataframe(trade_results_df.head())

    try:
        trade_results_df = calc_r_multiple_and_risk(excel_file_path_or_buffer, desired_stop_loss)

        if trade_results_df is not None and not trade_results_df.empty:
            # st.subheader("\nProcessed Trade Results (from R01):") # Displayed in app.py
            # st.dataframe(trade_results_df.head()) # Displayed in app.py

            summary_stats = summarize_r_multiple_stats(trade_results_df)
            # st.subheader("\nSummary Statistics (R-Multiples - from R01):") # Displayed in app.py
            # for stat, value in summary_stats.items():
            #     st.text(f"{stat}: {value:.4f}" if isinstance(value, (int, float)) else f"{stat}: {value}")
            return trade_results_df, summary_stats # Return both

        elif trade_results_df is None:
             st.error("❌ Error: calc_r_multiple_and_risk did not return a DataFrame.")
             return None, None
        else: # trade_results_df is not None but empty
             st.info("ℹ️ calc_r_multiple_and_risk returned an empty DataFrame. No trades to display or summarize.")
             return trade_results_df, None # Return empty df and no stats

    except (RuntimeError, ValueError, KeyError) as e:
        st.error(f"❌ Error during processing in R01_data_preparation: {e}")
        return None, None # Return None if there's an error
