# data_processing.py

import streamlit as st # ใช้สำหรับ st.warning, st.info, st.error
import pandas as pd
import numpy as np
from utils import clean_number, validate_stop_loss, safe_divide # เรียกใช้จาก utils.py ที่เราสร้างไว้

def calc_r_multiple_and_risk(xls_path, stop_loss_pct):
    # st.info(f"เริ่มการคำนวณ R-Multiple และ Risk ด้วย Stop Loss: {stop_loss_pct*100:.2f}% จากไฟล์: {xls_path}")
    stop_loss_pct = validate_stop_loss(stop_loss_pct) # ใช้ฟังก์ชันจาก utils.py

    # --- Load Data
    try:
        df_trades = pd.read_excel(xls_path, sheet_name='List of trades')
        df_props  = pd.read_excel(xls_path, sheet_name='Properties')
    except Exception as e:
        raise RuntimeError(f"โหลดไฟล์ Excel ผิดพลาด: {e}. กรุณาตรวจสอบว่าไฟล์ถูกต้องและมีชีทชื่อ 'List of trades' และ 'Properties'")

    # --- Extract Point Value
    try:
        point_value_row = df_props[df_props.iloc[:, 0].astype(str).str.contains("point value", case=False, na=False)]
        if point_value_row.empty:
            raise ValueError("ไม่พบคำว่า 'point value' ในคอลัมน์แรกของชีท 'Properties'")
        point_value = clean_number(point_value_row.iloc[0, 1]) # ใช้ฟังก์ชันจาก utils.py
        if np.isnan(point_value) or point_value <= 0:
            raise ValueError(f"Point Value ที่พบ ({point_value_row.iloc[0, 1]}) ไม่ถูกต้อง (เป็น NaN หรือน้อยกว่าหรือเท่ากับ 0)")
    except Exception as e:
         raise ValueError(f"ข้อผิดพลาดในการดึง Point Value จากชีท 'Properties': {e}")

    # --- Prepare Entry & Exit DataFrames
    try:
        df_entry_orig = df_trades[df_trades['Type'].astype(str).str.contains("Entry", case=False, na=False)].copy()
        df_exit_orig  = df_trades[df_trades['Type'].astype(str).str.contains("Exit", case=False, na=False)].copy()
        if df_entry_orig.empty:
             st.warning("⚠️ ไม่พบรายการ Entry trades ในไฟล์ Excel.")
        if df_exit_orig.empty:
             st.warning("⚠️ ไม่พบรายการ Exit trades ในไฟล์ Excel. จะไม่สามารถคำนวณผลลัพธ์ส่วนใหญ่ได้")
             expected_cols_final = [
                'Trade #', 'Entry Day', 'Entry HH:MM', 'Entry Time', 'Entry Signal',
                'Exit Time', 'Exit Type',
                'P&L USD', 'Run-up USD', 'Drawdown USD',
                'Risk USD', 'Profit(R)', 'MFE(R)', 'MAE(R)'
             ]
             empty_df = pd.DataFrame(columns=expected_cols_final)
             for col in ['Entry Time', 'Exit Time']:
                 if col in empty_df.columns: empty_df[col] = pd.to_datetime(empty_df[col])
             return empty_df
    except KeyError:
        raise KeyError("ไม่พบคอลัมน์ 'Type' ในชีท 'List of trades'.")
    except Exception as e:
        raise RuntimeError(f"ข้อผิดพลาดในการกรอง Entry/Exit trades จากคอลัมน์ 'Type': {e}")

    try:
        df_entry = df_entry_orig.copy()
        df_exit = df_exit_orig.copy()
        if not df_entry.empty: df_entry['Date/Time'] = pd.to_datetime(df_entry['Date/Time'], errors='coerce')
        if not df_exit.empty: df_exit['Date/Time'] = pd.to_datetime(df_exit['Date/Time'], errors='coerce')
    except KeyError: raise KeyError("ไม่พบคอลัมน์ 'Date/Time' ในชีท 'List of trades'.")

    for col in ['Price USD', 'Quantity']:
        if not df_entry.empty:
            if col not in df_entry.columns: raise KeyError(f"ไม่พบคอลัมน์ '{col}' ในข้อมูล Entry trades.")
            df_entry[col] = df_entry[col].map(clean_number) # ใช้ฟังก์ชันจาก utils.py
        if not df_exit.empty and col in df_exit.columns: # Check if col exists before mapping
            df_exit[col] = df_exit[col].map(clean_number) # ใช้ฟังก์ชันจาก utils.py

    if not df_entry.empty:
        df_entry['Risk USD'] = (df_entry['Price USD'] * stop_loss_pct * df_entry['Quantity'] * point_value)
        if df_entry['Risk USD'].isnull().any():
            st.warning("⚠️ มีบางรายการ Entry trades ที่ไม่สามารถคำนวณ 'Risk USD' ได้.")
    else: df_entry['Risk USD'] = np.nan # Or pd.Series(dtype=float) if df_entry is empty but used later

    if 'Trade #' not in df_trades.columns: raise KeyError("ไม่พบคอลัมน์ 'Trade #' ในชีท 'List of trades'.")
    if not df_entry.empty and df_entry['Trade #'].duplicated().any(): st.warning("⚠️ พบหมายเลข Trade # ซ้ำซ้อนในข้อมูล Entry trades.")
    if not df_exit.empty and df_exit['Trade #'].duplicated().any(): st.warning("⚠️ พบหมายเลข Trade # ซ้ำซ้อนในข้อมูล Exit trades.")

    n_missing_risk = 0
    if not df_exit.empty:
        if not df_entry.empty:
            df_entry_for_map = df_entry.dropna(subset=['Trade #']).drop_duplicates(subset=['Trade #'], keep='first')
            risk_map = df_entry_for_map.set_index('Trade #')['Risk USD']
            df_exit['Risk USD'] = df_exit['Trade #'].map(risk_map)
            n_missing_risk = df_exit['Risk USD'].isnull().sum()
            if n_missing_risk > 0: st.warning(f"⚠️ พบ Exit trades จำนวน {n_missing_risk} รายการ ที่ไม่สามารถหา 'Risk USD' ที่สอดคล้องกันได้.")
        else:
            st.warning("⚠️ ไม่มีข้อมูล Entry trades จึงไม่สามารถ map 'Risk USD' ไปยัง Exit trades ได้.")
            df_exit['Risk USD'] = np.nan
    elif 'Risk USD' not in df_exit.columns and not df_exit.empty : df_exit['Risk USD'] = pd.Series(dtype=float)
    elif 'Risk USD' not in df_exit.columns and df_exit.empty: pass # No need to create column if df_exit is empty

    calc_fields = [('Profit(R)', 'P&L USD'), ('MFE(R)', 'Run-up USD'), ('MAE(R)', 'Drawdown USD')]
    if not df_exit.empty:
        for r_col, src_col in calc_fields:
            if src_col not in df_exit.columns: raise KeyError(f"ไม่พบคอลัมน์ '{src_col}' ในข้อมูล Exit trades ซึ่งจำเป็นสำหรับคำนวณ '{r_col}'.")
            df_exit[src_col] = df_exit[src_col].map(clean_number) # ใช้ฟังก์ชันจาก utils.py
            if 'Risk USD' not in df_exit.columns: df_exit['Risk USD'] = np.nan
            df_exit[r_col] = safe_divide(df_exit[src_col], df_exit['Risk USD']) # ใช้ฟังก์ชันจาก utils.py
            if df_exit[r_col].isnull().sum() > n_missing_risk and n_missing_risk < len(df_exit): # Check if more NaNs than just from missing risk
                st.warning(f"⚠️ มีค่า NaN เพิ่มเติมในคอลัมน์ '{r_col}' มากกว่าที่คาดไว้ (อาจเกิดจากค่า Risk USD เป็น 0 หรือ NaN สำหรับ trade ที่มี P&L).")
        for col in ['Profit(R)', 'MFE(R)', 'MAE(R)']:
            if col in df_exit.columns and not df_exit[col].isnull().all():
                if (df_exit[col].abs() > 20).any(): st.warning(f"⚠️ พบค่า outlier ในคอลัมน์ '{col}' (ค่าสัมบูรณ์ > 20R) จำนวน {(df_exit[col].abs() > 20).sum()} trade.")
            elif col in df_exit.columns: st.info(f"ℹ️ คอลัมน์ '{col}' สำหรับ Outlier Check ว่างเปล่าหรือมีแต่ NaN.")
    else: # df_exit is empty
        for r_col, _ in calc_fields: df_exit[r_col] = pd.Series(dtype=float)


    df_result = df_exit.copy()
    if not df_result.empty:
        if not df_entry.empty:
            df_entry_for_map = df_entry.dropna(subset=['Trade #', 'Date/Time']).drop_duplicates(subset=['Trade #'], keep='first')
            entry_time_map = df_entry_for_map.set_index('Trade #')['Date/Time']
            df_result['Entry Time'] = df_result['Trade #'].map(entry_time_map)

            if 'Signal' in df_entry.columns:
                if 'Signal' in df_entry_for_map.columns :
                    entry_signal_map = df_entry_for_map.set_index('Trade #')['Signal']
                    df_result['Entry Signal'] = df_result['Trade #'].map(entry_signal_map)
                else:
                    df_result['Entry Signal'] = np.nan
            else:
                st.info("ℹ️ ไม่พบคอลัมน์ 'Signal' ในข้อมูล Entry trades. 'Entry Signal' จะเป็นค่าว่าง.")
                df_result['Entry Signal'] = np.nan
        else:
            st.warning("⚠️ ไม่มีข้อมูล Entry trades. 'Entry Time' และ 'Entry Signal' จะเป็นค่าว่าง.")
            df_result['Entry Time'] = pd.NaT
            df_result['Entry Signal'] = np.nan

        df_result['Entry Time'] = pd.to_datetime(df_result['Entry Time'], errors='coerce')
        df_result['Entry Day'] = df_result['Entry Time'].dt.day_name()
        df_result['Entry HH:MM'] = df_result['Entry Time'].dt.strftime('%H:%M')
        df_result.loc[df_result['Entry Time'].isnull(), ['Entry Day', 'Entry HH:MM']] = np.nan

        rename_cols_exit = {'Date/Time': 'Exit Time'}
        if 'Signal' in df_result.columns: # Check if 'Signal' exists from df_exit copy
            rename_cols_exit['Signal'] = 'Exit Type'
        else: # 'Signal' was not in df_exit to begin with
            st.info("ℹ️ ไม่พบคอลัมน์ 'Signal' ในข้อมูล Exit trades. 'Exit Type' จะถูกสร้างเป็นค่าว่าง.")
            df_result['Exit Type'] = np.nan # Create it if it doesn't exist
        df_result.rename(columns=rename_cols_exit, inplace=True)
        if 'Exit Type' not in df_result.columns: df_result['Exit Type'] = np.nan # Ensure it exists after rename attempt

    else: # df_result is empty (because df_exit was empty)
        expected_cols_final = ['Trade #', 'Entry Day', 'Entry HH:MM', 'Entry Time', 'Entry Signal', 'Exit Time', 'Exit Type', 'P&L USD', 'Run-up USD', 'Drawdown USD', 'Risk USD', 'Profit(R)', 'MFE(R)', 'MAE(R)']
        df_result = pd.DataFrame(columns=expected_cols_final)
        for col in ['Entry Time', 'Exit Time']:
            if col in df_result.columns: df_result[col] = pd.to_datetime(df_result[col])

    desired_columns = ['Trade #', 'Entry Day', 'Entry HH:MM', 'Entry Time', 'Entry Signal', 'Exit Time', 'Exit Type', 'P&L USD', 'Run-up USD', 'Drawdown USD', 'Risk USD', 'Profit(R)', 'MFE(R)', 'MAE(R)']
    for col in desired_columns:
        if col not in df_result.columns:
            if 'Time' in col:
                df_result[col] = pd.NaT
            elif col in ['P&L USD', 'Run-up USD', 'Drawdown USD', 'Risk USD', 'Profit(R)', 'MFE(R)', 'MAE(R)']:
                df_result[col] = np.nan
            else: # For 'Trade #', 'Entry Day', 'Entry HH:MM', 'Entry Signal', 'Exit Type'
                df_result[col] = pd.Series(dtype='object') # Use object for string/mixed types

    df_result = df_result[desired_columns]
    return df_result


def summarize_r_multiple_stats(df_result_input):
    if df_result_input is None or df_result_input.empty:
        st.warning("⚠️ ไม่สามารถคำนวณสถิติได้ เนื่องจากไม่มีข้อมูลเทรดที่ประมวลผลแล้ว (DataFrame ว่างเปล่า)")
        return {"Profit Factor": np.nan, "Net Profit (R)": 0, "Maximum Equity DD (R)": 0, "Net Profit to Max Drawdown Ratio": np.nan, "Drawdown Period (Days)": 0, "Total Trades": 0, "Winning Trades": 0, "Losing Trades": 0, "Breakeven Trades": 0, "Win %": np.nan, "BE %": np.nan, "Win+BE %": np.nan}

    df = df_result_input.copy()
    for col_name in ['Exit Time', 'Profit(R)']:
        if col_name not in df.columns:
            st.error(f"❌ ไม่พบคอลัมน์ '{col_name}' ใน DataFrame สำหรับการสรุปสถิติ.")
            # Return a dictionary of NaNs to avoid breaking the calling code expecting a dict
            return {stat: np.nan for stat in ["Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)", "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)", "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades", "Win %", "BE %", "Win+BE %"]}

    try:
        df['Exit Time'] = pd.to_datetime(df['Exit Time'], errors='coerce')
    except Exception as e:
        st.error(f"❌ ไม่สามารถแปลง 'Exit Time' เป็น datetime ได้: {e}")
        return {stat: np.nan for stat in ["Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)", "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)", "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades", "Win %", "BE %", "Win+BE %"]}

    df_valid = df.dropna(subset=['Profit(R)', 'Exit Time']).copy()
    n_total = len(df_valid)

    if n_total == 0:
        st.info("ℹ️ ไม่มีเทรดที่มี Profit(R) และ Exit Time ที่ถูกต้องหลังจากกรอง NaN จึงไม่สามารถคำนวณสถิติ R-Multiple ได้")
        return {"Profit Factor": np.nan, "Net Profit (R)": 0, "Maximum Equity DD (R)": 0, "Net Profit to Max Drawdown Ratio": np.nan, "Drawdown Period (Days)": 0, "Total Trades": 0, "Winning Trades": 0, "Losing Trades": 0, "Breakeven Trades": 0, "Win %": 0, "BE %": 0, "Win+BE %": 0} # Use 0 for counts and percentages when no valid trades

    n_win = (df_valid['Profit(R)'] > 0).sum()
    n_loss = (df_valid['Profit(R)'] < 0).sum()
    n_be = (np.isclose(df_valid['Profit(R)'], 0)).sum() # isclose for float comparison

    win_sum = df_valid.loc[df_valid['Profit(R)'] > 0, 'Profit(R)'].sum()
    loss_sum = df_valid.loc[df_valid['Profit(R)'] < 0, 'Profit(R)'].sum() # This is a negative sum

    profit_factor = safe_divide(win_sum, abs(loss_sum)) # ใช้ฟังก์ชันจาก utils.py
    net_profit_r = df_valid['Profit(R)'].sum()

    df_valid = df_valid.sort_values(by='Exit Time').reset_index(drop=True)
    equity_curve = df_valid['Profit(R)'].cumsum()
    equity_high = equity_curve.cummax()
    dd_curve = equity_curve - equity_high
    max_drawdown = dd_curve.min() if not dd_curve.empty else 0 # Will be 0 or negative

    np_dd_ratio = safe_divide(net_profit_r, abs(max_drawdown)) # Use abs for max_drawdown

    dd_periods_days = []
    current_dd_start_date = None
    if not df_valid.empty:
        in_dd_flag = (dd_curve < -1e-9) # Use a small tolerance for float comparison
        for idx in df_valid.index:
            if in_dd_flag[idx] and current_dd_start_date is None:
                current_dd_start_date = df_valid.loc[idx, 'Exit Time']
            elif not in_dd_flag[idx] and current_dd_start_date is not None:
                # DD ended on the previous trade's exit time
                dd_end_date = df_valid.loc[idx-1, 'Exit Time'] if idx > 0 else current_dd_start_date
                if pd.notnull(dd_end_date) and pd.notnull(current_dd_start_date):
                    days_in_dd = (dd_end_date - current_dd_start_date).days + 1
                    dd_periods_days.append(days_in_dd)
                current_dd_start_date = None
        # Check for ongoing DD at the end
        if current_dd_start_date is not None:
            dd_end_date = df_valid.loc[df_valid.index[-1], 'Exit Time']
            if pd.notnull(dd_end_date) and pd.notnull(current_dd_start_date):
                days_in_dd = (dd_end_date - current_dd_start_date).days + 1
                dd_periods_days.append(days_in_dd)

    max_dd_period_days = max(dd_periods_days) if dd_periods_days else 0

    win_pct = 100 * safe_divide(n_win, n_total)
    be_pct = 100 * safe_divide(n_be, n_total)
    winbe_pct = 100 * safe_divide((n_win + n_be), n_total)

    return {
        "Profit Factor": profit_factor,
        "Net Profit (R)": net_profit_r,
        "Maximum Equity DD (R)": max_drawdown,
        "Net Profit to Max Drawdown Ratio": np_dd_ratio,
        "Drawdown Period (Days)": max_dd_period_days,
        "Total Trades": n_total,
        "Winning Trades": n_win,
        "Losing Trades": n_loss,
        "Breakeven Trades": n_be,
        "Win %": win_pct,
        "BE %": be_pct,
        "Win+BE %": winbe_pct
    }