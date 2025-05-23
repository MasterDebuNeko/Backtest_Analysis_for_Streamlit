# plotting_equity.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns # ถ้ามีการใช้สีจาก seaborn โดยตรง หรือ style

def display_overall_equity_curve(df_equity_all_input):
    st.header("2. 📈 Overall Equity Curve Analysis")
    st.markdown("กราฟนี้แสดงผลกำไร/ขาดทุนสะสม (Cumulative R-Multiple) ของพอร์ตโดยรวมเมื่อเวลาผ่านไป พร้อมไฮไลท์ช่วงที่เกิด Drawdown ที่ยาวนานที่สุด 3 อันดับแรก")

    if df_equity_all_input is None or df_equity_all_input.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดสำหรับแสดง Overall Equity Curve.")
        return

    df_equity_all = df_equity_all_input.copy()

    if 'Entry Time' not in df_equity_all.columns or 'Profit(R)' not in df_equity_all.columns:
        st.error("❌ ไม่พบคอลัมน์ 'Entry Time' หรือ 'Profit(R)' ที่จำเป็นสำหรับ Overall Equity Curve.")
        return

    try:
        df_equity_all['Entry Time'] = pd.to_datetime(df_equity_all['Entry Time'], errors='coerce')
    except Exception:
        df_equity_all['Entry Time'] = pd.NaT # Should not happen with errors='coerce' but as a safeguard
        st.warning("⚠️ เกิดปัญหาในการแปลง 'Entry Time' เป็น datetime สำหรับ Overall Equity Curve.")

    df_equity_all.dropna(subset=['Entry Time', 'Profit(R)'], inplace=True)

    if df_equity_all.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดที่ถูกต้อง (หลังจากการกรอง NaN) สำหรับการสร้าง Overall Equity Curve.")
        return

    try:
        df_equity_all = df_equity_all.sort_values('Entry Time').reset_index(drop=True)
        df_equity_all['Entry Date'] = df_equity_all['Entry Time'].dt.normalize()
        df_equity_all['Profit(R)'] = df_equity_all['Profit(R)'].astype(float) # Ensure float
        df_equity_all['Cumulative R'] = df_equity_all['Profit(R)'].cumsum()

        equity_curve = df_equity_all['Cumulative R']
        high_water_mark = equity_curve.cummax()
        drawdown_values = equity_curve - high_water_mark

        drawdown_periods_info = []
        period_start_idx = None
        for i in df_equity_all.index:
            if drawdown_values.loc[i] < -1e-9 and period_start_idx is None: # Use tolerance
                period_start_idx = i
            elif drawdown_values.loc[i] >= -1e-9 and period_start_idx is not None: # Use tolerance
                period_end_idx = i - 1 if i > 0 else 0 # DD ended at previous trade
                if period_start_idx <= period_end_idx:
                    start_date = df_equity_all.loc[period_start_idx, 'Entry Date']
                    end_date = df_equity_all.loc[period_end_idx, 'Entry Date']
                    if pd.notnull(start_date) and pd.notnull(end_date):
                        duration = (end_date - start_date).days + 1
                        period_dd_slice = drawdown_values.loc[period_start_idx : period_end_idx]
                        if not period_dd_slice.empty:
                            valley_r_value = period_dd_slice.min()
                            valley_idx_in_df = period_dd_slice.idxmin()
                            drawdown_periods_info.append({
                                'start_idx': period_start_idx, 'end_idx': period_end_idx,
                                'start_date': start_date, 'end_date': end_date,
                                'duration': duration, 'valley_r': valley_r_value,
                                'valley_idx_in_df': valley_idx_in_df
                            })
                period_start_idx = None

        # Handle ongoing drawdown at the end
        if period_start_idx is not None:
            period_end_idx = df_equity_all.index[-1]
            if period_start_idx <= period_end_idx:
                start_date = df_equity_all.loc[period_start_idx, 'Entry Date']
                end_date = df_equity_all.loc[period_end_idx, 'Entry Date']
                if pd.notnull(start_date) and pd.notnull(end_date):
                    duration = (end_date - start_date).days + 1
                    period_dd_slice = drawdown_values.loc[period_start_idx : period_end_idx]
                    if not period_dd_slice.empty:
                        valley_r_value = period_dd_slice.min()
                        valley_idx_in_df = period_dd_slice.idxmin()
                        drawdown_periods_info.append({
                            'start_idx': period_start_idx, 'end_idx': period_end_idx,
                            'start_date': start_date, 'end_date': end_date,
                            'duration': duration, 'valley_r': valley_r_value,
                            'valley_idx_in_df': valley_idx_in_df
                        })

        drawdown_periods_info = sorted(drawdown_periods_info, key=lambda x: x['duration'], reverse=True)
        top_3_longest_dd = drawdown_periods_info[:min(3, len(drawdown_periods_info))]

        fig_eq_all, ax_eq_all = plt.subplots(figsize=(14, 7))
        ax_eq_all.plot(df_equity_all['Entry Date'], df_equity_all['Cumulative R'], label='Overall Equity Curve', color='dodgerblue', linewidth=2)

        dd_colors = ['salmon', 'lightgreen', 'lightskyblue']
        for i, dd_info in enumerate(top_3_longest_dd):
            if pd.notnull(dd_info['start_date']) and pd.notnull(dd_info['end_date']) and 'valley_idx_in_df' in dd_info and dd_info['valley_idx_in_df'] in df_equity_all.index:
                ax_eq_all.axvspan(dd_info['start_date'], dd_info['end_date'], color=dd_colors[i % len(dd_colors)], alpha=0.3, label=f"DD Period {i+1} ({dd_info['duration']} days)")
                valley_date = df_equity_all.loc[dd_info['valley_idx_in_df'], 'Entry Date']
                valley_equity_r = df_equity_all.loc[dd_info['valley_idx_in_df'], 'Cumulative R'] # Peak of the DD valley on equity curve
                annotation_text = f"{dd_info['duration']}d, {dd_info['valley_r']:.2f}R" # valley_r is the depth of DD
                ax_eq_all.annotate(annotation_text,
                                 xy=(valley_date, valley_equity_r),
                                 xytext=(0, -25 if dd_info['valley_r'] < 0 else 25), # Offset based on DD depth sign (although valley_r should be neg)
                                 textcoords='offset points', ha='center',
                                 va='top' if dd_info['valley_r'] < 0 else 'bottom', # Adjust va based on where the valley_equity_r is
                                 fontsize=9, fontweight='bold', color='black',
                                 bbox=dict(boxstyle='round,pad=0.3', fc='ivory', alpha=0.75, ec='gray'))

        ax_eq_all.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
        ax_eq_all.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_eq_all.get_xticklabels(), rotation=30, ha="right")
        ax_eq_all.set_xlabel('Entry Date'); ax_eq_all.set_ylabel('Cumulative Profit (R-Multiple)'); ax_eq_all.set_title('Overall Equity Curve with Longest Drawdown Periods Highlighted', fontsize=15)
        ax_eq_all.grid(True, linestyle=':', alpha=0.6); ax_eq_all.legend(fontsize='small'); ax_eq_all.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        st.pyplot(fig_eq_all)
        plt.close(fig_eq_all) # Close figure to free memory

        if top_3_longest_dd:
            st.subheader("รายละเอียดช่วง Drawdown ที่ยาวนานที่สุด (Top 3):")
            dd_display_data = [{
                "อันดับ": i+1,
                "วันที่เริ่ม DD": dd['start_date'].strftime('%Y-%m-%d') if pd.notnull(dd['start_date']) else "N/A",
                "วันที่สิ้นสุด DD": dd['end_date'].strftime('%Y-%m-%d') if pd.notnull(dd['end_date']) else "N/A",
                "ระยะเวลา (วัน)": dd['duration'],
                "Drawdown สูงสุด (R)": f"{dd['valley_r']:.2f}"
            } for i, dd in enumerate(top_3_longest_dd)]
            st.table(pd.DataFrame(dd_display_data))
        else:
            st.info("✅ ยอดเยี่ยมมาก! ไม่พบช่วง Drawdown ที่มีนัยสำคัญในข้อมูลนี้เลยเจ้าค่ะ")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการสร้างกราฟ Overall Equity Curve: {e}")
        # st.exception(e) # Uncomment for more detailed error in dev

def display_equity_by_day(df_equity_by_day_base_input):
    st.header("2A. 🗓️ Equity Curve Analysis by Day of the Week")
    st.markdown("กราฟนี้แสดงผลกำไร/ขาดทุนสะสม (Cumulative R-Multiple) แยกตามวันที่เข้าเทรด (Entry Day) เพื่อดูประสิทธิภาพในแต่ละวันของสัปดาห์ พร้อมไฮไลท์ช่วง Drawdown ที่ยาวนานที่สุด 3 อันดับแรกของแต่ละวัน")

    if df_equity_by_day_base_input is None or df_equity_by_day_base_input.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดสำหรับแสดง Equity Curve by Day.")
        return

    df_equity_by_day_base = df_equity_by_day_base_input.copy()
    df_equity_by_day_source = pd.DataFrame()

    if 'Entry Time' not in df_equity_by_day_base.columns or 'Profit(R)' not in df_equity_by_day_base.columns:
        st.error("❌ ไม่พบคอลัมน์ 'Entry Time' หรือ 'Profit(R)' ที่จำเป็นสำหรับ Equity Curve by Day.")
        return

    try:
        df_equity_by_day_base['Entry Time'] = pd.to_datetime(df_equity_by_day_base['Entry Time'], errors='coerce')
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการแปลง 'Entry Time' เป็น datetime สำหรับ Equity Curve by Day: {e}")
        df_equity_by_day_base['Entry Time'] = pd.NaT

    original_rows = len(df_equity_by_day_base)
    df_equity_by_day_source = df_equity_by_day_base.dropna(subset=['Entry Time', 'Profit(R)']).copy()
    dropped_rows = original_rows - len(df_equity_by_day_source)
    if dropped_rows > 0:
        st.warning(f"⚠️ ได้ลบ {dropped_rows} แถว เนื่องจาก 'Entry Time' หรือ 'Profit(R)' เป็นค่าว่าง/ไม่ถูกต้อง ก่อนการวิเคราะห์รายวัน.")

    if df_equity_by_day_source.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดที่สมบูรณ์ (หลังจากการกรองแถวที่มี 'Entry Time' หรือ 'Profit(R)' เป็นค่าว่าง) สำหรับการสร้าง Equity Curve by Day.")
        return

    try:
        if 'Entry Day' not in df_equity_by_day_source.columns: # Ensure 'Entry Day' exists
             df_equity_by_day_source['Entry Day'] = df_equity_by_day_source['Entry Time'].dt.day_name()
        df_equity_by_day_source['Entry Date'] = df_equity_by_day_source['Entry Time'].dt.normalize()
        df_equity_by_day_source.sort_values('Entry Time', inplace=True)

        day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        unique_entry_days_in_data = df_equity_by_day_source['Entry Day'].dropna().unique()
        valid_days_for_plot = [day for day in day_order if day in unique_entry_days_in_data]

        if not valid_days_for_plot:
            st.info("ℹ️ ไม่มีวันที่มีการเทรดที่ถูกต้อง (หลังจากกรอง NaN และตรวจสอบวัน) สำหรับการสร้าง Equity Curve by Day.")
            return

        num_days_to_plot = len(valid_days_for_plot)
        ncols_day = 2
        nrows_day = (num_days_to_plot + ncols_day - 1) // ncols_day

        fig_eq_by_day, axes_eq_by_day = plt.subplots(nrows=nrows_day, ncols=ncols_day, figsize=(15, 6 * nrows_day), squeeze=False)
        axes_flat_day = axes_eq_by_day.flatten()
        plot_idx = 0
        trade_counts_by_day_list = []
        dd_colors_daily = ['#FFB6C1', '#ADD8E6', '#90EE90', '#FFDAB9', '#E6E6FA', '#F0E68C', '#B0E0E6']

        for day_name in valid_days_for_plot:
            df_day = df_equity_by_day_source[df_equity_by_day_source['Entry Day'] == day_name].copy()
            if df_day.empty: continue

            df_day = df_day.sort_values('Entry Time').reset_index(drop=True)
            df_day['Profit(R)'] = df_day['Profit(R)'].astype(float) # Ensure float for cumsum
            df_day['Cumulative R'] = df_day['Profit(R)'].cumsum()
            trade_counts_by_day_list.append({'Entry Day': day_name, '# of Trades': len(df_day)})

            ax = axes_flat_day[plot_idx]
            ax.plot(df_day['Entry Date'], df_day['Cumulative R'], label=f'{day_name} Equity', linewidth=1.5, color=sns.color_palette("husl", num_days_to_plot)[plot_idx % num_days_to_plot]) # Cycle colors if needed

            if not df_day.empty and len(df_day) > 1:
                equity_day_curve = df_day['Cumulative R']
                high_water_day = equity_day_curve.cummax()
                drawdown_day_values = equity_day_curve - high_water_day

                dd_periods_info_day = []
                period_start_idx_day = None
                for k_idx in df_day.index:
                    if drawdown_day_values.loc[k_idx] < -1e-9 and period_start_idx_day is None:
                        period_start_idx_day = k_idx
                    elif drawdown_day_values.loc[k_idx] >= -1e-9 and period_start_idx_day is not None:
                        period_end_idx_day = k_idx - 1 if k_idx > 0 else 0
                        if period_start_idx_day <= period_end_idx_day:
                            start_d, end_d = df_day.loc[period_start_idx_day, 'Entry Date'], df_day.loc[period_end_idx_day, 'Entry Date']
                            if pd.notnull(start_d) and pd.notnull(end_d):
                                dur = (end_d - start_d).days + 1
                                p_dd_slice = drawdown_day_values.loc[period_start_idx_day : period_end_idx_day]
                                if not p_dd_slice.empty:
                                    val_r, val_idx = p_dd_slice.min(), p_dd_slice.idxmin()
                                    dd_periods_info_day.append({'start_date': start_d, 'end_date': end_d, 'duration': dur, 'valley_r': val_r, 'valley_idx_in_df': val_idx})
                        period_start_idx_day = None
                if period_start_idx_day is not None:
                    period_end_idx_day = df_day.index[-1]
                    if period_start_idx_day <= period_end_idx_day:
                        start_d, end_d = df_day.loc[period_start_idx_day, 'Entry Date'], df_day.loc[period_end_idx_day, 'Entry Date']
                        if pd.notnull(start_d) and pd.notnull(end_d):
                            dur = (end_d - start_d).days + 1
                            p_dd_slice = drawdown_day_values.loc[period_start_idx_day : period_end_idx_day]
                            if not p_dd_slice.empty:
                                val_r, val_idx = p_dd_slice.min(), p_dd_slice.idxmin()
                                dd_periods_info_day.append({'start_date': start_d, 'end_date': end_d, 'duration': dur, 'valley_r': val_r, 'valley_idx_in_df': val_idx})

                dd_periods_info_day = sorted(dd_periods_info_day, key=lambda x: x['duration'], reverse=True)
                top_3_dd_day = dd_periods_info_day[:min(3, len(dd_periods_info_day))]

                for dd_idx, dd_info_d in enumerate(top_3_dd_day):
                    if pd.notnull(dd_info_d['start_date']) and pd.notnull(dd_info_d['end_date']) and 'valley_idx_in_df' in dd_info_d and dd_info_d['valley_idx_in_df'] in df_day.index:
                        ax.axvspan(dd_info_d['start_date'], dd_info_d['end_date'], color=dd_colors_daily[dd_idx % len(dd_colors_daily)], alpha=0.25)
                        valley_d = df_day.loc[dd_info_d['valley_idx_in_df'], 'Entry Date']
                        valley_eq_r_d = df_day.loc[dd_info_d['valley_idx_in_df'], 'Cumulative R']
                        ann_text_d = f"{dd_info_d['duration']}d, {dd_info_d['valley_r']:.2f}R"

                        # Determine annotation position based on valley relative to min DD value
                        min_dd_val_for_day = drawdown_day_values.min() if not drawdown_day_values.empty else 0
                        offset_y = -20 if valley_eq_r_d > min_dd_val_for_day else 20 # Adjust logic as needed
                        va_pos = 'top' if valley_eq_r_d > min_dd_val_for_day else 'bottom'

                        ax.annotate(ann_text_d, xy=(valley_d, valley_eq_r_d),
                                    xytext=(0, offset_y),
                                    textcoords='offset points', ha='center',
                                    va=va_pos, fontsize=7, color='dimgray',
                                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='lightgray'))

            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
            plt.setp(ax.get_xticklabels(which="major"), rotation=30, ha="right", fontsize=8)
            ax.set_title(f'{day_name}', fontsize=11)
            ax.set_xlabel('Date', fontsize=9); ax.set_ylabel('Cum. R', fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.4); ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
            ax.legend(fontsize='xx-small', loc='upper left')
            plot_idx += 1

        for i in range(plot_idx, len(axes_flat_day)): # Remove unused subplots
            fig_eq_by_day.delaxes(axes_flat_day[i])

        fig_eq_by_day.tight_layout(pad=2.0, h_pad=3.0)
        st.pyplot(fig_eq_by_day)
        plt.close(fig_eq_by_day) # Close figure

        if trade_counts_by_day_list:
            st.subheader("สรุปจำนวนเทรดในแต่ละวัน (Entry Day):")
            df_counts_display = pd.DataFrame(trade_counts_by_day_list)
            df_counts_display['Entry Day'] = pd.Categorical(df_counts_display['Entry Day'], categories=day_order, ordered=True)
            df_counts_display.sort_values('Entry Day', inplace=True)
            st.table(df_counts_display.set_index('Entry Day'))

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการสร้างกราฟ Equity Curve by Day: {e}")
        # st.exception(e)