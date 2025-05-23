# plotting_trade_counts.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Potentially for styling or color palettes

def display_trade_count_by_entry_day(df_tc_entry_day_base_input):
    st.header("5. 🗓️ Trade Count by Entry Day")
    st.markdown("กราฟแท่งและตารางนี้แสดงจำนวนเทรด (Win, Loss, Breakeven) และเปอร์เซ็นต์เทียบกับจำนวนเทรดทั้งหมดในแต่ละวัน โดยพิจารณาจากวันเข้าเทรด (Entry Day) เฉพาะวันอาทิตย์ถึงวันศุกร์")

    if df_tc_entry_day_base_input is None or df_tc_entry_day_base_input.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดสำหรับแสดง Trade Count by Entry Day.")
        return

    df_tc_entry_day_base = df_tc_entry_day_base_input.copy()

    required_cols = ['Entry Time', 'Profit(R)']
    if not all(col in df_tc_entry_day_base.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_tc_entry_day_base.columns]
        st.error(f"❌ ไม่พบคอลัมน์ที่จำเป็น ({', '.join(missing)}) สำหรับ Trade Count by Entry Day.")
        return

    try:
        df_tc_entry_day_base['Entry Time'] = pd.to_datetime(df_tc_entry_day_base['Entry Time'], errors='coerce')
        df_tc_entry_day_base['Profit(R)'] = pd.to_numeric(df_tc_entry_day_base['Profit(R)'], errors='coerce')
        df_tc_entry_day_base.dropna(subset=['Entry Time', 'Profit(R)'], inplace=True)


        if df_tc_entry_day_base.empty:
            st.info("ℹ️ ไม่มีข้อมูลเทรดที่สมบูรณ์ (หลังกรอง NaN) สำหรับการวิเคราะห์ Trade Count by Entry Day.")
            return

        if 'Entry Day' not in df_tc_entry_day_base.columns:
            df_tc_entry_day_base['Entry Day'] = df_tc_entry_day_base['Entry Time'].dt.day_name()
        df_tc_entry_day_base.dropna(subset=['Entry Day'], inplace=True)

        # Categorize trades by result (use tolerance for float comparison)
        df_tc_entry_day_base['Result Type'] = 'Breakeven' # Default
        df_tc_entry_day_base.loc[df_tc_entry_day_base['Profit(R)'] > 1e-9, 'Result Type'] = 'Win'
        df_tc_entry_day_base.loc[df_tc_entry_day_base['Profit(R)'] < -1e-9, 'Result Type'] = 'Loss'

        day_order_tc = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] # Sun-Fri
        result_order_tc = ['Win', 'Loss', 'Breakeven']

        df_tc_entry_day_filtered = df_tc_entry_day_base[df_tc_entry_day_base['Entry Day'].isin(day_order_tc)].copy()

        if df_tc_entry_day_filtered.empty:
            st.info(f"ℹ️ ไม่พบข้อมูลเทรดที่เข้าเงื่อนไขวัน (อาทิตย์-ศุกร์) สำหรับ Trade Count by Entry Day.")
            return

        trade_counts_entry_day = df_tc_entry_day_filtered.groupby(['Entry Day', 'Result Type'], observed=False).size().unstack(fill_value=0)

        for day in day_order_tc:
            if day not in trade_counts_entry_day.index:
                trade_counts_entry_day.loc[day] = 0
        for res in result_order_tc:
            if res not in trade_counts_entry_day.columns:
                trade_counts_entry_day[res] = 0

        trade_counts_entry_day = trade_counts_entry_day.reindex(index=day_order_tc, fill_value=0) # Order rows and fill missing days
        trade_counts_entry_day = trade_counts_entry_day.reindex(columns=result_order_tc, fill_value=0) # Order columns
        trade_counts_entry_day['Total'] = trade_counts_entry_day.sum(axis=1)


        fig_tc_day, ax_tc_day = plt.subplots(figsize=(12, 7))
        bar_width = 0.25
        x_tc_day = np.arange(len(day_order_tc)) # Use length of day_order_tc for x positions

        colors_tc = {'Win': 'deepskyblue', 'Loss': 'salmon', 'Breakeven': '#b0b0b0'}

        rects1_tc = ax_tc_day.bar(x_tc_day - bar_width, trade_counts_entry_day['Win'], bar_width, label='Win', color=colors_tc['Win'])
        rects2_tc = ax_tc_day.bar(x_tc_day, trade_counts_entry_day['Loss'], bar_width, label='Loss', color=colors_tc['Loss'])
        rects3_tc = ax_tc_day.bar(x_tc_day + bar_width, trade_counts_entry_day['Breakeven'], bar_width, label='Breakeven', color=colors_tc['Breakeven'])

        def add_labels_trade_count(rects, result_type_name, counts_df, ax_plot, day_order_ref):
            # Ensure counts_df is indexed by day_order_ref for correct iloc access
            counts_df_reindexed = counts_df.reindex(day_order_ref)
            for i, rect in enumerate(rects):
                height = rect.get_height()
                # Get total for the specific day corresponding to the bar
                day_for_bar = day_order_ref[i]
                total_day = counts_df_reindexed.loc[day_for_bar, 'Total'] if day_for_bar in counts_df_reindexed.index else 0

                if height > 0 or total_day > 0:
                    percentage = (height / total_day) * 100 if total_day > 0 else 0
                    ax_plot.annotate(f'{int(height)}\n({percentage:.1f}%)',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8, color=colors_tc[result_type_name])

        add_labels_trade_count(rects1_tc, 'Win', trade_counts_entry_day, ax_tc_day, day_order_tc)
        add_labels_trade_count(rects2_tc, 'Loss', trade_counts_entry_day, ax_tc_day, day_order_tc)
        add_labels_trade_count(rects3_tc, 'Breakeven', trade_counts_entry_day, ax_tc_day, day_order_tc)

        ax_tc_day.set_xlabel('Entry Day of Week')
        ax_tc_day.set_ylabel('Number of Trades')
        ax_tc_day.set_title('Trade Counts by Entry Day and Result Type (Sun-Fri)')
        ax_tc_day.set_xticks(x_tc_day)
        ax_tc_day.set_xticklabels(day_order_tc)
        ax_tc_day.legend(title='Result Type')
        ax_tc_day.grid(axis='y', linestyle='--', alpha=0.7)
        if not trade_counts_entry_day[result_order_tc].empty:
             max_y_val = trade_counts_entry_day[result_order_tc].max().max()
             ax_tc_day.set_ylim(0, max_y_val * 1.25 if max_y_val > 0 else 10) # Adjust for labels
        else:
            ax_tc_day.set_ylim(0, 10)

        st.pyplot(fig_tc_day)
        plt.close(fig_tc_day)

        st.subheader("ตารางสรุป Trade Counts and Percentage by Entry Day (Sun-Fri)")
        summary_data_tc_day = []
        for day_code_tc in day_order_tc: # Iterate through defined day order
            if day_code_tc in trade_counts_entry_day.index:
                day_counts_tc = trade_counts_entry_day.loc[day_code_tc]
                total_trades_day_tc = day_counts_tc['Total']
                row_data_tc = {'Entry Day': day_code_tc}
                for res_type_tc in result_order_tc:
                    count_tc = day_counts_tc[res_type_tc]
                    percentage_tc = (count_tc / total_trades_day_tc) * 100 if total_trades_day_tc > 0 else 0
                    row_data_tc[f'{res_type_tc} Count'] = int(count_tc)
                    row_data_tc[f'{res_type_tc} %'] = f"{percentage_tc:.1f}%"
                row_data_tc['Total Trades'] = int(total_trades_day_tc)
                summary_data_tc_day.append(row_data_tc)

        summary_df_tc_day = pd.DataFrame(summary_data_tc_day)
        if not summary_df_tc_day.empty:
            st.table(summary_df_tc_day.set_index('Entry Day'))
        else:
            st.info("ℹ️ ไม่สามารถสร้างตารางสรุป Trade Counts by Entry Day ได้ เนื่องจากไม่มีข้อมูลที่ตรงกัน")


    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ Trade Count by Entry Day: {e}")
        # st.exception(e)


def display_trade_count_by_time_of_day(df_source_input, time_column_name, plot_title_prefix, bin_size_minutes_input, key_suffix_for_plot):
    # This function will be called from app.py for sections 6 & 7
    # The st.header, st.markdown for the main section (6&7) and sub-sections (6. or 7.)
    # and st.number_input will be in app.py

    if df_source_input is None or df_source_input.empty:
        st.info(f"ℹ️ ไม่มีข้อมูลเทรดสำหรับ {plot_title_prefix}.")
        return

    df_plot_time = df_source_input.copy()

    if time_column_name not in df_plot_time.columns or 'Profit(R)' not in df_plot_time.columns:
        st.error(f"❌ ไม่พบคอลัมน์ '{time_column_name}' หรือ 'Profit(R)' สำหรับ {plot_title_prefix}.")
        return

    try:
        df_plot_time[time_column_name] = pd.to_datetime(df_plot_time[time_column_name], errors='coerce')
        df_plot_time['Profit(R)'] = pd.to_numeric(df_plot_time['Profit(R)'], errors='coerce')
        df_plot_time.dropna(subset=[time_column_name, 'Profit(R)'], inplace=True)

        if df_plot_time.empty:
            st.info(f"ℹ️ ไม่มีข้อมูลเทรดที่สมบูรณ์สำหรับ {plot_title_prefix} หลังจากกรอง NaN.")
            return

        df_plot_time['Time of Day Seconds'] = (df_plot_time[time_column_name].dt.hour * 3600 +
                                           df_plot_time[time_column_name].dt.minute * 60 +
                                           df_plot_time[time_column_name].dt.second)
        df_plot_time['Result Type'] = 'Breakeven'
        df_plot_time.loc[df_plot_time['Profit(R)'] > 1e-9, 'Result Type'] = 'Win'
        df_plot_time.loc[df_plot_time['Profit(R)'] < -1e-9, 'Result Type'] = 'Loss'

        bin_size_seconds = bin_size_minutes_input * 60
        total_seconds_in_day = 24 * 3600
        time_bins = np.arange(0, total_seconds_in_day + bin_size_seconds, bin_size_seconds)
        time_bin_labels = []
        for i in range(len(time_bins) - 1):
            start_t_obj = pd.to_datetime(time_bins[i], unit='s').time()
            # For end_t, ensure it doesn't exceed 23:59:59 for the last bin of the day
            end_s_val = time_bins[i+1] - 1
            if end_s_val >= total_seconds_in_day: # If bin extends to or past midnight
                end_t_obj = pd.Timestamp('23:59:59').time()
            else:
                end_t_obj = pd.to_datetime(end_s_val, unit='s').time()
            time_bin_labels.append(f"{start_t_obj.strftime('%H:%M')}-{end_t_obj.strftime('%H:%M')}")


        df_plot_time['Time Bin'] = pd.cut(df_plot_time['Time of Day Seconds'], bins=time_bins, labels=time_bin_labels, right=False, include_lowest=True)

        trade_counts_time_all_bins = df_plot_time.groupby(['Time Bin', 'Result Type'], observed=False).size().unstack(fill_value=0)
        trade_counts_time_all_bins = trade_counts_time_all_bins.reindex(time_bin_labels, fill_value=0) # Ensure all bins are present
        result_order_time = ['Win', 'Loss', 'Breakeven']
        for res_t in result_order_time:
            if res_t not in trade_counts_time_all_bins.columns: trade_counts_time_all_bins[res_t] = 0
        trade_counts_time_all_bins = trade_counts_time_all_bins[result_order_time]
        trade_counts_time_all_bins['Total'] = trade_counts_time_all_bins.sum(axis=1)

        filtered_bin_labels_display = []
        for label in time_bin_labels: # Iterate through all generated bin labels
            start_hour_str, start_min_str = label.split('-')[0].split(':')
            start_hour, start_minute = int(start_hour_str), int(start_min_str)
            # Skip if the bin STARTS within 12:00 to 19:29
            if (start_hour >= 12 and start_hour < 19) or (start_hour == 19 and start_minute < 30):
                continue
            filtered_bin_labels_display.append(label)

        # Filter the DataFrame to only include the desired bins
        trade_counts_time_filtered = trade_counts_time_all_bins.loc[trade_counts_time_all_bins.index.isin(filtered_bin_labels_display)].copy()


        if trade_counts_time_filtered.empty or trade_counts_time_filtered['Total'].sum() == 0:
            st.info(f"ℹ️ ไม่มีข้อมูลเทรดในกรอบเวลาที่แสดง (หลังจากการข้ามช่วง 12:00-19:30) สำหรับ {plot_title_prefix}")
            return

        st.subheader(f"กราฟ: {plot_title_prefix} ({bin_size_minutes_input}-min bins, 12:00-19:30 Skipped)")
        fig_time, ax_time = plt.subplots(figsize=(max(15, len(trade_counts_time_filtered.index) * 0.6), 8)) # Dynamic width
        x_time_filt = np.arange(len(trade_counts_time_filtered))
        bar_width_time = 0.25
        colors_tc_time = {'Win': 'deepskyblue', 'Loss': 'salmon', 'Breakeven': '#b0b0b0'}

        ax_time.bar(x_time_filt - bar_width_time, trade_counts_time_filtered['Win'], bar_width_time, label='Win', color=colors_tc_time['Win'])
        ax_time.bar(x_time_filt, trade_counts_time_filtered['Loss'], bar_width_time, label='Loss', color=colors_tc_time['Loss'])
        ax_time.bar(x_time_filt + bar_width_time, trade_counts_time_filtered['Breakeven'], bar_width_time, label='Breakeven', color=colors_tc_time['Breakeven'])

        ax_time.set_xlabel(f'{plot_title_prefix} ({bin_size_minutes_input}-min bins)')
        ax_time.set_ylabel('Number of Trades')
        ax_time.set_title(f'Trade Counts by {plot_title_prefix} ({bin_size_minutes_input}-min bins, 12:00-19:30 Skipped)')
        ax_time.set_xticks(x_time_filt)
        ax_time.set_xticklabels(trade_counts_time_filtered.index, rotation=45, ha='right', fontsize=min(10, 200 / len(trade_counts_time_filtered.index) if len(trade_counts_time_filtered.index) > 0 else 10)) # Dynamic font size
        ax_time.legend(title='Result Type')
        ax_time.grid(axis='y', linestyle='--', alpha=0.7)

        if not trade_counts_time_filtered[result_order_time].empty:
            max_y_val_time = trade_counts_time_filtered[result_order_time].max().max()
            ax_time.set_ylim(0, max_y_val_time * 1.15 if max_y_val_time > 0 else 10)
        else:
             ax_time.set_ylim(0,10)
        st.pyplot(fig_time)
        plt.close(fig_time)

        st.subheader(f"ตารางสรุป: {plot_title_prefix} ({bin_size_minutes_input}-min bins, 12:00-19:30 Skipped)")
        summary_data_time_filt_list = []
        for t_bin, row_cts in trade_counts_time_filtered.iterrows():
            total_t_bin = row_cts['Total']
            if total_t_bin == 0 and not (row_cts['Win'] > 0 or row_cts['Loss'] > 0 or row_cts['Breakeven'] > 0) : continue # Skip empty bins in table
            row_d_t = {'Time Bin': t_bin}
            for res_t_t in result_order_time:
                ct_t = row_cts[res_t_t]
                perc_t = (ct_t / total_t_bin) * 100 if total_t_bin > 0 else 0
                row_d_t[f'{res_t_t} Count'] = int(ct_t)
                row_d_t[f'{res_t_t} %'] = f"{perc_t:.1f}%"
            row_d_t['Total Trades'] = int(total_t_bin)
            summary_data_time_filt_list.append(row_d_t)

        summary_df_time_filt = pd.DataFrame(summary_data_time_filt_list)
        if not summary_df_time_filt.empty:
            st.table(summary_df_time_filt.set_index('Time Bin'))
        else:
            st.info(f"ℹ️ ไม่มีข้อมูลในตารางสรุปสำหรับ {plot_title_prefix} (หลังจากการข้ามช่วง 12:00-19:30).")


    except Exception as e_time:
        st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ {plot_title_prefix}: {e_time}")
        # st.exception(e_time)