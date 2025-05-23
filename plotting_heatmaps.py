# plotting_heatmaps.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap # For custom colormap
from utils import CustomDivergingNorm # Import a custom utility

def display_profit_heatmap(df_source_input, time_column_name, day_column_name_to_derive, plot_title_prefix, bin_size_minutes_heatmap, key_suffix_for_plot):
    # Main header and number input for bin size will be in app.py

    if df_source_input is None or df_source_input.empty:
        st.info(f"ℹ️ ไม่มีข้อมูลเทรดสำหรับแสดง Heatmap ({plot_title_prefix}).")
        return

    df_heatmap = df_source_input.copy()

    required_heatmap_cols = [time_column_name, 'Profit(R)']
    if not all(col in df_heatmap.columns for col in required_heatmap_cols):
        missing = [col for col in required_heatmap_cols if col not in df_heatmap.columns]
        st.error(f"❌ ไม่พบคอลัมน์ที่จำเป็น ({', '.join(missing)}) สำหรับ Heatmap ({plot_title_prefix}).")
        return

    try:
        df_heatmap[time_column_name] = pd.to_datetime(df_heatmap[time_column_name], errors='coerce')
        df_heatmap['Profit(R)'] = pd.to_numeric(df_heatmap['Profit(R)'], errors='coerce')
        df_heatmap.dropna(subset=[time_column_name, 'Profit(R)'], inplace=True)

        if df_heatmap.empty:
            st.info(f"ℹ️ ไม่มีข้อมูลเทรดที่สมบูรณ์ (หลังกรอง NaN) สำหรับ Heatmap ({plot_title_prefix}).")
            return

        # Derive Day Name from the specified time_column_name
        df_heatmap[day_column_name_to_derive] = df_heatmap[time_column_name].dt.day_name()
        df_heatmap.dropna(subset=[day_column_name_to_derive], inplace=True)


        def map_time_to_bin_str(time_obj_or_timestamp, resolution_minutes):
            if pd.isnull(time_obj_or_timestamp): return np.nan
            
            # Convert to datetime.time if it's a Timestamp
            if isinstance(time_obj_or_timestamp, pd.Timestamp):
                time_obj = time_obj_or_timestamp.time()
            elif hasattr(time_obj_or_timestamp, 'hour') and hasattr(time_obj_or_timestamp, 'minute'): # Check if it's time-like
                time_obj = time_obj_or_timestamp
            else: # Not a recognized time format
                return np.nan

            total_minutes_since_midnight = time_obj.hour * 60 + time_obj.minute
            binned_minutes_since_midnight = (total_minutes_since_midnight // resolution_minutes) * resolution_minutes
            bin_hour = binned_minutes_since_midnight // 60
            bin_minute = binned_minutes_since_midnight % 60
            return f"{bin_hour:02d}:{bin_minute:02d}"

        df_heatmap['Time Bin'] = df_heatmap[time_column_name].apply(
            lambda t: map_time_to_bin_str(t, bin_size_minutes_heatmap)
        )
        df_heatmap.dropna(subset=['Time Bin'], inplace=True) # Drop rows where Time Bin could not be created

        if df_heatmap.empty:
            st.info(f"ℹ️ ไม่มีข้อมูลหลังจากการสร้าง Time Bin หรือ Day Column สำหรับ Heatmap ({plot_title_prefix}).")
            return

        agg_data_heatmap = df_heatmap.groupby([day_column_name_to_derive, 'Time Bin'], observed=False)['Profit(R)'].agg(['sum', 'count', 'mean']).reset_index()

        def time_string_to_total_minutes(time_str_hm): # HH:MM format
            if pd.isnull(time_str_hm): return -1 # Or some other indicator for NaN
            h_hm, m_hm = map(int, time_str_hm.split(':'))
            return h_hm * 60 + m_hm

        skip_start_minutes = time_string_to_total_minutes('12:00')
        skip_end_minutes = time_string_to_total_minutes('19:30') # Bins starting before 19:30 will be skipped

        # Filter agg_data_heatmap based on 'Time Bin' before pivoting
        agg_data_heatmap['Time Bin Minutes'] = agg_data_heatmap['Time Bin'].apply(time_string_to_total_minutes)
        agg_data_filtered_heatmap = agg_data_heatmap[
            ~((agg_data_heatmap['Time Bin Minutes'] >= skip_start_minutes) &
              (agg_data_heatmap['Time Bin Minutes'] < skip_end_minutes))
        ].copy()
        agg_data_filtered_heatmap.drop(columns=['Time Bin Minutes'], inplace=True)


        if agg_data_filtered_heatmap.empty:
            st.info(f"ℹ️ ไม่มีข้อมูลเทรดในกรอบเวลาที่แสดง (หลังจากการข้ามช่วง 12:00-19:30) สำหรับ Heatmap ({plot_title_prefix}).")
            return

        # Create a full list of possible time bins for the day to ensure complete columns
        all_possible_time_bins_for_day = []
        for h_bin_list in range(24):
            for m_bin_list in range(0, 60, bin_size_minutes_heatmap):
                all_possible_time_bins_for_day.append(f"{h_bin_list:02d}:{m_bin_list:02d}")

        # Filter this full list to get the bins we actually want to display
        display_time_bins_final = [
            b_str for b_str in all_possible_time_bins_for_day
            if not (time_string_to_total_minutes(b_str) >= skip_start_minutes and
                    time_string_to_total_minutes(b_str) < skip_end_minutes)
        ]
        
        # Pivot tables
        heatmap_sum_df = agg_data_filtered_heatmap.pivot_table(index=day_column_name_to_derive, columns='Time Bin', values='sum')
        heatmap_count_df = agg_data_filtered_heatmap.pivot_table(index=day_column_name_to_derive, columns='Time Bin', values='count')
        heatmap_mean_df = agg_data_filtered_heatmap.pivot_table(index=day_column_name_to_derive, columns='Time Bin', values='mean')

        day_order_heatmap = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        # Reindex to ensure correct day order and all displayable time bins are present
        # Only include days that actually have some data in agg_data_filtered_heatmap
        present_days_in_agg = agg_data_filtered_heatmap[day_column_name_to_derive].unique()
        ordered_present_days = [day for day in day_order_heatmap if day in present_days_in_agg]

        if not ordered_present_days:
            st.info(f"ℹ️ ไม่มีข้อมูลสำหรับวันที่ระบุ (อาทิตย์-เสาร์) ใน Heatmap ({plot_title_prefix}) หลังจากกรองเวลาพัก.")
            return

        heatmap_sum_df = heatmap_sum_df.reindex(index=ordered_present_days, columns=display_time_bins_final)
        heatmap_count_df = heatmap_count_df.reindex(index=ordered_present_days, columns=display_time_bins_final) # Fill with NaN by default
        heatmap_mean_df = heatmap_mean_df.reindex(index=ordered_present_days, columns=display_time_bins_final)


        if heatmap_sum_df.empty or heatmap_sum_df.isnull().all().all(): # Check if all values are NaN after reindex
            st.info(f"ℹ️ ไม่มีข้อมูลสำหรับสร้าง Heatmap ของ {plot_title_prefix} หลังจากการกรองและจัดเรียง.")
            return

        # Create annotation matrix
        annotation_matrix_hm = np.full(heatmap_sum_df.shape, "", dtype=object) # Initialize with empty strings
        for r_idx, day_label in enumerate(heatmap_sum_df.index):
            for c_idx, time_label in enumerate(heatmap_sum_df.columns):
                sum_val_hm = heatmap_sum_df.iloc[r_idx, c_idx]
                # Use .get(time_label) for count and mean as they might not have all columns after pivot if a day/time combo was missing
                count_val_hm = heatmap_count_df.loc[day_label].get(time_label) if day_label in heatmap_count_df.index else np.nan
                mean_val_hm = heatmap_mean_df.loc[day_label].get(time_label) if day_label in heatmap_mean_df.index else np.nan

                if pd.notna(sum_val_hm): # Only annotate if there's a sum value
                    count_str_hm = f"({int(count_val_hm)})" if pd.notna(count_val_hm) and count_val_hm > 0 else "(0)"
                    mean_str_hm = f"{mean_val_hm:.2f}" if pd.notna(mean_val_hm) else "N/A"
                    annotation_matrix_hm[r_idx, c_idx] = f"{sum_val_hm:.2f}\n{count_str_hm}\n{mean_str_hm}"
                else: # Cell had no sum data (NaN)
                    annotation_matrix_hm[r_idx, c_idx] = "" # Keep it empty or put "N/A" for sum

        colors_list_hm = [(0.9, 0.2, 0.1, 0.8), (0.98, 0.98, 0.98, 0.3), (0.1, 0.5, 0.9, 0.8)] # Red-LightTransparentWhite-Blue
        cmap_custom_hm = LinearSegmentedColormap.from_list("custom_heat", colors_list_hm, N=256)

        min_val_hm, max_val_hm = np.nanmin(heatmap_sum_df.values), np.nanmax(heatmap_sum_df.values)
        norm_final_hm = None
        if pd.notnull(min_val_hm) and pd.notnull(max_val_hm) and not np.isclose(min_val_hm, max_val_hm):
             norm_final_hm = CustomDivergingNorm(vmin=min_val_hm, vcenter=0, vmax=max_val_hm)
        elif pd.notnull(min_val_hm) and pd.notnull(max_val_hm) and np.isclose(min_val_hm, max_val_hm) and not np.isclose(min_val_hm,0): # all same non-zero value
             norm_final_hm = CustomDivergingNorm(vmin=min_val_hm - abs(min_val_hm*0.1) - 0.1, vcenter=0, vmax=max_val_hm + abs(max_val_hm*0.1) + 0.1) # create small range around it
        # If all values are zero or all NaN, norm_final_hm remains None, and cmap will be coolwarm or similar default

        st.subheader(f"Heatmap: {plot_title_prefix} ({bin_size_minutes_heatmap}-min bins, 12:00-19:30 Skipped)")
        fig_hm, ax_hm = plt.subplots(figsize=(max(15, len(heatmap_sum_df.columns) * 0.7), max(6, len(heatmap_sum_df.index) * 0.9)))

        sns.heatmap(heatmap_sum_df, cmap=cmap_custom_hm if norm_final_hm else "RdBu_r", norm=norm_final_hm,
                    annot=annotation_matrix_hm, fmt="", linewidths=.5, linecolor='lightgray',
                    cbar=True if norm_final_hm else False, ax=ax_hm, annot_kws={"size": 7}, center=0 if not norm_final_hm else None) # Center if no norm

        ax_hm.set_title(f'Sum of Profit(R) by {day_column_name_to_derive} and {plot_title_prefix} ({bin_size_minutes_heatmap}-min Bins)', fontsize=12)
        ax_hm.set_xlabel(f'{plot_title_prefix} ({bin_size_minutes_heatmap}-min Bins)', fontsize=10)
        ax_hm.set_ylabel(day_column_name_to_derive, fontsize=10)
        plt.setp(ax_hm.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax_hm.get_yticklabels(), rotation=0, fontsize=9) # Keep y-axis labels horizontal

        st.pyplot(fig_hm)
        plt.close(fig_hm)

        st.subheader(f"ตารางข้อมูล Heatmap: {plot_title_prefix}")
        agg_data_filtered_heatmap_display = agg_data_filtered_heatmap.copy()
        agg_data_filtered_heatmap_display.rename(columns={'sum':'Sum(R)', 'count':'Trades', 'mean':'Avg(R)', day_column_name_to_derive: 'Day'}, inplace=True)
        # Order columns for display
        display_cols_heatmap_table = ['Day', 'Time Bin', 'Sum(R)', 'Trades', 'Avg(R)']
        # Filter out columns not present if any, though they should be
        display_cols_heatmap_table = [col for col in display_cols_heatmap_table if col in agg_data_filtered_heatmap_display.columns]

        st.dataframe(agg_data_filtered_heatmap_display[display_cols_heatmap_table].style.format({'Sum(R)': "{:.2f}", 'Avg(R)': "{:.2f}"}, na_rep="N/A"))

    except Exception as e_hm:
        st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง Heatmap สำหรับ {plot_title_prefix}: {e_hm}")
        # st.exception(e_hm)