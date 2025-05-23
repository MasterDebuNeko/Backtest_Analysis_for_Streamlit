# plotting_distributions.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # สำหรับ Histograms
from utils import safe_divide # Import ฟังก์ชันที่เราสร้างไว้

def display_profit_distribution_all(df_profit_hist_all_source_input):
    st.header("4. 📊 Profit(R) Distribution - All Trades")
    st.markdown("Histogram นี้แสดงการกระจายตัวของผลกำไร/ขาดทุน (R-Multiple) จากทุกเทรด")

    if df_profit_hist_all_source_input is None or df_profit_hist_all_source_input.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดสำหรับแสดง Profit(R) Distribution (All Trades).")
        return

    df_profit_hist_all_source = df_profit_hist_all_source_input.copy()

    if 'Profit(R)' not in df_profit_hist_all_source.columns:
        st.error("❌ ไม่พบคอลัมน์ 'Profit(R)' ที่จำเป็นสำหรับการสร้าง Profit Histogram (All Trades).")
        return

    # Drop rows where Profit(R) is NaN, as they cannot be plotted or used in calculations
    df_profit_hist_all_valid = df_profit_hist_all_source.dropna(subset=['Profit(R)']).copy()
    # Ensure Profit(R) is float after dropping NaNs
    if not df_profit_hist_all_valid.empty:
        df_profit_hist_all_valid['Profit(R)'] = df_profit_hist_all_valid['Profit(R)'].astype(float)

    if df_profit_hist_all_valid.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดที่มี Profit(R) ที่ถูกต้อง (หลังจากการกรอง NaN) สำหรับการสร้าง Profit Histogram (All Trades).")
        return

    try:
        r_values_all = df_profit_hist_all_valid['Profit(R)']

        expectancy_all = r_values_all.mean() if not r_values_all.empty else np.nan
        win_mask_all = r_values_all > 1e-9 # Use tolerance
        loss_mask_all = r_values_all < -1e-9 # Use tolerance
        n_win_all = win_mask_all.sum()
        n_loss_all = loss_mask_all.sum()
        total_trades_all = len(r_values_all)
        win_rate_all = 100 * safe_divide(n_win_all, total_trades_all)

        r_values_win_all = r_values_all[win_mask_all]
        r_values_loss_all = r_values_all[loss_mask_all]
        avg_win_all = r_values_win_all.mean() if not r_values_win_all.empty else np.nan
        avg_loss_all = r_values_loss_all.mean() if not r_values_loss_all.empty else np.nan

        fig_profit_hist, ax_profit_hist = plt.subplots(figsize=(12, 6))

        num_bins_all = min(50, max(10, int(np.sqrt(total_trades_all) * 2))) if total_trades_all > 0 else 10
        # Ensure bins cover the data range appropriately
        min_r_all, max_r_all = r_values_all.min(), r_values_all.max()
        bins_all = np.linspace(min_r_all, max_r_all, num_bins_all + 1) if pd.notnull(min_r_all) and pd.notnull(max_r_all) and min_r_all < max_r_all else num_bins_all

        if not r_values_win_all.empty:
            ax_profit_hist.hist(r_values_win_all, bins=bins_all, color='deepskyblue', alpha=0.7, label=f'Wins (n={n_win_all})', edgecolor='white')
        if not r_values_loss_all.empty:
            ax_profit_hist.hist(r_values_loss_all, bins=bins_all, color='salmon', alpha=0.7, label=f'Losses (n={n_loss_all})', edgecolor='white')

        if pd.notnull(expectancy_all):
            ax_profit_hist.axvline(expectancy_all, color='purple', linestyle='dashed', linewidth=1.5, label=f'Expectancy ({expectancy_all:.2f} R)')

        ax_profit_hist.set_title('Distribution of Trade R-Multiples (All Trades)', fontsize=14)
        ax_profit_hist.set_xlabel('Profit(R)', fontsize=12)
        ax_profit_hist.set_ylabel('Frequency', fontsize=12)
        ax_profit_hist.legend(fontsize='small')
        ax_profit_hist.grid(axis='y', linestyle=':', alpha=0.6)
        st.pyplot(fig_profit_hist)
        plt.close(fig_profit_hist)

        st.subheader("สรุปสถิติ R-Multiple Performance (All Trades):")
        summary_stats_all_dict = {
            "Expectancy (R)": expectancy_all,
            "Win Rate (%)": win_rate_all,
            "Avg Win (R)": avg_win_all,
            "Avg Loss (R)": avg_loss_all,
            "Number of Wins": n_win_all,
            "Number of Losses": n_loss_all,
            "Total Trades": total_trades_all
        }
        summary_df_all = pd.DataFrame([summary_stats_all_dict])
        # Format for display
        for col in ["Expectancy (R)", "Win Rate (%)", "Avg Win (R)", "Avg Loss (R)"]:
            if col in summary_df_all.columns:
                 summary_df_all[col] = summary_df_all[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        st.table(summary_df_all)

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง Profit Histogram (All Trades): {e}")
        # st.exception(e)

def display_profit_distribution_by_day(df_profit_hist_day_base_input):
    st.header("4A. 📅 Profit(R) Distribution by Entry Day")
    st.markdown("Histograms นี้แสดงการกระจายตัวของผลกำไร/ขาดทุน (R-Multiple) แยกตามวันที่เข้าเทรด")

    if df_profit_hist_day_base_input is None or df_profit_hist_day_base_input.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดสำหรับแสดง Profit(R) Distribution by Entry Day.")
        return

    df_profit_hist_day_base = df_profit_hist_day_base_input.copy()

    required_cols = ['Entry Time', 'Profit(R)']
    if not all(col in df_profit_hist_day_base.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_profit_hist_day_base.columns]
        st.error(f"❌ ไม่พบคอลัมน์ที่จำเป็น ({', '.join(missing)}) สำหรับ Profit Histogram by Day.")
        return

    try:
        df_profit_hist_day_base['Entry Time'] = pd.to_datetime(df_profit_hist_day_base['Entry Time'], errors='coerce')
        df_profit_hist_day_base.dropna(subset=['Entry Time', 'Profit(R)'], inplace=True)
        
        if not df_profit_hist_day_base.empty: # Ensure Profit(R) is float
            df_profit_hist_day_base['Profit(R)'] = df_profit_hist_day_base['Profit(R)'].astype(float)


        if df_profit_hist_day_base.empty:
            st.info("ℹ️ ไม่มีข้อมูลเทรดที่สมบูรณ์ (หลังกรอง NaN) สำหรับการสร้าง Profit Histogram by Day.")
            return

        if 'Entry Day' not in df_profit_hist_day_base.columns:
            df_profit_hist_day_base['Entry Day'] = df_profit_hist_day_base['Entry Time'].dt.day_name()
        df_profit_hist_day_base.dropna(subset=['Entry Day'], inplace=True) # Drop if Entry Day became NaN

        day_order_profit_hist = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        unique_days_profit_hist = df_profit_hist_day_base['Entry Day'].unique()
        valid_days_for_plot = [day for day in day_order_profit_hist if day in unique_days_profit_hist]

        if not valid_days_for_plot:
            st.info("ℹ️ ไม่มีวันที่มีการเทรดที่ถูกต้องสำหรับการสร้าง Profit Histogram by Day.")
            return

        num_days_ph = len(valid_days_for_plot)
        ncols_ph = 2
        nrows_ph = (num_days_ph + ncols_ph - 1) // ncols_ph

        fig_ph_by_day, axes_ph_by_day = plt.subplots(nrows=nrows_ph, ncols=ncols_ph, figsize=(15, 5 * nrows_ph), squeeze=False)
        axes_flat_ph = axes_ph_by_day.flatten()
        plot_idx_ph = 0
        daily_summary_stats_list_ph = []

        for day_name_ph in valid_days_for_plot:
            df_day_ph = df_profit_hist_day_base[df_profit_hist_day_base['Entry Day'] == day_name_ph].copy()
            if df_day_ph.empty or df_day_ph['Profit(R)'].isnull().all(): continue

            r_values_day_ph = df_day_ph['Profit(R)'] # Already float

            n_win_day_ph = (r_values_day_ph > 1e-9).sum()
            n_loss_day_ph = (r_values_day_ph < -1e-9).sum()
            total_trades_day_ph = len(r_values_day_ph)
            expectancy_day_ph = r_values_day_ph.mean() if total_trades_day_ph > 0 else np.nan
            win_rate_day_ph = 100 * safe_divide(n_win_day_ph, total_trades_day_ph)

            r_win_day_ph = r_values_day_ph[r_values_day_ph > 1e-9]
            r_loss_day_ph = r_values_day_ph[r_values_day_ph < -1e-9]
            avg_win_day_ph = r_win_day_ph.mean() if not r_win_day_ph.empty else np.nan
            avg_loss_day_ph = r_loss_day_ph.mean() if not r_loss_day_ph.empty else np.nan

            daily_summary_stats_list_ph.append({
                "Entry Day": day_name_ph, "Expectancy (R)": expectancy_day_ph,
                "Win Rate (%)": win_rate_day_ph, "Avg Win (R)": avg_win_day_ph,
                "Avg Loss (R)": avg_loss_day_ph, "Number of Wins": n_win_day_ph,
                "Number of Losses": n_loss_day_ph, "Total Trades": total_trades_day_ph
            })

            ax_ph = axes_flat_ph[plot_idx_ph]
            num_bins_day_ph = min(30, max(5, int(np.sqrt(total_trades_day_ph)))) if total_trades_day_ph > 0 else 5
            min_r_day, max_r_day = r_values_day_ph.min(), r_values_day_ph.max()
            bins_day = np.linspace(min_r_day, max_r_day, num_bins_day_ph + 1) if pd.notnull(min_r_day) and pd.notnull(max_r_day) and min_r_day < max_r_day else num_bins_day_ph


            if not r_win_day_ph.empty:
                ax_ph.hist(r_win_day_ph, bins=bins_day, color='deepskyblue', alpha=0.7, label=f'Wins (n={n_win_day_ph})', edgecolor='white')
            if not r_loss_day_ph.empty:
                ax_ph.hist(r_loss_day_ph, bins=bins_day, color='salmon', alpha=0.7, label=f'Losses (n={n_loss_day_ph})', edgecolor='white')

            if pd.notnull(expectancy_day_ph):
                ax_ph.axvline(expectancy_day_ph, color='purple', linestyle='dashed', linewidth=1.2, label=f'Exp. ({expectancy_day_ph:.2f}R)')

            ax_ph.set_title(f'{day_name_ph} R-Multiple Distribution', fontsize=11)
            ax_ph.set_xlabel('Profit(R)', fontsize=9); ax_ph.set_ylabel('Frequency', fontsize=9)
            ax_ph.tick_params(axis='both', which='major', labelsize=8)
            ax_ph.legend(fontsize='xx-small')
            ax_ph.grid(axis='y', linestyle=':', alpha=0.5)
            plot_idx_ph += 1

        for i in range(plot_idx_ph, len(axes_flat_ph)): # Remove unused subplots
            fig_ph_by_day.delaxes(axes_flat_ph[i])

        fig_ph_by_day.tight_layout(pad=2.0, h_pad=3.0)
        st.pyplot(fig_ph_by_day)
        plt.close(fig_ph_by_day)

        if daily_summary_stats_list_ph:
            st.subheader("สรุปสถิติ R-Multiple Performance รายวัน:")
            daily_stats_df_ph = pd.DataFrame(daily_summary_stats_list_ph)
            daily_stats_df_ph['Entry Day'] = pd.Categorical(daily_stats_df_ph['Entry Day'], categories=day_order_profit_hist, ordered=True)
            daily_stats_df_ph = daily_stats_df_ph.sort_values('Entry Day').set_index("Entry Day")

            format_dict = {
                "Expectancy (R)": "{:.2f}", "Win Rate (%)": "{:.2f}", # Removed % for direct float comparison later if needed
                "Avg Win (R)": "{:.2f}", "Avg Loss (R)": "{:.2f}"
            }
            # Create a styled DataFrame for display
            # Make sure all columns in format_dict exist in the DataFrame
            cols_to_format = {k: v for k, v in format_dict.items() if k in daily_stats_df_ph.columns}

            st.table(daily_stats_df_ph.style.format(cols_to_format, na_rep="N/A"))

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง Profit Histogram by Day: {e}")
        # st.exception(e)