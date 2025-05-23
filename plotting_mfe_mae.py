# plotting_mfe_mae.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.lines import Line2D # For custom legends in scatter plots

# --- Helper function for Scatter Plots (Internal to this module) ---
def _create_scatter_plot_internal(df_data, x_col, y_col, title):
    required_cols_scatter = [x_col, y_col, 'Profit(R)']
    if not all(col in df_data.columns for col in required_cols_scatter):
        missing_cols_scatter = [col for col in required_cols_scatter if col not in df_data.columns]
        st.error(f"❌ ไม่พบคอลัมน์ที่จำเป็น ({', '.join(missing_cols_scatter)}) สำหรับ Scatter Plot: {title}")
        return None

    # Ensure MFE(R) and MAE(R) are numeric, Profit(R) should already be
    for col_to_check in [x_col, y_col, 'Profit(R)']:
        if col_to_check in df_data.columns:
            df_data[col_to_check] = pd.to_numeric(df_data[col_to_check], errors='coerce')

    df_plot_scatter = df_data.dropna(subset=required_cols_scatter).copy()

    if df_plot_scatter.empty:
        st.info(f"ℹ️ ไม่มีข้อมูลที่สมบูรณ์ (หลังกรอง NaN สำหรับ {x_col}, {y_col}, Profit(R)) สำหรับ Scatter Plot: {title}")
        return None

    try:
        # Define colors based on Profit(R) using a small tolerance
        colors = np.select(
            [df_plot_scatter['Profit(R)'] > 1e-9, df_plot_scatter['Profit(R)'] < -1e-9],
            ['blue', 'red'],
            default='gray'
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df_plot_scatter[x_col], df_plot_scatter[y_col], c=colors, alpha=0.6, s=20, edgecolors='w', linewidth=0.5)
        ax.set_xlabel(f'{x_col}')
        ax.set_ylabel(f'{y_col}')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.7, alpha=0.7) # X-axis
        ax.axvline(0, color='black', linestyle='-', linewidth=0.7, alpha=0.7) # Y-axis

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Winning Trades', markerfacecolor='blue', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Losing Trades', markerfacecolor='red', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Breakeven Trades', markerfacecolor='gray', markersize=10, markeredgecolor='k')
        ]
        ax.legend(handles=legend_elements, loc='best')
        return fig
    except Exception as e_scatter:
        st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง Scatter Plot '{title}': {e_scatter}")
        # st.exception(e_scatter)
        return None

def display_mfe_mae_scatter_plots(df_scatter_base_input):
    st.header("10.  Scatter Plots: MFE, MAE, and Profit(R)")
    st.markdown("กราฟ Scatter เหล่านี้ช่วยให้เห็นความสัมพันธ์ระหว่าง MFE (Maximum Favorable Excursion), MAE (Maximum Adverse Excursion), และ Profit(R) ของแต่ละเทรด โดยแบ่งสีตามผลลัพธ์ของเทรด (Win, Loss, Breakeven)")

    if df_scatter_base_input is None or df_scatter_base_input.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดสำหรับแสดง MFE/MAE Scatter Plots.")
        return

    df_scatter_base = df_scatter_base_input.copy()

    st.subheader("10A. MFE(R) vs MAE(R)")
    fig_10a = _create_scatter_plot_internal(df_scatter_base, 'MFE(R)', 'MAE(R)', 'MFE(R) vs MAE(R) by Trade Outcome')
    if fig_10a:
        st.pyplot(fig_10a)
        plt.close(fig_10a)

    st.subheader("10B. MFE(R) vs Profit(R)")
    fig_10b = _create_scatter_plot_internal(df_scatter_base, 'MFE(R)', 'Profit(R)', 'MFE(R) vs Profit(R) by Trade Outcome')
    if fig_10b:
        st.pyplot(fig_10b)
        plt.close(fig_10b)

    st.subheader("10C. MAE(R) vs Profit(R)")
    fig_10c = _create_scatter_plot_internal(df_scatter_base, 'MAE(R)', 'Profit(R)', 'MAE(R) vs Profit(R) by Trade Outcome')
    if fig_10c:
        st.pyplot(fig_10c)
        plt.close(fig_10c)


# --- Helper function for MFE Histograms by Day (Internal to this module) ---
def _categorize_trade_outcome_internal(profit_r_value):
    if pd.isna(profit_r_value): return 'Unknown'
    if profit_r_value > 1e-9: return 'Winning'
    if profit_r_value < -1e-9: return 'Losing'
    return 'Breakeven'

def _plot_mfe_hist_by_day_internal(df_source, filter_trade_outcome=None, title_suffix="", plot_all_outcomes_segmented=False):
    # filter_trade_outcome can be "Winning", "Losing", "Breakeven", or None (for segmented plot)

    if 'Entry Time' not in df_source.columns or 'MFE(R)' not in df_source.columns or 'Profit(R)' not in df_source.columns:
        st.error(f"❌ ไม่พบคอลัมน์ที่จำเป็น ('Entry Time', 'MFE(R)', 'Profit(R)') สำหรับ MFE Histogram by Day {title_suffix}.")
        return

    df_day_base = df_source.copy()
    df_day_base['Entry Time'] = pd.to_datetime(df_day_base['Entry Time'], errors='coerce')
    df_day_base['MFE(R)'] = pd.to_numeric(df_day_base['MFE(R)'], errors='coerce') # Ensure numeric
    df_day_base['Profit(R)'] = pd.to_numeric(df_day_base['Profit(R)'], errors='coerce') # Ensure numeric
    df_day_base.dropna(subset=['Entry Time', 'MFE(R)', 'Profit(R)'], inplace=True)

    if 'Trade_Outcome' not in df_day_base.columns: # Ensure Trade_Outcome exists
        df_day_base['Trade_Outcome'] = df_day_base['Profit(R)'].apply(_categorize_trade_outcome_internal)

    if 'Entry Day' not in df_day_base.columns:
        df_day_base['Entry Day'] = df_day_base['Entry Time'].dt.day_name()
    df_day_base.dropna(subset=['Entry Day'], inplace=True)

    if df_day_base.empty:
        st.info(f"ℹ️ ไม่มีข้อมูลที่สมบูรณ์สำหรับ MFE Histogram by Day {title_suffix}.")
        return

    df_to_plot_day = df_day_base
    if filter_trade_outcome: # If a specific outcome is requested (e.g., "Losing", "Breakeven")
        df_to_plot_day = df_day_base[df_day_base['Trade_Outcome'] == filter_trade_outcome].copy()

    if df_to_plot_day.empty:
        st.info(f"ℹ️ ไม่มีข้อมูลเทรดที่ตรงตามเงื่อนไข ({title_suffix}) สำหรับ MFE Histogram by Day.")
        return

    try:
        day_order_mfe = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] # Sun-Fri typically
        days_present_mfe = [day for day in day_order_mfe if day in df_to_plot_day['Entry Day'].unique()]

        if not days_present_mfe:
            st.info(f"ℹ️ ไม่พบข้อมูลในวันที่ระบุ (อาทิตย์-ศุกร์) สำหรับ MFE Histogram by Day {title_suffix}.")
            return

        num_days_mfe = len(days_present_mfe)
        ncols_mfe = 2
        nrows_mfe = (num_days_mfe + ncols_mfe - 1) // ncols_mfe
        fig_mfe_day, axes_mfe_day = plt.subplots(nrows=nrows_mfe, ncols=ncols_mfe, figsize=(14, 6 * nrows_mfe), squeeze=False)
        axes_mfe_flat = axes_mfe_day.flatten()
        ax_idx_mfe = 0

        for day_mfe in days_present_mfe:
            df_current_day_mfe = df_to_plot_day[df_to_plot_day['Entry Day'] == day_mfe].copy()
            if df_current_day_mfe.empty or df_current_day_mfe['MFE(R)'].isnull().all(): continue

            ax_m = axes_mfe_flat[ax_idx_mfe]
            if plot_all_outcomes_segmented: # For 11A2: All Trades by Day, Segmented
                outcome_colors_mfe_day = {'Winning': 'blue', 'Losing': 'red', 'Breakeven': 'gray', 'Unknown':'purple'}
                outcome_order_mfe_day = ['Winning', 'Losing', 'Breakeven', 'Unknown']
                # Trade_Outcome column should already exist from above
                
                df_plot_current_day_seg = df_current_day_mfe[df_current_day_mfe['Trade_Outcome'] != 'Unknown']
                if df_plot_current_day_seg.empty and not df_current_day_mfe.empty:
                     df_plot_current_day_seg = df_current_day_mfe # Plot unknown if that's all

                if not df_plot_current_day_seg.empty:
                    sns.histplot(data=df_plot_current_day_seg, x='MFE(R)', hue='Trade_Outcome',
                                 palette={k:v for k,v in outcome_colors_mfe_day.items() if k in df_plot_current_day_seg['Trade_Outcome'].unique()},
                                 hue_order=[o for o in outcome_order_mfe_day if o in df_plot_current_day_seg['Trade_Outcome'].unique()],
                                 kde=False, edgecolor='white', alpha=0.8, bins=30, ax=ax_m, multiple="stack")
                    ax_m.legend(title='Trade Outcome', fontsize='x-small')
            else: # For specific outcome plots (11B2, 11C2)
                plot_color = 'salmon' if filter_trade_outcome == "Losing" else 'gray' if filter_trade_outcome == "Breakeven" else 'skyblue'
                sns.histplot(data=df_current_day_mfe, x='MFE(R)', kde=False, color=plot_color, edgecolor='white', alpha=0.8, bins=20, ax=ax_m)

                if filter_trade_outcome in ["Losing", "Breakeven"]:
                    mfe_values_day_specific = df_current_day_mfe['MFE(R)'].dropna()
                    if not mfe_values_day_specific.empty:
                        median_mfe_day = mfe_values_day_specific.median()
                        percentile_70_mfe_day = mfe_values_day_specific.quantile(0.70)
                        if pd.notnull(median_mfe_day):
                            ax_m.axvline(median_mfe_day, color='purple', linestyle='dashed', linewidth=1, label=f'Median ({median_mfe_day:.2f}R)')
                        if pd.notnull(percentile_70_mfe_day):
                            ax_m.axvline(percentile_70_mfe_day, color='green', linestyle='dashed', linewidth=1, label=f'70th Pctl ({percentile_70_mfe_day:.2f}R)')
                        if pd.notnull(median_mfe_day) or pd.notnull(percentile_70_mfe_day):
                            ax_m.legend(fontsize='x-small')
            
            ax_m.set_xlabel('MFE (R-Multiple)')
            ax_m.set_ylabel('Count')
            ax_m.set_title(f'MFE for {title_suffix} on {day_mfe}')
            ax_m.grid(axis='y', linestyle='--', alpha=0.7)
            if filter_trade_outcome == "Breakeven": ax_m.set_xlim(left=max(0, df_current_day_mfe['MFE(R)'].min() if not df_current_day_mfe['MFE(R)'].empty else 0)) # X-axis starts at 0 or min MFE if >0 for Breakeven MFE
            ax_idx_mfe += 1

        for i in range(ax_idx_mfe, len(axes_mfe_flat)): fig_mfe_day.delaxes(axes_mfe_flat[i])
        fig_mfe_day.suptitle(f'MFE Distribution for {title_suffix} by Entry Day', fontsize=16, y=1.00)
        fig_mfe_day.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
        st.pyplot(fig_mfe_day)
        plt.close(fig_mfe_day)

    except Exception as e_mfe_day:
        st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง MFE Histogram by Day {title_suffix}: {e_mfe_day}")
        # st.exception(e_mfe_day)


def display_mfe_histograms(df_mfe_base_input):
    st.header("11. 🌊 MFE (Maximum Favorable Excursion) Histograms")
    st.markdown("Histograms เหล่านี้แสดงการกระจายตัวของ MFE(R) ซึ่งคือจุดที่ราคาเคลื่อนไปในทิศทางที่เป็นกำไรสูงสุดระหว่างที่เปิดเทรดอยู่ ช่วยให้เห็นศักยภาพของเทรดแต่ละประเภท")

    if df_mfe_base_input is None or df_mfe_base_input.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดสำหรับแสดง MFE Histograms.")
        return

    df_mfe_base = df_mfe_base_input.copy()

    # Ensure required columns are present and numeric
    for col in ['Profit(R)', 'MFE(R)']:
        if col not in df_mfe_base.columns:
            st.error(f"❌ ไม่พบคอลัมน์ '{col}' ที่จำเป็นสำหรับ MFE Histograms.")
            return
        df_mfe_base[col] = pd.to_numeric(df_mfe_base[col], errors='coerce')
    
    df_mfe_base['Trade_Outcome'] = df_mfe_base['Profit(R)'].apply(_categorize_trade_outcome_internal)
    df_mfe_base.dropna(subset=['MFE(R)', 'Profit(R)'], inplace=True) # Drop if MFE or Profit became NaN

    # --- 11A1: MFE Histogram - All Trades (Segmented) ---
    st.subheader("11A1. MFE Distribution - All Trades (Segmented by Outcome)")
    df_plot_11a1 = df_mfe_base.copy() # Already has Trade_Outcome
    if df_plot_11a1.empty:
        st.info("ℹ️ ไม่มีข้อมูล MFE(R) และ Profit(R) ที่ถูกต้องสำหรับ MFE Histogram (All Trades).")
    else:
        try:
            fig_11a1, ax_11a1 = plt.subplots(figsize=(12, 7))
            outcome_colors_11a1 = {'Winning': 'blue', 'Losing': 'red', 'Breakeven': 'gray', 'Unknown': 'purple'}
            outcome_order_11a1 = ['Winning', 'Losing', 'Breakeven', 'Unknown']
            
            df_plot_11a1_filtered = df_plot_11a1[df_plot_11a1['Trade_Outcome'] != 'Unknown']
            if df_plot_11a1_filtered.empty and not df_plot_11a1.empty :
                 st.warning("ℹ️ ข้อมูล MFE ทั้งหมดมีผลลัพธ์ Profit(R) ที่ไม่สามารถระบุได้ (Unknown).")
                 df_plot_11a1_filtered = df_plot_11a1
            elif df_plot_11a1_filtered.empty:
                 st.info("ℹ️ ไม่มีข้อมูลสำหรับ MFE Histogram (All Trades) หลังจากการกรอง 'Unknown' outcomes.")

            if not df_plot_11a1_filtered.empty:
                sns.histplot(data=df_plot_11a1_filtered, x='MFE(R)', hue='Trade_Outcome',
                             palette={k: v for k, v in outcome_colors_11a1.items() if k in df_plot_11a1_filtered['Trade_Outcome'].unique()},
                             hue_order=[o for o in outcome_order_11a1 if o in df_plot_11a1_filtered['Trade_Outcome'].unique()],
                             kde=False, edgecolor='white', alpha=0.8, bins=50, ax=ax_11a1, multiple="stack")
                ax_11a1.set_xlabel('MFE (R-Multiple)')
                ax_11a1.set_ylabel('Count')
                ax_11a1.set_title('Distribution of MFE by Trade Outcome')
                ax_11a1.grid(axis='y', linestyle='--', alpha=0.7)
                ax_11a1.legend(title='Trade Outcome')
                st.pyplot(fig_11a1)
                plt.close(fig_11a1)
        except Exception as e_11a1:
            st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง MFE Histogram (All Trades): {e_11a1}")
            # st.exception(e_11a1)

    # --- 11A2: MFE Histogram - All Trades by Day (Segmented) ---
    st.subheader("11A2. MFE Distribution - All Trades by Entry Day (Segmented by Outcome)")
    _plot_mfe_hist_by_day_internal(df_mfe_base, filter_trade_outcome=None, title_suffix="All Trades (Segmented)", plot_all_outcomes_segmented=True)

    # --- 11B1: MFE Histogram - Losing Trades (Overall) ---
    st.subheader("11B1. MFE Distribution - Losing Trades")
    df_plot_11b1 = df_mfe_base[df_mfe_base['Trade_Outcome'] == 'Losing'].copy() # Already dropped MFE NaNs
    if df_plot_11b1.empty:
        st.info("ℹ️ ไม่มีเทรดที่ขาดทุน หรือไม่มีข้อมูล MFE(R) ที่ถูกต้องสำหรับเทรดที่ขาดทุน.")
    else:
        try:
            fig_11b1, ax_11b1 = plt.subplots(figsize=(12, 7))
            mfe_losses = df_plot_11b1['MFE(R)']
            median_mfe_losses = mfe_losses.median()
            percentile_70_mfe_losses = mfe_losses.quantile(0.70)
            sns.histplot(data=df_plot_11b1, x='MFE(R)', kde=False, color='salmon', edgecolor='white', alpha=0.8, bins=50, ax=ax_11b1)
            if pd.notnull(median_mfe_losses): ax_11b1.axvline(median_mfe_losses, color='purple', linestyle='dashed', linewidth=1.5, label=f'Median ({median_mfe_losses:.2f}R)')
            if pd.notnull(percentile_70_mfe_losses): ax_11b1.axvline(percentile_70_mfe_losses, color='green', linestyle='dashed', linewidth=1.5, label=f'70th Pctl ({percentile_70_mfe_losses:.2f}R)')
            ax_11b1.set_xlabel('MFE (R-Multiple)'); ax_11b1.set_ylabel('Count'); ax_11b1.set_title('Distribution of MFE for Losing Trades')
            ax_11b1.grid(axis='y', linestyle='--', alpha=0.7); ax_11b1.legend()
            st.pyplot(fig_11b1)
            plt.close(fig_11b1)
        except Exception as e_11b1: st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง MFE Histogram (Losing Trades): {e_11b1}"); # st.exception(e_11b1)

    # --- 11B2: MFE Histogram - Losing Trades by Day ---
    st.subheader("11B2. MFE Distribution - Losing Trades by Entry Day")
    _plot_mfe_hist_by_day_internal(df_mfe_base, filter_trade_outcome="Losing", title_suffix="Losing Trades")

    # --- 11C1: MFE Histogram - Breakeven Trades (Overall) ---
    st.subheader("11C1. MFE Distribution - Breakeven Trades")
    df_plot_11c1 = df_mfe_base[df_mfe_base['Trade_Outcome'] == 'Breakeven'].copy()
    if df_plot_11c1.empty:
        st.info("ℹ️ ไม่มีเทรดที่เสมอตัว หรือไม่มีข้อมูล MFE(R) ที่ถูกต้องสำหรับเทรดที่เสมอตัว.")
    else:
        try:
            fig_11c1, ax_11c1 = plt.subplots(figsize=(12, 7))
            mfe_be = df_plot_11c1['MFE(R)']
            median_mfe_be = mfe_be.median()
            percentile_70_mfe_be = mfe_be.quantile(0.70)
            
            min_mfe_be_val = mfe_be.min() if not mfe_be.empty else 0.0
            max_mfe_be_val = mfe_be.max() if not mfe_be.empty else 1.0
            # Ensure bins start at 0 or a sensible positive value for Breakeven MFE
            bin_start_be = max(0.0, min_mfe_be_val if pd.notnull(min_mfe_be_val) else 0.0)
            bins_11c1 = np.linspace(bin_start_be, max_mfe_be_val if pd.notnull(max_mfe_be_val) else 1.0, 20) if not mfe_be.empty else 20

            sns.histplot(data=df_plot_11c1, x='MFE(R)', kde=False, color='gray', edgecolor='white', alpha=0.8, bins=bins_11c1, ax=ax_11c1)
            if pd.notnull(median_mfe_be): ax_11c1.axvline(median_mfe_be, color='purple', linestyle='dashed', linewidth=1.5, label=f'Median ({median_mfe_be:.2f}R)')
            if pd.notnull(percentile_70_mfe_be): ax_11c1.axvline(percentile_70_mfe_be, color='green', linestyle='dashed', linewidth=1.5, label=f'70th Pctl ({percentile_70_mfe_be:.2f}R)')
            ax_11c1.set_xlabel('MFE (R-Multiple)'); ax_11c1.set_ylabel('Count'); ax_11c1.set_title('Distribution of MFE for Breakeven Trades')
            ax_11c1.grid(axis='y', linestyle='--', alpha=0.7); ax_11c1.legend()
            ax_11c1.set_xlim(left=bin_start_be) # Ensure x-axis starts appropriately for BE MFE
            st.pyplot(fig_11c1)
            plt.close(fig_11c1)
        except Exception as e_11c1: st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง MFE Histogram (Breakeven Trades): {e_11c1}"); # st.exception(e_11c1)

    # --- 11C2: MFE Histogram - Breakeven Trades by Day ---
    st.subheader("11C2. MFE Distribution - Breakeven Trades by Entry Day")
    _plot_mfe_hist_by_day_internal(df_mfe_base, filter_trade_outcome="Breakeven", title_suffix="Breakeven Trades")