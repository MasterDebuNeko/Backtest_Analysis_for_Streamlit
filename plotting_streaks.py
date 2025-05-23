# plotting_streaks.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # สำหรับ Scatter plot timeline
import seaborn as sns # สำหรับ Histogram

def display_losing_streak_analysis(df_streak_source_input):
    st.header("3. 📉 Losing Streak Analysis")
    st.markdown("การวิเคราะห์ช่วงที่ขาดทุนติดต่อกัน (Losing Streaks) เพื่อทำความเข้าใจความถี่และความยาวนานของช่วงเวลาที่ผลการเทรดไม่เป็นใจ")

    if df_streak_source_input is None or df_streak_source_input.empty:
        st.info("ℹ️ ไม่มีข้อมูลเทรดสำหรับวิเคราะห์ Losing Streak.")
        return

    df_streak_source = df_streak_source_input.copy()

    if 'Entry Time' not in df_streak_source.columns or 'Profit(R)' not in df_streak_source.columns:
        st.error("❌ ไม่พบคอลัมน์ 'Entry Time' หรือ 'Profit(R)' ที่จำเป็นสำหรับการวิเคราะห์ Losing Streak.")
        return

    try:
        df_streak_source['Entry Time'] = pd.to_datetime(df_streak_source['Entry Time'], errors='coerce')
        # Ensure Profit(R) is numeric for comparison
        df_streak_source['Profit(R)'] = pd.to_numeric(df_streak_source['Profit(R)'], errors='coerce')
        df_streak_source.dropna(subset=['Entry Time', 'Profit(R)'], inplace=True)

        if df_streak_source.empty:
            st.info("ℹ️ ไม่มีข้อมูลเทรดที่สมบูรณ์ (หลังกรอง NaN) สำหรับการวิเคราะห์ Losing Streak.")
            return

        df_streak_source = df_streak_source.sort_values('Entry Time').reset_index(drop=True)

        if 'Entry Day' not in df_streak_source.columns:
             df_streak_source['Entry Day'] = df_streak_source['Entry Time'].dt.day_name()
        df_streak_source.dropna(subset=['Entry Day'], inplace=True) # Ensure Entry Day is valid

        # Determine losses (use a small tolerance for float comparison)
        df_streak_source['Is_Loss'] = df_streak_source['Profit(R)'] < -1e-9

        # Identify streak starts and ends
        df_streak_source['Streak_Start'] = df_streak_source['Is_Loss'] & (~df_streak_source['Is_Loss'].shift(1, fill_value=False))
        # A streak ends when a trade is NOT a loss, AND the PREVIOUS trade WAS a loss.
        df_streak_source['Streak_End_Signal'] = ~df_streak_source['Is_Loss'] & (df_streak_source['Is_Loss'].shift(1, fill_value=False))

        losing_streaks_list = []
        current_streak_start_idx = None

        for index, row in df_streak_source.iterrows():
            if row['Streak_Start']:
                current_streak_start_idx = index
            # If an end signal is found AND we are currently in a streak
            elif row['Streak_End_Signal'] and current_streak_start_idx is not None:
                # The streak ended on the *previous* trade (index - 1)
                streak_end_idx = index - 1
                if streak_end_idx >= current_streak_start_idx : # Ensure valid range
                    streak_df_slice = df_streak_source.loc[current_streak_start_idx : streak_end_idx]
                    # Double check if all trades in this slice are indeed losses
                    if not streak_df_slice.empty and streak_df_slice['Is_Loss'].all():
                        losing_streaks_list.append({
                            'Start Date': streak_df_slice.iloc[0]['Entry Time'].date(),
                            'End Date': streak_df_slice.iloc[-1]['Entry Time'].date(),
                            'Length': len(streak_df_slice),
                            'Entry Day of Week': streak_df_slice.iloc[0]['Entry Day']
                        })
                current_streak_start_idx = None # Reset for the next streak

        # Handle a streak that might be ongoing at the end of the DataFrame
        if current_streak_start_idx is not None:
            streak_df_slice = df_streak_source.loc[current_streak_start_idx : df_streak_source.index[-1]]
            if not streak_df_slice.empty and streak_df_slice['Is_Loss'].all():
                losing_streaks_list.append({
                    'Start Date': streak_df_slice.iloc[0]['Entry Time'].date(),
                    'End Date': streak_df_slice.iloc[-1]['Entry Time'].date(),
                    'Length': len(streak_df_slice),
                    'Entry Day of Week': streak_df_slice.iloc[0]['Entry Day']
                })

        streaks_df = pd.DataFrame(losing_streaks_list)

        # --- 1. Losing Streaks Table ---
        st.subheader("ตารางสรุปช่วงเวลาที่ขาดทุนติดต่อกัน (Losing Streaks)")
        if streaks_df.empty:
            st.info("🎉 ยอดเยี่ยม! ไม่พบช่วงเวลาการขาดทุนติดต่อกันในข้อมูลนี้")
        else:
            day_order_table_streak = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            streaks_df_display = streaks_df[streaks_df['Entry Day of Week'].isin(day_order_table_streak)].copy()
            if streaks_df_display.empty:
                st.info("ℹ️ ไม่พบช่วงเวลาการขาดทุนที่เริ่มต้นในวันที่กำหนด (อาทิตย์-เสาร์) หรือข้อมูล 'Entry Day of Week' อาจมีปัญหา")
            else:
                streaks_df_display['Entry Day of Week'] = pd.Categorical(streaks_df_display['Entry Day of Week'], categories=day_order_table_streak, ordered=True)
                streaks_df_display = streaks_df_display.sort_values(['Start Date', 'Entry Day of Week']).reset_index(drop=True)
                st.dataframe(streaks_df_display)

        # --- 2. Histogram of Streak Lengths ---
        st.subheader("กราฟ Histogram แสดงความถี่ของความยาวช่วงที่ขาดทุนติดต่อกัน")
        if streaks_df.empty or 'Length' not in streaks_df.columns or streaks_df['Length'].empty:
            st.info("ℹ️ ไม่มีข้อมูลความยาวของช่วงขาดทุนสำหรับสร้าง Histogram")
        else:
            fig_streak_hist, ax_streak_hist = plt.subplots(figsize=(10, 6))
            max_len_streak = streaks_df['Length'].max() if not streaks_df['Length'].empty else 1
            
            if max_len_streak > 0 :
                bins_streak = np.arange(0.5, streaks_df['Length'].max() + 1.5, 1) # Center bins on integers
            else:
                bins_streak = np.arange(0.5, 1.5, 1) # Default bins if no streaks or max_len is 0

            sns.histplot(data=streaks_df, x='Length', bins=bins_streak, color='#F08080', edgecolor='white', alpha=0.8, ax=ax_streak_hist, discrete=True) # discrete can help align bars

            for p in ax_streak_hist.patches:
                height = p.get_height()
                if height > 0:
                    ax_streak_hist.annotate(f'{int(height)}',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha = 'center', va = 'center',
                                    xytext = (0, 5),
                                    textcoords = 'offset points',
                                    fontsize=8, color='black')

            ax_streak_hist.set_xticks(np.arange(1, max_len_streak + 1)) # Ensure integer ticks
            ax_streak_hist.set_xlabel('ความยาวของช่วงขาดทุนติดต่อกัน (จำนวนเทรด)')
            ax_streak_hist.set_ylabel('ความถี่')
            ax_streak_hist.set_title('Histogram of Losing Streak Lengths')
            ax_streak_hist.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig_streak_hist)
            plt.close(fig_streak_hist)

        # --- 3. Timeline Scatter Plot ---
        st.subheader("กราฟ Scatter แสดงความยาวของช่วงขาดทุนตามช่วงเวลาที่เริ่มเกิด")
        if streaks_df.empty or 'Start Date' not in streaks_df.columns or 'Length' not in streaks_df.columns:
            st.info("ℹ️ ไม่มีข้อมูลสำหรับสร้าง Scatter Plot ของช่วงขาดทุน")
        else:
            streaks_df_plot_scatter = streaks_df.copy()
            try: # Ensure 'Start Date' is datetime, it should be from .date() but convert just in case
                streaks_df_plot_scatter['Start Date'] = pd.to_datetime(streaks_df_plot_scatter['Start Date'])

                fig_streak_scatter, ax_streak_scatter = plt.subplots(figsize=(12, 6))
                ax_streak_scatter.scatter(streaks_df_plot_scatter['Start Date'], streaks_df_plot_scatter['Length'], color='#4682B4', alpha=0.7, s=50)

                ax_streak_scatter.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax_streak_scatter.get_xticklabels(), rotation=30, ha="right")
                ax_streak_scatter.set_xlabel('วันที่เริ่มช่วงขาดทุน')
                ax_streak_scatter.set_ylabel('ความยาวของช่วงขาดทุน')
                ax_streak_scatter.set_title('Losing Streak Lengths Over Time')
                ax_streak_scatter.grid(True, linestyle=':', alpha=0.6)

                if not streaks_df_plot_scatter['Length'].empty:
                     max_len_scatter = streaks_df_plot_scatter['Length'].max()
                     # Set y-ticks to be integers, with reasonable step
                     step = 1
                     if max_len_scatter > 20: step = max(1, max_len_scatter // 10)
                     elif max_len_scatter > 10: step = 2
                     ax_streak_scatter.set_yticks(np.arange(0, max_len_scatter + step, step))


                st.pyplot(fig_streak_scatter)
                plt.close(fig_streak_scatter)
            except Exception as e_scatter_timeline:
                 st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง Scatter Plot Timeline: {e_scatter_timeline}")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ Losing Streak: {e}")
        # st.exception(e)