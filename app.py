# app.py (ฉบับปรับปรุง)

# 1. Import Libraries ที่จำเป็น
import streamlit as st
import pandas as pd # ยังคงจำเป็นสำหรับการจัดการ DataFrame ในบางส่วนของ UI หรือการส่งต่อ
import numpy as np

# Import ฟังก์ชันจากไฟล์ที่เราสร้างขึ้นใหม่
# from utils import CustomDivergingNorm # ไม่น่าจะต้องใช้ CustomDivergingNorm โดยตรงใน app.py แล้ว
from data_processing import calc_r_multiple_and_risk, summarize_r_multiple_stats
from plotting_equity import display_overall_equity_curve, display_equity_by_day
from plotting_streaks import display_losing_streak_analysis
from plotting_distributions import display_profit_distribution_all, display_profit_distribution_by_day
from plotting_trade_counts import display_trade_count_by_entry_day, display_trade_count_by_time_of_day
from plotting_heatmaps import display_profit_heatmap
from plotting_mfe_mae import display_mfe_mae_scatter_plots, display_mfe_histograms

# ตั้งค่าให้หน้าเว็บแสดงผลเต็มความกว้าง
st.set_page_config(layout="wide")

# หัวข้อหลักของ Dashboard ของเรา
st.title("🚀 Backtest Analysis Dashboard ของท่านพี่")
st.write("ยินดีต้อนรับสู่ Dashboard วิเคราะห์ผลการเทรด! อัปโหลดไฟล์ Excel แล้วมาดูกันเลยเจ้าค่ะ")
st.markdown("---")

# --- ส่วน UI หลักของ Streamlit ---
st.header("1. 📂 Data Preparation and Initial Analysis")
uploaded_file = st.file_uploader("กรุณาอัปโหลดไฟล์ Excel รายการเทรดของท่านพี่ (.xlsx)", type=["xlsx"], help="ไฟล์ควรมีชีทชื่อ 'List of trades' และ 'Properties' ตามรูปแบบที่กำหนดนะเจ้าคะ")
desired_stop_loss = st.number_input("ระบุ Stop Loss Percentage (เช่น 0.2% ให้ใส่ 0.002)", min_value=0.000001, max_value=0.999999, value=0.002, step=0.0001, format="%.6f", help="ค่า SL ต้องเป็นเลขทศนิยมที่มากกว่า 0 และน้อยกว่า 1")

# ใช้ st.session_state เพื่อติดตามว่าปุ่มถูกกดหรือยัง (ช่วยเรื่องการแสดง info message ตอนเริ่มต้น)
if 'analysis_button_pressed' not in st.session_state:
    st.session_state.analysis_button_pressed = False

if st.button("เริ่มการวิเคราะห์ชุดข้อมูลนี้ 🚀", help="กดปุ่มนี้หลังจากอัปโหลดไฟล์และตั้งค่า SL เรียบร้อยแล้ว", key="main_analysis_button"):
    st.session_state.analysis_button_pressed = True # ตั้งค่าเมื่อปุ่มถูกกด
    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily to pass its path
            # In a real-world scenario with larger files or cloud, other methods might be better
            with open("temp_uploaded_trade_list.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            excel_file_path_temp = "temp_uploaded_trade_list.xlsx"

            with st.spinner("กำลังประมวลผลข้อมูลการเทรดของท่านพี่... กรุณารอสักครู่เจ้าค่ะ... ⏳"):
                trade_results_df = calc_r_multiple_and_risk(excel_file_path_temp, desired_stop_loss)
            
            st.session_state['trade_results_df'] = trade_results_df # Store in session state
            st.success("ประมวลผลข้อมูลสำเร็จแล้วเจ้าค่ะ! 🎉")

            st.subheader("ตารางผลลัพธ์การเทรดเบื้องต้น (5 แถวแรก):")
            if trade_results_df is not None and not trade_results_df.empty:
                st.dataframe(trade_results_df.head())
                st.subheader("สรุปสถิติ R-Multiples โดยรวม:")
                summary_stats = summarize_r_multiple_stats(trade_results_df) # From data_processing.py
                if summary_stats:
                    col1, col2, col3 = st.columns(3)
                    stats_keys = list(summary_stats.keys())
                    
                    # Helper function for displaying metrics (could also be in utils.py if used elsewhere)
                    def display_metric_safe(column, label, value):
                        display_val = "N/A" if pd.isna(value) else (f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value))
                        column.metric(label=label, value=display_val)

                    metrics_per_col = (len(stats_keys) + 2) // 3 # Ensure it divides as evenly as possible
                    current_col_idx = 0
                    cols_ui = [col1, col2, col3]
                    for i, key in enumerate(stats_keys):
                        display_metric_safe(cols_ui[current_col_idx], key, summary_stats[key])
                        if (i + 1) % metrics_per_col == 0 and current_col_idx < 2:
                            current_col_idx += 1
                else:
                    st.info("ℹ️ ไม่สามารถคำนวณสถิติสรุปได้ (อาจไม่มีข้อมูลเทรดที่ถูกต้อง).")
            elif trade_results_df is None: # Should not happen if calc_r_multiple_and_risk always returns a df
                st.error("❌ เกิดข้อผิดพลาด: ฟังก์ชัน `calc_r_multiple_and_risk` ไม่ได้คืนค่า DataFrame.")
            else: # trade_results_df is empty
                st.info("ℹ️ ฟังก์ชัน `calc_r_multiple_and_risk` คืนค่าเป็น DataFrame ที่ว่างเปล่า (อาจไม่มีข้อมูลเทrด Exit ที่ถูกต้อง).")

        except ValueError as ve: st.error(f"❌ ข้อมูล Input ไม่ถูกต้อง หรือมีปัญหาในการประมวลผลข้อมูล: {ve}")
        except RuntimeError as re: st.error(f"❌ เกิดข้อผิดพลาดขณะโหลดหรือประมวลผลไฟล์: {re}")
        except KeyError as ke: st.error(f"❌ ไม่พบคอลัมน์ที่สำคัญในไฟล์ Excel: {ke}.")
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
            st.exception(e) # Show full traceback for unexpected errors
            st.error("กรุณาตรวจสอบรูปแบบไฟล์ Excel และลองใหม่อีกครั้งนะเจ้าคะ หรือติดต่อผู้พัฒนาหากปัญหายังคงอยู่")
    else:
        st.warning("⚠️ กรุณาอัปโหลดไฟล์ Excel ก่อนกดปุ่มเริ่มการวิเคราะห์นะเจ้าคะ")
        # Reset trade_results_df if no file is uploaded to clear old results
        if 'trade_results_df' in st.session_state:
            del st.session_state['trade_results_df']


st.markdown("---")
st.caption("ℹ️ *หากท่านพี่อัปโหลดไฟล์ใหม่หรือเปลี่ยนค่า Stop Loss กรุณากดปุ่ม 'เริ่มการวิเคราะห์ฯ' อีกครั้งเพื่ออัปเดตผลลัพธ์ทั้งหมดนะเจ้าคะ*")

# --- ส่วนการแสดงผลการวิเคราะห์ต่างๆ ---
# Check if data is processed and available in session_state
if 'trade_results_df' in st.session_state and \
   st.session_state['trade_results_df'] is not None and \
   not st.session_state['trade_results_df'].empty:

    current_df = st.session_state['trade_results_df']

    # ส่วนที่ 2: Equity Curve Analysis
    display_overall_equity_curve(current_df) # From plotting_equity.py
    display_equity_by_day(current_df)        # From plotting_equity.py
    st.markdown("---")

    # ส่วนที่ 3: Losing Streak Analysis
    display_losing_streak_analysis(current_df) # From plotting_streaks.py
    st.markdown("---")

    # ส่วนที่ 4: Profit(R) Distribution
    display_profit_distribution_all(current_df)    # From plotting_distributions.py
    display_profit_distribution_by_day(current_df) # From plotting_distributions.py
    st.markdown("---")

    # ส่วนที่ 5: Trade Count by Entry Day
    display_trade_count_by_entry_day(current_df) # From plotting_trade_counts.py
    st.markdown("---")

    # ส่วนที่ 6 & 7: Trade Count by Time of Day (Entry & Exit)
    st.header("6 & 7. ⏰ Trade Count by Time of Day (Entry & Exit)")
    st.markdown("วิเคราะห์จำนวนเทรดตามช่วงเวลาของวัน โดยสามารถปรับขนาดของช่วงเวลา (Bin size) ได้ และจะมีการข้ามช่วงเวลา 12:00-19:30 น.")

    st.markdown("### 6. วิเคราะห์ตามเวลาเข้าเทรด (Entry Time)")
    bin_size_entry_time_input_app = st.number_input(
        "เลือกขนาด Bin สำหรับเวลาเข้าเทรด (นาที):",
        min_value=1, max_value=120, value=10, step=1,
        key="bin_entry_time_app", # Ensure unique key
        help="ขนาดของแต่ละช่วงเวลาที่จะใช้ในการจัดกลุ่มข้อมูล เช่น 10 นาที, 30 นาที, 60 นาที"
    )
    display_trade_count_by_time_of_day(current_df, 'Entry Time', 'Entry Time of Day', bin_size_entry_time_input_app, "entry_app_tc")

    st.markdown("---")
    st.markdown("### 7. วิเคราะห์ตามเวลาออกจากเทรด (Exit Time)")
    bin_size_exit_time_input_app = st.number_input(
        "เลือกขนาด Bin สำหรับเวลาออกจากเทรด (นาที):",
        min_value=1, max_value=120, value=60, step=1,
        key="bin_exit_time_app", # Ensure unique key
        help="ขนาดของแต่ละช่วงเวลาที่จะใช้ในการจัดกลุ่มข้อมูล"
    )
    if 'Exit Time' in current_df.columns:
        display_trade_count_by_time_of_day(current_df, 'Exit Time', 'Exit Time of Day', bin_size_exit_time_input_app, "exit_app_tc")
    else:
        st.warning("⚠️ ไม่พบคอลัมน์ 'Exit Time' ในข้อมูล. ไม่สามารถทำการวิเคราะห์ตามเวลาออกจากเทรดได้.")
    st.markdown("---")

    # ส่วนที่ 8 & 9: Heatmap Analysis
    st.header("8 & 9. 🔥 Heatmap Analysis: Profit(R) by Time and Day")
    st.markdown("Heatmap แสดงผลรวมของ Profit(R), จำนวนเทรด, และค่าเฉลี่ย Profit(R) โดยแบ่งตามวันในสัปดาห์และช่วงเวลาของวัน (สามารถปรับขนาด Bin ได้) โดยจะข้ามการแสดงผลช่วงเวลา 12:00-19:30 น.")

    st.markdown("### 8. Heatmap ตามเวลาเข้าเทรด (Entry Time)")
    bin_size_heatmap_entry_input_app = st.number_input(
        "เลือกขนาด Bin สำหรับ Heatmap เวลาเข้าเทรด (นาที):",
        min_value=1, max_value=120, value=20, step=1,
        key="bin_heatmap_entry_app", # Ensure unique key
        help="ขนาดของแต่ละช่วงเวลาที่จะใช้ในการจัดกลุ่มข้อมูลสำหรับ Heatmap"
    )
    # For display_profit_heatmap, the 'day_column_name_to_derive' is derived from 'time_column_name'
    display_profit_heatmap(current_df, 'Entry Time', 'Entry Day', 'Entry Time of Day', bin_size_heatmap_entry_input_app, "heatmap_entry_app")

    st.markdown("---")
    st.markdown("### 9. Heatmap ตามเวลาออกจากเทรด (Exit Time)")
    bin_size_heatmap_exit_input_app = st.number_input(
        "เลือกขนาด Bin สำหรับ Heatmap เวลาออกจากเทรด (นาที):",
        min_value=1, max_value=120, value=20, step=1,
        key="bin_heatmap_exit_app", # Ensure unique key
        help="ขนาดของแต่ละช่วงเวลาที่จะใช้ในการจัดกลุ่มข้อมูลสำหรับ Heatmap"
    )
    if 'Exit Time' in current_df.columns:
        # display_profit_heatmap will derive 'Exit Day' from 'Exit Time' inside the function
        # No need to create df_for_exit_heatmap_app here as the function handles copying and derivation
        display_profit_heatmap(current_df, 'Exit Time', 'Exit Day', 'Exit Time of Day', bin_size_heatmap_exit_input_app, "heatmap_exit_app")
    else:
        st.warning("⚠️ ไม่พบคอลัมน์ 'Exit Time' ในข้อมูล. ไม่สามารถสร้าง Heatmap ตามเวลาออกจากเทรดได้.")
    st.markdown("---")

    # ส่วนที่ 10: MFE/MAE Scatter Plots
    display_mfe_mae_scatter_plots(current_df) # From plotting_mfe_mae.py
    st.markdown("---")

    # ส่วนที่ 11: MFE Histograms
    display_mfe_histograms(current_df) # From plotting_mfe_mae.py

elif st.session_state.analysis_button_pressed: # If button was pressed but data is not available/empty
    st.info("ℹ️ ไม่มีข้อมูลเทรดที่ประมวลผลได้ หรือข้อมูลว่างเปล่า กรุณาตรวจสอบไฟล์ Excel ของท่านพี่ หรือลองอัปโหลดไฟล์ใหม่อีกครั้งนะเจ้าคะ")
# else: # Initial state, button not pressed yet, no file uploaded, or after clearing results
#    st.info("กรุณาอัปโหลดไฟล์ Excel และกดปุ่ม 'เริ่มการวิเคราะห์ฯ' เพื่อดูผลลัพธ์นะเจ้าคะ")
