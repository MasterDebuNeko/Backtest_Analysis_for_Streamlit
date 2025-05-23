import streamlit as st
import pandas as pd
from modules import R01_data_preparation
from modules import R02_equity_curves
from modules import R03_losing_streak
from modules import R04_profit_histogram
from modules import R05_trade_count_by_day
from modules import R06_trade_count_by_entry_time
from modules import R07_trade_count_by_exit_time
from modules import R08_heatmap_entry_time
from modules import R09_heatmap_exit_time
from modules import R10_mfe_mae_scatter
from modules import R11_mfe_histograms

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Backtest Analysis Dashboard")

# --- App Title ---
st.title("🚀 Backtest Analysis Dashboard")

# --- Initialize session state ---
if 'trade_results_df' not in st.session_state:
    st.session_state.trade_results_df = None
if 'excel_file_path_in_memory' not in st.session_state:
    st.session_state.excel_file_path_in_memory = None

# --- Sidebar for Inputs ---
st.sidebar.header("📌 Inputs")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File (List of trades & Properties)", type=["xlsx"])

if uploaded_file:
    # To read the excel file from memory without saving it temporarily
    st.session_state.excel_file_path_in_memory = uploaded_file
    st.sidebar.success("✅ File Uploaded Successfully!")
else:
    st.session_state.excel_file_path_in_memory = None # Reset if no file or file is removed
    st.session_state.trade_results_df = None # Reset dataframe if file is removed


# --- Main Application Logic ---
if st.session_state.excel_file_path_in_memory:
    st.header("Part 1: Data Preparation & Basic Stats")
    # 2. Stop Loss Input for R01
    default_stop_loss = 0.002
    desired_stop_loss = st.sidebar.number_input(
        "💸 Desired Stop Loss % (e.g., 0.002 for 0.2%)",
        min_value=0.0001,
        max_value=0.9999,
        value=default_stop_loss,
        step=0.0001,
        format="%.4f",
        key="stop_loss_pct_R01" # Unique key for this input
    )

    # Button to trigger analysis
    if st.sidebar.button("⚙️ Run Analysis", key="run_analysis_button"):
        try:
            # --- 01. Data Preparation ---
            with st.spinner("Processing Data Preparation..."):
                trade_results_df, summary_stats_R01 = R01_data_preparation.process_data(
                    st.session_state.excel_file_path_in_memory,
                    desired_stop_loss
                )
                st.session_state.trade_results_df = trade_results_df

            st.subheader("Processed Trade Results (First 5 Rows):")
            if st.session_state.trade_results_df is not None and not st.session_state.trade_results_df.empty:
                st.dataframe(st.session_state.trade_results_df.head())

                st.subheader("Summary Statistics (R-Multiples) from Data Prep:")
                if summary_stats_R01:
                    # Convert dict to DataFrame for better display
                    summary_df_R01 = pd.DataFrame(list(summary_stats_R01.items()), columns=['Statistic', 'Value'])
                    st.table(summary_df_R01.style.format({"Value": "{:.4f}"})) # Format float values
                else:
                    st.warning("No summary statistics generated from data preparation.")
            elif st.session_state.trade_results_df is not None and st.session_state.trade_results_df.empty:
                st.info("ℹ️ Data preparation resulted in an empty DataFrame. No trades to display or analyze.")
            else:
                st.error("❌ Error: Data preparation did not return a DataFrame.")

        except Exception as e:
            st.error(f"An error occurred during Data Preparation: {e}")
            st.session_state.trade_results_df = None # Reset on error

    # --- Subsequent Analysis Steps ---
    # These will only run if trade_results_df is available and the initial analysis was run.
    if st.session_state.trade_results_df is not None and not st.session_state.trade_results_df.empty:
        st.markdown("---")
        st.header("Part 2: Equity Curves")
        with st.spinner("Generating Equity Curves..."):
            R02_equity_curves.plot_equity_curve_allday(st.session_state.trade_results_df)
            R02_equity_curves.plot_equity_curve_by_day(st.session_state.trade_results_df)
        st.success("✅ Equity Curves Generated")

        st.markdown("---")
        st.header("Part 3: Losing Streak Analysis")
        with st.spinner("Analyzing Losing Streaks..."):
            R03_losing_streak.analyze_losing_streaks(st.session_state.trade_results_df)
        st.success("✅ Losing Streak Analysis Complete")

        st.markdown("---")
        st.header("Part 4: Profit Histograms")
        with st.spinner("Generating Profit Histograms..."):
            R04_profit_histogram.plot_profit_histogram_allday(st.session_state.trade_results_df)
            R04_profit_histogram.plot_profit_histogram_by_day(st.session_state.trade_results_df)
        st.success("✅ Profit Histograms Generated")

        st.markdown("---")
        st.header("Part 5: Trade Count by Entry Day")
        with st.spinner("Generating Trade Counts by Entry Day..."):
            R05_trade_count_by_day.plot_trade_count_by_entry_day(st.session_state.trade_results_df)
        st.success("✅ Trade Counts by Entry Day Generated")

        st.markdown("---")
        st.header("Part 6: Trade Count by Entry Time")
        # Input for R06 bin size
        bin_size_R06 = st.sidebar.number_input(
            "⏱️ Bin Size (minutes) for Entry Time",
            min_value=1,
            max_value=120,
            value=10, # Default from your script
            step=1,
            key="bin_size_R06"
        )
        if st.button("📊 Generate Entry Time Counts", key="run_R06"):
            with st.spinner("Generating Trade Counts by Entry Time..."):
                R06_trade_count_by_entry_time.plot_trade_count_by_entry_time(st.session_state.trade_results_df, bin_size_R06)
            st.success("✅ Trade Counts by Entry Time Generated")


        st.markdown("---")
        st.header("Part 7: Trade Count by Exit Time")
        # Input for R07 bin size
        bin_size_R07 = st.sidebar.number_input(
            "⏱️ Bin Size (minutes) for Exit Time",
            min_value=1,
            max_value=120,
            value=60, # Default from your script
            step=1,
            key="bin_size_R07"
        )
        if st.button("📊 Generate Exit Time Counts", key="run_R07"):
            with st.spinner("Generating Trade Counts by Exit Time..."):
                R07_trade_count_by_exit_time.plot_trade_count_by_exit_time(st.session_state.trade_results_df, bin_size_R07)
            st.success("✅ Trade Counts by Exit Time Generated")


        st.markdown("---")
        st.header("Part 8: Heatmap - Profit by Entry Time")
        # Input for R08 bin size
        bin_size_R08 = st.sidebar.number_input(
            "🌡️ Bin Size (minutes) for Entry Time Heatmap",
            min_value=1,
            max_value=120,
            value=20, # Default from your script
            step=1,
            key="bin_size_R08"
        )
        if st.button("🗺️ Generate Entry Time Heatmap", key="run_R08"):
            with st.spinner("Generating Entry Time Heatmap..."):
                R08_heatmap_entry_time.plot_heatmap_entry_time(st.session_state.trade_results_df, bin_size_R08)
            st.success("✅ Entry Time Heatmap Generated")


        st.markdown("---")
        st.header("Part 9: Heatmap - Profit by Exit Time")
        # Input for R09 bin size
        bin_size_R09 = st.sidebar.number_input(
            "🌡️ Bin Size (minutes) for Exit Time Heatmap",
            min_value=1,
            max_value=120,
            value=20, # Default from your script
            step=1,
            key="bin_size_R09"
        )
        if st.button("🗺️ Generate Exit Time Heatmap", key="run_R09"):
            with st.spinner("Generating Exit Time Heatmap..."):
                R09_heatmap_exit_time.plot_heatmap_exit_time(st.session_state.trade_results_df, bin_size_R09)
            st.success("✅ Exit Time Heatmap Generated")


        st.markdown("---")
        st.header("Part 10: MFE & MAE Scatter Plots")
        with st.spinner("Generating MFE & MAE Scatter Plots..."):
            R10_mfe_mae_scatter.plot_mfe_mae(st.session_state.trade_results_df)
            R10_mfe_mae_scatter.plot_mfe_profit(st.session_state.trade_results_df)
            R10_mfe_mae_scatter.plot_mae_profit(st.session_state.trade_results_df) # Assuming you meant MAE vs Profit for 10C
        st.success("✅ MFE & MAE Scatter Plots Generated")


        st.markdown("---")
        st.header("Part 11: MFE Histograms")
        with st.spinner("Generating MFE Histograms..."):
            R11_mfe_histograms.plot_mfe_histogram_all(st.session_state.trade_results_df)
            R11_mfe_histograms.plot_mfe_histogram_all_by_day(st.session_state.trade_results_df)
            R11_mfe_histograms.plot_mfe_histogram_losing(st.session_state.trade_results_df)
            R11_mfe_histograms.plot_mfe_histogram_losing_by_day(st.session_state.trade_results_df)
            R11_mfe_histograms.plot_mfe_histogram_breakeven(st.session_state.trade_results_df)
            R11_mfe_histograms.plot_mfe_histogram_breakeven_by_day(st.session_state.trade_results_df)
        st.success("✅ MFE Histograms Generated")

    elif st.session_state.trade_results_df is not None and st.session_state.trade_results_df.empty:
        st.info("ℹ️ Initial data processing resulted in an empty DataFrame. Subsequent analysis steps are skipped.")
    # else: (no file uploaded yet, or processing error previously)
    #    st.info("Please upload an Excel file and run the analysis to see the results.")

elif not uploaded_file:
    st.info("👋 Welcome! Please upload your Excel backtest file and set parameters to begin.")

# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.info("App by ข้า for ท่านพี่")