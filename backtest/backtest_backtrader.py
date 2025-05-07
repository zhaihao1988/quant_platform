# backtest/backtest_backtrader.py
import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
from datetime import datetime
import os
import sys
from sqlalchemy import create_engine, text  # 导入 sqlalchemy

# --- 添加项目根目录和策略导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategies.backtrader_ma_pullback_strategy import MAPullbackPeakCondBtStrategy
from config.settings import settings  # 导入数据库配置


# --- 数据加载函数 (从数据库加载) ---
def load_data_from_db(symbol, start_date_str, end_date_str):
    """
    从 PostgreSQL 数据库加载数据并转换为 Backtrader Feed 格式。
    """

    print(f"Loading data for {symbol} from DB ({start_date_str} to {end_date_str})...")

    db_url = settings.DB_URL  # 从配置文件获取数据库连接字符串
    if not db_url:
        print("Error: DB_URL not found in settings.")
        return None

    engine = None  # Initialize engine to None
    try:
        engine = create_engine(db_url)

        query = text(f"""
            SELECT date, open, high, low, close, volume
            FROM stock_daily
            WHERE symbol = :symbol
              AND date >= :start_date
              AND date <= :end_date
            ORDER BY date ASC
        """)

        with engine.connect() as connection:
            dataframe = pd.read_sql(query, connection, params={
                "symbol": symbol,
                "start_date": start_date_str,
                "end_date": end_date_str
            }, index_col='date', parse_dates=['date'])
        # 在 backtest/backtest_backtrader.py 的 load_data_from_db 函数中
        # ...
        if not dataframe.empty:
            print(
                f"INFO: For symbol {symbol}, actual data in DataFrame from {dataframe.index.min()} to {dataframe.index.max()}")  # <--- 确认这行已添加
        # ...
        if dataframe.empty:
            print(f"No data found for {symbol} in DB for the specified date range.")
            return None

        dataframe.columns = [x.lower() for x in dataframe.columns]

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in dataframe.columns for col in required_cols):
            print(f"Error: Missing required columns from DB for {symbol}. Need: {required_cols}")
            return None

        dataframe['openinterest'] = 0.0

        print(f"Data loaded successfully for {symbol} from DB: {len(dataframe)} rows")
        return bt.feeds.PandasData(dataname=dataframe)

    except ImportError:
        print("Error: SQLAlchemy not installed. Cannot connect to database. pip install SQLAlchemy psycopg2-binary")
        return None
    except Exception as e:
        print(f"Error loading data from database for {symbol}: {e}")
        if engine:  # Check if engine was initialized before disposing
            engine.dispose()
        return None
    finally:
        if engine:  # Ensure engine exists and dispose
            engine.dispose()


# --- Custom Commission Scheme ---
class MyCommissionInfo(bt.CommInfoBase):
    """
    Custom commission scheme that includes a minimum commission.
    """
    params = (
        ('commission', 0.0),  # Standard commission rate (e.g., 0.0002 for 0.02%)
        ('mincommission', 0.0),  # Minimum commission per trade
        ('stocklike', True),  # True for stocks (percentage-based commission)
        ('commtype', bt.CommInfoBase.COMM_PERC),  # Commission type: COMM_PERC or COMM_FIXED
        ('percabs', True),  # If True, commission is an absolute percentage (e.g., 0.01 = 1%)
        # If False, commission is like 1 for 1%
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        Calculates the commission for a trade.
        Args:
            size: The number of shares/contracts in the trade.
            price: The price at which the trade occurs.
            pseudoexec: Boolean, True if the commission is being calculated for a potential
                        trade rather than an actual execution.
        Returns:
            The calculated commission amount.
        """
        # Calculate the base commission
        if self.p.commtype == bt.CommInfoBase.COMM_PERC:
            # Percentage-based commission
            # Ensure size is absolute for commission calculation
            comm_amount = abs(size) * price * self.p.commission
        elif self.p.commtype == bt.CommInfoBase.COMM_FIXED:
            # Fixed amount per share/contract
            comm_amount = abs(size) * self.p.commission
        else:
            # Should not happen with current setup, but good to have a fallback
            comm_amount = 0.0

        # Apply the minimum commission
        # The commission cannot be less than the minimum commission
        final_commission = max(comm_amount, self.p.mincommission)
        return final_commission


# --- 回测参数 ---
STOCK_POOL = {
    #'000887': '中鼎股份', '300138': '晨光生物', '002480': '大金重工',
    #'002607': '胜宏科技',
    '603920': '世运电路'
}
START_DATE_STR = "2010-01-01"
END_DATE_STR = "2025-05-07"
INITIAL_CAPITAL = 1000000.0
COMMISSION_RATE = 0.0002  # Example: 0.02%
MIN_COMMISSION = 5.0  # Example: 5 currency units minimum
SLIPPAGE_PERCENT = 0.001
MAX_SINGLE_POSITION_RATIO = 0.20

if __name__ == '__main__':
    cerebro = bt.Cerebro(stdstats=False)  # Disable standard observers for cleaner output

    # Add strategy
    cerebro.addstrategy(MAPullbackPeakCondBtStrategy, printlog=True, max_position_ratio=MAX_SINGLE_POSITION_RATIO)

    data_loaded_count = 0
    for symbol in STOCK_POOL.keys():
        data_feed = load_data_from_db(symbol, START_DATE_STR, END_DATE_STR)

        if data_feed is not None:
            cerebro.adddata(data_feed, name=symbol)
            data_loaded_count += 1
        else:
            print(f"Warning: Could not load data for {symbol}. Skipping.")

    if data_loaded_count == 0:
        print("Error: No data loaded. Backtest cannot run.")
        sys.exit()

    # Set initial cash
    cerebro.broker.setcash(INITIAL_CAPITAL)

    # --- CORRECTED COMMISSION SETUP ---
    comminfo = MyCommissionInfo(commission=COMMISSION_RATE, mincommission=MIN_COMMISSION)
    cerebro.broker.addcommissioninfo(comminfo)
    # --- END CORRECTED COMMISSION SETUP ---

    # Set slippage
    cerebro.broker.set_slippage_perc(perc=SLIPPAGE_PERCENT)

    # Add analyzers
    # Corrected: bt.analyzers.Sharpe to bt.analyzers.sharpe
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years) # 使用 SharpeRatio 类
    # Note: For SharpeRatio, the class is typically bt.analyzers.SharpeRatio,
    # but the error message suggests 'sharpe' is a direct attribute.
    # We'll stick to the error's suggestion. If other issues arise, we might need to use SharpeRatio.
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Years)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')  # System Quality Number

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()  # Run the backtest
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Print analysis results
    try:
        strat = results[0]  # Get the first strategy (assuming only one)
        print("\n--- Backtrader Analysis Results ---")

        # Returns Analysis
        if hasattr(strat.analyzers, 'returns') and strat.analyzers.returns.get_analysis():
            returns_analysis = strat.analyzers.returns.get_analysis()
            print(f"Annualized Return: {returns_analysis.get('rann', 0) * 100:.2f}%")
        else:
            print("Returns analysis not available.")

        # Sharpe Ratio Analysis
        # Ensure we use the correct name ('sharpe') used in addanalyzer
        if hasattr(strat.analyzers, 'sharpe') and strat.analyzers.sharpe.get_analysis():
            sharpe_analysis = strat.analyzers.sharpe.get_analysis()
            # The key for Sharpe ratio in the results is often 'sharperatio'
            print(f"Sharpe Ratio (Annualized): {sharpe_analysis.get('sharperatio', 'N/A')}")
        else:
            print("Sharpe Ratio analysis not available.")

        # Drawdown Analysis
        if hasattr(strat.analyzers, 'drawdown') and strat.analyzers.drawdown.get_analysis():
            drawdown_analysis = strat.analyzers.drawdown.get_analysis()
            print(f"Max Drawdown: {drawdown_analysis.max.drawdown:.2f}%")
            print(f"Max Drawdown Money: {drawdown_analysis.max.moneydown:.2f}")
        else:
            print("Drawdown analysis not available.")

        # SQN Analysis
        if hasattr(strat.analyzers, 'sqn') and strat.analyzers.sqn.get_analysis():
            sqn_analysis = strat.analyzers.sqn.get_analysis()
            print(f"System Quality Number (SQN): {sqn_analysis.get('sqn', 'N/A'):.2f}")
        else:
            print("SQN analysis not available.")

        # Trade Analysis
        if hasattr(strat.analyzers, 'trades') and strat.analyzers.trades.get_analysis():
            trade_analysis = strat.analyzers.trades.get_analysis()
            total_trades = trade_analysis.total.total if trade_analysis.total is not None else 0
            won_trades = trade_analysis.won.total if trade_analysis.won is not None else 0
            lost_trades = trade_analysis.lost.total if trade_analysis.lost is not None else 0

            total_pnl = trade_analysis.pnl.net.total if trade_analysis.pnl and trade_analysis.pnl.net else 0
            won_pnl = trade_analysis.won.pnl.total if trade_analysis.won and trade_analysis.won.pnl else 0
            lost_pnl = trade_analysis.lost.pnl.total if trade_analysis.lost and trade_analysis.lost.pnl else 0

            print(f"Total Trades: {total_trades}")
            if total_trades > 0:
                print(f"  - Won: {won_trades}")
                print(f"  - Lost: {lost_trades}")
                win_rate = (won_trades / total_trades * 100) if total_trades else 0
                print(f"Win Rate: {win_rate:.2f}%")
                avg_win_pnl = (won_pnl / won_trades) if won_trades else 0
                print(f"Avg Win PNL: {avg_win_pnl:.2f}")
                avg_loss_pnl = (lost_pnl / lost_trades) if lost_trades else 0
                print(f"Avg Loss PNL: {avg_loss_pnl:.2f}")
                profit_factor = (won_pnl / abs(lost_pnl)) if lost_pnl != 0 else float('inf')
                print(f"Profit Factor: {profit_factor:.2f}")
        else:
            print("Trade analysis not available.")

    except Exception as e:
        print(f"Error getting analysis results: {e}")

    # Plotting the results
    output_dir = "backtest_results_backtrader"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if data_loaded_count > 0:  # Only plot if data was loaded
        try:
            combined_plot_filename = os.path.join(output_dir, 'backtrader_ALL_STOCKS_plot_清晰版.png')  # 新文件名
            print(f"\nAttempting to save combined plot to {combined_plot_filename}")

            # cerebro.plot() returns a list of figures, typically [[figure]] for numfigs=1
            figures = cerebro.plot(style='candlestick', barup='red', bardown='green', volup='red', voldown='green',
                                   iplot=False, numfigs=1,  # numfigs=1 produces one figure with subplots for each data
                                   savefig=False,  # Set savefig=False to handle saving manually for better control
                                   figscale=1.8,  # Increase scale for better readability
                                   plotdist=0.05  # Adjust distance between plots if needed
                                   )

            if figures and figures[0] and figures[0][0]:
                # figures[0][0] is the matplotlib Figure object
                fig = figures[0][0]
                fig.savefig(combined_plot_filename, dpi=300)  # Save with high DPI
                print(f"Combined plot saved successfully with high DPI to {combined_plot_filename}")
            else:
                # Fallback if figure object wasn't retrieved as expected
                print("Could not retrieve figure object directly, attempting standard save.")
                fallback_filename = os.path.join(output_dir, 'backtrader_ALL_STOCKS_plot_fallback.png')
                cerebro.plot(style='candlestick', barup='red', bardown='green', volup='red', voldown='green',
                             iplot=False, numfigs=1, savefig=True, figfilename=fallback_filename, figscale=1.5)
                print(f"Plot saved (fallback method) to {fallback_filename}")

        except Exception as plot_err:
            print(f"Error during plotting: {plot_err}")
            print(
                "Plotting requires matplotlib. Ensure it's installed and a suitable backend is configured (e.g., Agg for non-GUI).")
    else:
        print("No data was loaded, skipping plot generation.")