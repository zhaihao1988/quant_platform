# ui/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from db.database import get_engine

st.title("量化策略研究平台")

# 侧边栏选择股票及日期范围
symbol = st.sidebar.text_input("股票代码", "000300.XSHG")
start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("结束日期", pd.to_datetime("2023-12-31"))

if st.sidebar.button("查询"):
    engine = get_engine()
    query = f"SELECT date, close FROM stock_price WHERE symbol = '{symbol}' AND date BETWEEN '{start_date}' AND '{end_date}' ORDER BY date"
    df = pd.read_sql(query, engine, parse_dates=["date"])
    if df.empty:
        st.write("无数据，请先更新数据库。")
    else:
        df.set_index("date", inplace=True)
        st.line_chart(df["close"], use_container_width=True)
        st.write(f"{symbol} 收盘价走势图")
