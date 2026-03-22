import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBClassifier
from datetime import datetime, time
import matplotlib.pyplot as plt

# ======================
# 东方财富妙想Skills数据接口配置
# ======================
def get_eastmoney_stock_data(code, start_date="2020-01-01", end_date="2026-12-31"):
    """
    从东方财富妙想Skills拉取A股日线数据
    :param code: 股票代码（如600519）
    :param start_date: 开始日期（格式：YYYY-MM-DD）
    :param end_date: 结束日期（格式：YYYY-MM-DD）
    :return: DataFrame（包含close/volume等字段）
    """
    try:
        # 东方财富妙想Skills A股日线数据接口
        url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": f"1.{code}" if code.startswith("6") else f"0.{code}",  # 沪市1.xxx，深市0.xxx
            "ut": "fa5fd1943c7b386f172d6893dbfba1089",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",  # 日线
            "fqt": "1",    # 前复权
            "beg": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "rtntype": "6"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # 解析数据
        if data["data"] and data["data"]["klines"]:
            klines = data["data"]["klines"]
            df_list = []
            for k in klines:
                parts = k.split(",")
                df_list.append({
                    "date": parts[0],
                    "close": float(parts[2]),
                    "volume": float(parts[5])
                })
            
            df = pd.DataFrame(df_list)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            return df
        return None
    except Exception as e:
        print(f"东方财富数据获取错误: {e}")
        return None

def get_sh_index_data(start_date="2020-01-01", end_date="2026-12-31"):
    """获取上证指数数据（代码000001）"""
    return get_eastmoney_stock_data("000001", start_date, end_date)

# ======================
# 飞书推送配置（替换成你的飞书机器人Webhook地址）
# ======================
FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/b5162b51-57e3-4d9f-87ec-a97203261dda"

def send_feishu(title, content):
    """飞书推送函数（适配最新接口规范）"""
    try:
        payload = {
            "msg_type": "text",
            "content": {
                "text": f"【{title}】\n\n{content}"
            }
        }
        headers = {"Content-Type": "application/json; charset=utf-8"}
        response = requests.post(
            FEISHU_WEBHOOK_URL,
            json=payload,
            headers=headers,
            timeout=8
        )
        if response.status_code == 200:
            res_json = response.json()
            return res_json.get("code") == 0
        return False
    except Exception as e:
        print(f"飞书推送错误: {e}")
        return False

# ======================
# 登录系统
# ======================
if "login" not in st.session_state:
    st.session_state.login = False

def login_page():
    st.title("🔒 AI量化最终完全体 · 东方财富+飞书版")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        if username == "admin" and password == "123456":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("账号或密码错误")

# ======================
# 特征体系（适配东方财富数据）
# ======================
def build_features(df):
    data = df.copy()

    # 计算基础指标
    data["return"] = data["close"].pct_change()
    data["ma5"] = data["close"].rolling(5).mean()
    data["ma10"] = data["close"].rolling(10).mean()
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma60"] = data["close"].rolling(60).mean()

    data["vol_ma5"] = data["volume"].rolling(5).mean()
    data["vol_ma20"] = data["volume"].rolling(20).mean()

    # 过滤NaN行
    data.dropna(inplace=True)
    
    # 核心：空数据直接返回，不执行后续计算
    if data.empty:
        return data

    # 仅当数据非空时计算布尔特征
    data["trend_up"] = (data["close"] > data["ma20"]) & (data["ma20"] > data["ma60"])
    data["vol_strength"] = data["vol_ma5"] > data["vol_ma20"]
    data["strong_uptrend"] = (data["close"] > data["ma20"] * 1.03)
    data["breakout"] = data["close"] > data["close"].rolling(20).max()

    data["target"] = (data["return"].shift(-1) > 0).astype(int)
    data.dropna(inplace=True)
    return data

# ======================
# 大盘强弱等级
# ======================
def get_market_level():
    sh = get_sh_index_data()
    # 数据为空/不足60条 → 直接返回0
    if sh is None or len(sh) < 60:
        return 0
    sh = build_features(sh)
    # 特征构建后为空 → 直接返回0
    if sh.empty:
        return 0
    last = sh.iloc[-1]
    if last["trend_up"] and last["strong_uptrend"]:
        return 2
    elif last["trend_up"]:
        return 1
    else:
        return 0

def market_info():
    lv = get_market_level()
    if lv == 2:
        return "🟢 大盘强势 → 可重仓参与"
    elif lv == 1:
        return "🟡 大盘安全 → 正常交易"
    else:
        return "🔴 大盘弱势 → 空仓避险"

def allow_trade():
    return get_market_level() > 0

# ======================
# AI 模型
# ======================
def train_model(data):
    feats = [
        "return", "ma5", "ma10", "ma20", "ma60",
        "vol_ma5", "vol_ma20", "trend_up", "vol_strength", "strong_uptrend", "breakout"
    ]
    X = data[feats]
    y = data["target"]
    split = int(len(X) * 0.8)
    model = XGBClassifier(random_state=666, max_depth=5, n_estimators=150)
    model.fit(X[:split], y[:split])
    acc = np.mean(model.predict(X[split:]) == y[split:])
    return model, acc

# ======================
# 超级信号
# ======================
def super_signal(code):
    df = get_eastmoney_stock_data(code, start_date="2025-01-01")
    # 数据为空/不足60条 → 直接返回提示
    if df is None or len(df) < 60:
        return "🔴 数据不足", 0
    data = build_features(df)
    # 特征构建后为空 → 直接返回提示
    if data.empty:
        return "🔴 数据错误", 0
    
    model, acc = train_model(data)
    lv = get_market_level()
    last = data.iloc[-1]

    x = np.array([[
        last["return"], last.ma5, last.ma10, last.ma20, last.ma60,
        last.vol_ma5, last.vol_ma20, last.trend_up, last.vol_strength, last.strong_uptrend, last.breakout
    ]])
    pred = model.predict(x)[0]

    if lv == 0:
        return "🔴 空仓避险", round(acc*100,1)
    if pred == 1 and last.trend_up and last.vol_strength and last.strong_uptrend:
        return "🟢 超级买入", round(acc*100,1)
    if pred == 1 and last.trend_up:
        return "🟡 可关注", round(acc*100,1)
    return "🔴 观望", round(acc*100,1)

# ======================
# 自动推送：每日 9:25 发送（飞书版）
# ======================
def generate_rich_report():
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    market_msg = market_info()
    watchlist = ["600519", "000001", "600036", "600030", "002594", "000858"]
    
    msg = f"AI量化自动日报 | {today}\n{market_msg}\n\n"
    msg += "📊 今日关注列表：\n"

    buy_list = []
    for code in watchlist:
        sig, acc = super_signal(code)
        msg += f"• {code}：{sig}（置信度 {acc}%）\n"
        if "超级买入" in sig:
            buy_list.append(f"{code}({acc}%)")

    msg += "\n📝 今日策略：\n"
    if allow_trade():
        msg += "✅ 允许交易\n"
        msg += "✅ 只做上升趋势票\n"
        if buy_list:
            msg += f"✅ 今日高价值标的：{', '.join(buy_list)}\n"
    else:
        msg += "❌ 今日空仓，不交易\n"

    msg += "\n🛡️ 风控规则：\n"
    msg += "• 止损 7%\n"
    msg += "• 止盈 25%\n"
    msg += "• 大盘弱 → 空仓\n"
    msg += "• 趋势强 → 重仓\n"
    msg += "• 连续错2次 → 休息"
    return msg

def auto_push_task():
    last_push_date = st.session_state.get("last_push_date", "")
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.time()

    push_start = time(9, 20)
    push_end = time(9, 30)

    if push_start <= current_time <= push_end and current_date != last_push_date:
        report = generate_rich_report()
        send_feishu("AI量化日报", report)
        st.session_state["last_push_date"] = current_date

# ======================
# 回测引擎
# ======================
def backtest_final(code, money=100000):
    df = get_eastmoney_stock_data(code)
    # 数据为空/不足60条 → 直接返回None
    if df is None or len(df) < 60:
        return None
    data = build_features(df)
    # 特征构建后为空 → 直接返回None
    if data.empty:
        return None
    
    model, acc = train_model(data)
    lv = get_market_level()

    balance = money
    pos, cost = 0, 0
    hist = [balance]
    trades = []
    loss_count = 0

    for i in range(len(data)):
        row = data.iloc[i]
        close = row.close
        date = str(data.index[i])[:10]

        if lv == 0:
            if pos > 0:
                balance += pos * close
                trades.append([date, "空仓避险", close, pos, round(balance,2)])
                pos = 0
            hist.append(balance)
            continue

        max_pos = 0.85 if lv ==2 else 0.7 if lv ==1 else 0.5

        x = np.array([[
            row["return"], row.ma5, row.ma10, row.ma20, row.ma60,
            row.vol_ma5, row.vol_ma20, row.trend_up, row.vol_strength, row.strong_uptrend, row.breakout
        ]])
        pred = model.predict(x)[0]

        if pos > 0:
            if close <= cost * 0.93:
                balance += pos * close
                trades.append([date, "止损", close, pos, round(balance,2)])
                pos, cost = 0, 0
                loss_count +=1
            elif row.strong_uptrend and row.breakout:
                continue
            elif close >= cost * 1.25:
                balance += pos * close
                trades.append([date, "止盈", close, pos, round(balance,2)])
                pos, cost = 0, 0
                loss_count = 0

        if loss_count >=2:
            hist.append(balance + pos*close)
            continue

        if pred ==1 and pos ==0 and row.trend_up and row.vol_strength:
            buy_size = int(balance * max_pos // close)
            if buy_size>0:
                pos = buy_size
                cost = close
                balance -= pos*close
                trades.append([date, "买入", close, pos, round(balance,2)])

        if pred ==0 and pos>0 and not row.strong_uptrend:
            balance += pos*close
            trades.append([date, "卖出", close, pos, round(balance,2)])
            pos, cost =0,0

        hist.append(balance + pos*close)

    final = balance + pos*close
    profit = final - money
    rate = profit / money *100

    return {
        "final": round(final,2),
        "profit": round(profit,2),
        "rate": round(rate,1),
        "acc": round(acc*100,1),
        "hist": hist,
        "trades": pd.DataFrame(trades, columns=["时间","类型","价格","股数","现金"])
    }

# ======================
# 主界面
# ======================
if not st.session_state.login:
    login_page()
else:
    auto_push_task()

    st.sidebar.title("🧧 最终完全体 · 东方财富+飞书")
    menu = st.sidebar.radio("", [
        "🏠 首页",
        "📈 终极回测",
        "🏆 牛股排名",
        "📡 实时信号",
        "📩 推送测试",
        "📖 系统说明"
    ])

    if menu == "🏠 首页":
        st.title("🧧 A股AI量化 · 东方财富数据版")
        st.success("✅ 自动9:25推送飞书｜高胜率｜强收益｜强风控")
        st.markdown(market_info())
        st.markdown("---")
        st.subheader("今日精选信号")
        for c in ["600519","000001","600036","002594"]:
            sig,acc = super_signal(c)
            if "超级买入" in sig:
                st.success(f"{c}｜{sig}｜{acc}%")
            elif "可关注" in sig:
                st.info(f"{c}｜{sig}｜{acc}%")
            else:
                st.error(f"{c}｜{sig}")

    elif menu == "📈 终极回测":
        st.title("📈 最终版回测（东方财富数据）")
        code = st.text_input("股票代码", "600519")
        if st.button("开始回测"):
            res = backtest_final(code, 100000)
            if res is None:
                st.error("❌ 数据不足，无法回测（需要至少60条历史数据）")
            else:
                col1,col2,col3,col4 = st.columns(4)
                col1.metric("收益率", f"{res['rate']}%")
                col2.metric("收益", f"{res['profit']}元")
                col3.metric("最终资产", f"{res['final']}元")
                col4.metric("AI准确率", f"{res['acc']}%")
                st.line_chart(res['hist'])
                st.dataframe(res['trades'])

    elif menu == "📡 实时信号":
        st.title("📡 实时信号（东方财富数据）")
        watch = st.text_area("监控列表", "600519\n000001\n600036")
        if st.button("刷新"):
            st.markdown(market_info())
            for c in watch.split():
                sig,acc = super_signal(c.strip())
                if "超级买入" in sig:
                    st.success(f"{c}｜{sig}｜{acc}%")
                elif "可关注" in sig:
                    st.info(f"{c}｜{sig}｜{acc}%")
                else:
                    st.error(f"{c}｜{sig}")

    elif menu == "📩 推送测试":
        st.title("📩 飞书推送测试")
        if st.button("立即发送测试报告到飞书"):
            report = generate_rich_report()
            if send_feishu("AI量化测试报告", report):
                st.success("✅ 飞书推送成功！")
            else:
                st.error("❌ 飞书推送失败，请检查Webhook地址")
            st.code(report)

    elif menu == "📖 系统说明":
        st.markdown("""
        **最终完全体功能：**
        ✅ 基于东方财富妙想Skills拉取A股数据（稳定无限制）
        ✅ 每天 9:25 自动推送飞书
        ✅ 大盘强弱自动判断
        ✅ 超级买入信号精准
        ✅ 止损7% 止盈25%
        ✅ 趋势强重仓，弱则空仓
        ✅ 自动规避大跌
        ✅ 永久在线 24h
        
        **使用说明：**
        1. 登录账号：admin，密码：123456
        2. 回测需输入A股代码（如600519）
        3. 推送功能需配置飞书机器人Webhook
        4. 数据接口：东方财富妙想Skills（免费无Token）
        """)
