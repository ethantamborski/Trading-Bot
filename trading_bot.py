#!/usr/bin/env python3
"""
Autonomous trading bot — full institutional-grade analysis.
6-member committee + live news + options sentiment + sector rotation +
trailing stops + take-profit + Fear & Greed + MACD + Bollinger + ATR.
Pass 'morning' or 'eod' as sys.argv[1] (default: morning).
"""

import os, json, sys, time
from datetime import date, datetime

import robin_stocks.robinhood as r
import yfinance as yf
import requests
import anthropic

# ── Config ────────────────────────────────────────────────────────────────
ROBINHOOD_EMAIL    = os.environ['ROBINHOOD_EMAIL']
ROBINHOOD_PASSWORD = os.environ['ROBINHOOD_PASSWORD']
ROBINHOOD_SESSION  = os.environ['ROBINHOOD_SESSION']
SLACK_WEBHOOK_URL  = os.environ['SLACK_WEBHOOK_URL']
ANTHROPIC_API_KEY  = os.environ['ANTHROPIC_API_KEY']

ACCOUNT_NUMBER = '456166776'
SLACK_MENTION  = '<@U0B8ZNEB9N2>'
STOP_PCT       = 0.12

CASH_RESERVE          = 0.20
TRAIL_BREAKEVEN_AT    = 0.08   # trail stop to +0.5% when up 8%
TRAIL_PROFIT_AT       = 0.15   # trail stop to +5% when up 15%
TAKE_PROFIT_AT        = 0.20   # full auto-sell when up 20%

# 55-stock high-momentum universe
CANDIDATES = list(dict.fromkeys([
    # AI / semiconductors
    'NVDA','AMD','ARM','AVGO','TSM','SMCI','QCOM','MU','AMAT',
    # Cybersecurity
    'CRWD','PANW','ZS','FTNT',
    # Cloud / SaaS
    'DDOG','SNOW','NOW','CRM','GTLB','MNDY','BILL','DOCN',
    # AI / quantum / space
    'PLTR','IONQ','SOUN','CRCL','RKLB',
    # Mega-cap tech
    'META','GOOGL','MSFT','AAPL','AMZN','TSLA','NFLX',
    # Fintech / crypto
    'COIN','MSTR','MARA','HOOD','SOFI','AFRM','UPST','SQ',
    # Consumer / social
    'SHOP','RBLX','RDDT','PINS','SNAP','TTD','ROKU','UBER',
    # Health / consumer growth
    'HIMS','CELH','DUOL',
    # Other
    'APP','XOM','MSTR',
]))

SECTOR_ETFS = {
    'Technology':    'XLK',
    'Healthcare':    'XLV',
    'Financials':    'XLF',
    'Energy':        'XLE',
    'Consumer Disc': 'XLY',
    'Industrials':   'XLI',
    'Communication': 'XLC',
    'Materials':     'XLB',
}

ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Technical indicators ──────────────────────────────────────────────────

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period-1) + gains[i]) / period
        al = (al * (period-1) + losses[i]) / period
    return 100 - 100 / (1 + ag / al) if al else 100.0


def calc_macd(closes, fast=12, slow=26, signal=9):
    def ema(data, p):
        k = 2 / (p + 1)
        e = data[0]
        out = [e]
        for x in data[1:]:
            e = x * k + e * (1 - k)
            out.append(e)
        return out
    if len(closes) < slow + signal:
        return {'macd': 0, 'signal_line': 0, 'histogram': 0, 'bullish_cross': False, 'above_zero': False}
    ef   = ema(closes, fast)
    es   = ema(closes, slow)
    ml   = [f - s for f, s in zip(ef[slow-1:], es[slow-1:])]
    sl   = ema(ml, signal)
    hist = ml[-1] - sl[-1]
    prev = ml[-2] - sl[-2] if len(ml) > 1 else hist
    return {
        'macd':         round(ml[-1], 4),
        'signal_line':  round(sl[-1], 4),
        'histogram':    round(hist, 4),
        'bullish_cross': hist > 0 and prev <= 0,
        'above_zero':   ml[-1] > 0,
    }


def calc_atr(hist_df, period=14):
    try:
        closes = list(hist_df['Close'])
        highs  = list(hist_df['High'])
        lows   = list(hist_df['Low'])
        trs    = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
                  for i in range(1, len(closes))]
        return sum(trs[-period:]) / min(period, len(trs)) if trs else 0
    except:
        return 0


def calc_bollinger(closes, period=20, devs=2):
    if len(closes) < period:
        return {'pct_b': 0.5, 'squeeze': False, 'above_mid': True, 'upper': 0, 'lower': 0}
    sl  = closes[-period:]
    ma  = sum(sl) / period
    std = (sum((x-ma)**2 for x in sl) / period) ** 0.5
    upper = ma + devs * std
    lower = ma - devs * std
    price = closes[-1]
    pct_b = (price - lower) / (upper - lower) if upper != lower else 0.5
    return {
        'upper':     round(upper, 2),
        'lower':     round(lower, 2),
        'pct_b':     round(pct_b, 2),
        'squeeze':   (upper - lower) / ma < 0.05 if ma else False,
        'above_mid': price > ma,
    }


# ── Market data ───────────────────────────────────────────────────────────

def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period='3mo')
        if hist.empty or len(hist) < 20:
            return None
        closes    = list(hist['Close'])
        volumes   = list(hist['Volume'])
        price     = closes[-1]
        prev      = closes[-2]
        n50       = min(50, len(closes))
        ma20      = sum(closes[-20:]) / 20
        ma50      = sum(closes[-n50:]) / n50
        avg_vol   = sum(volumes[-20:]) / 20
        atr       = calc_atr(hist)
        try:
            pre_mkt = getattr(ticker.fast_info, 'pre_market_price', None)
        except:
            pre_mkt = None
        return {
            'symbol':        symbol,
            'price':         round(price, 2),
            'day_chg':       round((price - prev) / prev * 100, 2),
            'ma20':          round(ma20, 2),
            'ma50':          round(ma50, 2),
            'above_ma20':    price > ma20,
            'above_ma50':    price > ma50,
            'rsi':           round(calc_rsi(closes[-30:] if len(closes) >= 30 else closes), 1),
            'macd':          calc_macd(closes),
            'bollinger':     calc_bollinger(closes),
            'atr':           round(atr, 2),
            'atr_pct':       round(atr / price * 100, 2) if price else 0,
            'vol_ratio':     round(volumes[-1] / avg_vol if avg_vol else 1.0, 2),
            'pct_from_high': round((price - max(hist['High'])) / max(hist['High']) * 100, 1),
            'pct_from_low':  round((price - min(hist['Low']))  / min(hist['Low'])  * 100, 1),
            'pre_market':    round(pre_mkt, 2) if pre_mkt else None,
        }
    except Exception as e:
        print(f"  Data error {symbol}: {e}")
        return None


def get_deep_data(symbol, price):
    """News, options put/call, short interest, analyst target."""
    out = {'news': [], 'pc_ratio': None, 'short_float': None,
           'analyst_target': None, 'analyst_upside': None}
    try:
        ticker = yf.Ticker(symbol)
        # News (last 72h)
        try:
            cutoff = datetime.now().timestamp() - 72 * 3600
            out['news'] = [n['title'] for n in (ticker.news or [])
                           if n.get('providerPublishTime', 0) > cutoff][:6]
        except:
            pass
        # Options put/call ratio
        try:
            exps = ticker.options
            if exps:
                chain   = ticker.option_chain(exps[0])
                call_oi = chain.calls['openInterest'].sum()
                put_oi  = chain.puts['openInterest'].sum()
                if call_oi > 0:
                    out['pc_ratio'] = round(put_oi / call_oi, 2)
        except:
            pass
        # Short interest + analyst target
        try:
            info = ticker.info
            sf = info.get('shortPercentOfFloat')
            if sf:
                out['short_float'] = round(sf * 100, 1)
            tgt = info.get('targetMeanPrice')
            if tgt and price:
                out['analyst_target'] = round(tgt, 2)
                out['analyst_upside']  = round((tgt - price) / price * 100, 1)
        except:
            pass
    except Exception as e:
        print(f"  Deep data error {symbol}: {e}")
    return out


def get_sector_performance(spy_chg):
    perf = {}
    for sector, etf in SECTOR_ETFS.items():
        d = get_stock_data(etf)
        if d:
            perf[sector] = {'chg': d['day_chg'], 'rs': round(d['day_chg'] - spy_chg, 2), 'etf': etf}
        time.sleep(0.1)
    return perf


def days_to_earnings(symbol):
    try:
        ticker = yf.Ticker(symbol)
        cal    = ticker.calendar
        if cal is None:
            return None
        if hasattr(cal, 'columns'):
            ed = cal['Earnings Date'].iloc[0] if 'Earnings Date' in cal.columns else None
        elif isinstance(cal, dict):
            dates = cal.get('Earnings Date', [])
            ed    = dates[0] if dates else None
        else:
            return None
        if ed is None:
            return None
        if hasattr(ed, 'date'):
            ed = ed.date()
        elif isinstance(ed, str):
            ed = datetime.strptime(ed[:10], '%Y-%m-%d').date()
        days = (ed - date.today()).days
        return days if days >= 0 else None
    except:
        return None


# ── VCP score (6 criteria) ────────────────────────────────────────────────

def vcp_score(d, spy_chg):
    rs = d['day_chg'] - spy_chg
    return sum([
        1 if d['pct_from_high'] > -15 else 0,
        1 if d['above_ma20']          else 0,
        1 if d['above_ma50']          else 0,
        1 if 50 <= d['rsi'] <= 80     else 0,
        1 if d['vol_ratio'] > 1.1     else 0,
        1 if rs > 0                   else 0,
    ])


# ── Fear & Greed composite ────────────────────────────────────────────────

def calc_fear_greed(vix, spy_chg, qqq_chg, sector_perf):
    score = 50
    # VIX
    score += {True: 15, False: 0}[vix < 12] or \
             ({True: 8}[vix < 16] if vix < 16 else \
             ({True: 3}[vix < 20] if vix < 20 else \
             ({True: -5}[vix < 25] if vix < 25 else \
             ({True: -15}[vix < 30] if vix < 30 else -25))))
    # Market momentum
    score += 10 if spy_chg > 1.0 else (5 if spy_chg > 0.3 else (-10 if spy_chg < -1.0 else (-5 if spy_chg < -0.3 else 0)))
    # Tech leadership (risk-on signal)
    tech_rs = qqq_chg - spy_chg
    score += 5 if tech_rs > 0.5 else (-5 if tech_rs < -0.5 else 0)
    # Sector breadth
    if sector_perf:
        pos = sum(1 for s in sector_perf.values() if s['chg'] > 0)
        score += (pos / len(sector_perf) - 0.5) * 20
    return max(0, min(100, round(score)))


def fg_label(score):
    if score <= 25:  return f"😱 Extreme Fear ({score}/100)"
    if score <= 45:  return f"😨 Fear ({score}/100)"
    if score <= 55:  return f"😐 Neutral ({score}/100)"
    if score <= 75:  return f"😎 Greed ({score}/100)"
    return               f"🤑 Extreme Greed ({score}/100)"


# ── Full committee analysis ───────────────────────────────────────────────

def run_committee(symbol, data, deep, spy_chg, qqq_chg, vix, fg_score,
                  sector_perf, account_equity, held):
    rs       = data['day_chg'] - spy_chg
    vcp      = vcp_score(data, spy_chg)
    dte      = days_to_earnings(symbol)
    pos_ctx  = ', '.join([f"{s} ({h['pnl']:+.1f}%)" for s, h in held.items()]) or 'None'
    macd     = data['macd']
    boll     = data['bollinger']

    news_str = '\n'.join([f"  • {h}" for h in deep['news']]) if deep['news'] else '  No recent news'
    top_sectors = sorted(sector_perf.items(), key=lambda x: x[1]['rs'], reverse=True)[:4]
    sector_str  = ' | '.join([f"{s}: {v['chg']:+.1f}% (RS {v['rs']:+.1f}%)" for s, v in top_sectors])

    pc_str      = (f"{deep['pc_ratio']} — {'⚠️ bearish hedging' if deep['pc_ratio'] and deep['pc_ratio'] > 1.2 else '✅ bullish'}"
                   if deep['pc_ratio'] else 'N/A')
    short_str   = f"{deep['short_float']}% of float {'🔥 HIGH — squeeze risk' if deep['short_float'] and deep['short_float'] > 15 else ''}" if deep['short_float'] else 'N/A'
    analyst_str = f"${deep['analyst_target']} ({deep['analyst_upside']:+.1f}% upside)" if deep['analyst_target'] else 'N/A'
    pre_str     = (f"${data['pre_market']} ({(data['pre_market']-data['price'])/data['price']*100:+.1f}%)"
                   if data['pre_market'] else 'N/A')

    earnings_note = ''
    if dte is not None:
        earnings_note = (f'🚨 EARNINGS IN {dte} DAYS — binary risk, committee must weigh this heavily' if dte <= 2
                        else f'⚠️ Earnings in {dte} days — must factor into conviction and sizing')

    atr_stop = round(data['price'] - 1.5 * data['atr'], 2)
    regime   = ('STRONG RISK-ON' if spy_chg > 1.0 and vix < 18 else
                'RISK-ON'        if spy_chg > 0.3 and vix < 22 else
                'RISK-OFF'       if spy_chg < -0.5 or vix > 28 else 'NEUTRAL/MIXED')

    prompt = f"""You are a six-member institutional investment committee analyzing a real swing trade with real money at stake. Operate with full rigor — as if billions of dollars are at risk. Prioritize truth and probability accuracy. Genuine disagreement between members is required where warranted. Do not reach false consensus.

══════════════════════════════════════
MARKET ENVIRONMENT
══════════════════════════════════════
Regime: {regime} | Fear & Greed: {fg_label(fg_score)}
SPY: {spy_chg:+.2f}% | QQQ: {qqq_chg:+.2f}% | VIX: {vix:.1f}
Leading sectors today: {sector_str}

══════════════════════════════════════
CANDIDATE: {symbol}
══════════════════════════════════════
Price: ${data['price']} | Day change: {data['day_chg']:+.2f}% | RS vs SPY: {rs:+.2f}%
Pre-market: {pre_str}

TECHNICALS:
  RSI(14): {data['rsi']} | MA20: ${data['ma20']} | MA50: ${data['ma50']}
  Above MA20: {data['above_ma20']} | Above MA50: {data['above_ma50']}
  MACD: {macd['macd']} / Signal: {macd['signal_line']} | Above zero: {macd['above_zero']} | Fresh bullish cross: {macd['bullish_cross']}
  Bollinger %B: {boll['pct_b']} (0=lower band, 1=upper) | Squeeze forming: {boll['squeeze']}
  Volume: {data['vol_ratio']}x avg | ATR: ${data['atr']} ({data['atr_pct']}% of price) | ATR-based stop: ${atr_stop}
  Distance from 52w high: {data['pct_from_high']}% | From 52w low: +{data['pct_from_low']}%
  VCP/CANSLIM score: {vcp}/6

SENTIMENT DATA:
  Options put/call ratio: {pc_str}
  Short interest: {short_str}
  Analyst consensus target: {analyst_str}

RECENT NEWS (last 72 hours):
{news_str}

{earnings_note}

ACCOUNT CONTEXT:
  Total equity: ${account_equity:.2f} | Open positions: {pos_ctx}
  Parameters: 12% hard stop | 20% take-profit target | R/R ≥ 1.5:1
  Trailing stops: breakeven at +8%, +5% floor at +15%
  Time horizon: 1-4 week swing | Risk: Aggressive, maximum ROI

══════════════════════════════════════
COMMITTEE — each member analyzes independently, then reassesses after the Devil's Advocate:
══════════════════════════════════════

MEMBER 1 — FUNDAMENTAL ANALYST
Assess {symbol}: revenue growth quality, margin trajectory, competitive moat, balance sheet strength, and valuation vs. peers and history. Does the fundamental picture justify chasing this setup?

MEMBER 2 — MACRO STRATEGIST
Factor in today's specific macro backdrop — Fed posture, rates, sector rotation (use the sector data above), dollar, and global risk. Is the macro wind at {symbol}'s back or in its face right now?

MEMBER 3 — TECHNICAL & FLOW ANALYST
Evaluate ALL signals: trend structure, RSI momentum, MACD cross and position, Bollinger %B, volume confirmation, ATR volatility, pre-market action, and distance from 52w high. Is this a clean, high-quality technical entry today?

MEMBER 4 — DEVIL'S ADVOCATE (be aggressive — this is your only job)
Attack this thesis hard and specifically. Is the move already over? Does the put/call ratio suggest smart money hedging? Is the news already priced in? What is the precise catalyst that makes this trade fail? Give a specific failure scenario with realistic probability.

MEMBER 5 — INNOVATION & INDUSTRY SPECIALIST
Assess {symbol}'s sector positioning given today's rotation. Is the innovation tailwind secular or cyclical? Is disruption potential real and under-appreciated, or fully priced in at current multiples?

MEMBER 6 — PROBABILITY & RISK MANAGER
Run explicit probability-weighted scenario analysis:
  Bull case (X% probability): price target and what has to go right
  Base case (X% probability): realistic outcome
  Bear case (X% probability): how bad and what triggers it
Probabilities must sum to 100. Calculate expected value. State clearly whether EV is positive.

After all 6 members share views, they reassess after hearing the Devil's Advocate's attack. Does the thesis survive scrutiny?

══════════════════════════════════════
RESPOND IN THIS EXACT JSON — no markdown, no commentary outside the JSON:
══════════════════════════════════════
{{
  "fundamental":    {{"verdict": "BULL|BEAR|NEUTRAL", "analysis": "<2 sentences specific to {symbol}>", "score": <1-10>}},
  "macro":          {{"verdict": "BULL|BEAR|NEUTRAL", "analysis": "<2 sentences>", "score": <1-10>}},
  "technical":      {{"verdict": "BULL|BEAR|NEUTRAL", "analysis": "<2 sentences covering MACD, RSI, volume>", "score": <1-10>}},
  "devil_advocate": {{"main_attack": "<2 hard-hitting sentences>", "failure_scenario": "<specific scenario>", "failure_probability": "<X%>"}},
  "innovation":     {{"verdict": "BULL|BEAR|NEUTRAL", "analysis": "<2 sentences>", "score": <1-10>}},
  "risk_manager":   {{"ev": "POSITIVE|NEGATIVE|NEUTRAL", "bull_prob": <int>, "base_prob": <int>, "bear_prob": <int>, "bull_target": "<price>", "bear_target": "<price>", "analysis": "<2 sentences>"}},
  "reassessment":   "<1-2 sentences: what changed or was reinforced after Devil's Advocate>",
  "conviction":     <final 1-10 integer after full debate>,
  "action":         "BUY" or "SKIP",
  "final_thesis":   "<3 sentences: why enter now, what makes this setup exceptional, primary catalyst>",
  "primary_risk":   "<single most important risk that could invalidate this trade>"
}}"""

    try:
        resp = ai_client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1100,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = resp.content[0].text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        result = json.loads(text)
        result['earnings_days'] = dte
        return result
    except Exception as e:
        print(f"  Committee error {symbol}: {e}")
        return None


# ── Position management with trailing stops + take-profit ─────────────────

def load_positions(buying_power):
    held         = {}
    stops_hit    = []
    targets_hit  = []
    trailing_log = []

    raw = r.account.get_open_stock_positions(account_number=ACCOUNT_NUMBER) or []
    for pos in raw:
        try:
            instr = r.stocks.get_instrument_by_url(pos['instrument'])
            sym   = instr['symbol']
            qty   = float(pos['quantity'])
            cost  = float(pos['average_buy_price'])
            price = float((r.stocks.get_latest_price(sym) or [cost])[0])
            pnl   = (price - cost) / cost * 100

            if pnl >= TAKE_PROFIT_AT * 100:
                print(f"  TAKE PROFIT: {sym} +{pnl:.1f}%")
                r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                targets_hit.append({'symbol': sym, 'price': price, 'pnl': pnl})
                buying_power += qty * price
                continue

            if pnl >= TRAIL_PROFIT_AT * 100:
                effective_stop = round(cost * 1.05, 2)
                trailing_log.append(f"{sym} stop → +5% floor (${effective_stop})")
                if price <= effective_stop:
                    print(f"  TRAIL STOP (+5%): {sym}")
                    r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                    stops_hit.append({'symbol': sym, 'price': price, 'pnl': pnl, 'type': 'trail +5%'})
                    buying_power += qty * price
                    continue
                stop = effective_stop

            elif pnl >= TRAIL_BREAKEVEN_AT * 100:
                effective_stop = round(cost * 1.005, 2)
                trailing_log.append(f"{sym} stop → breakeven (${effective_stop})")
                if price <= effective_stop:
                    print(f"  TRAIL STOP (breakeven): {sym}")
                    r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                    stops_hit.append({'symbol': sym, 'price': price, 'pnl': pnl, 'type': 'trail breakeven'})
                    buying_power += qty * price
                    continue
                stop = effective_stop

            else:
                stop = round(cost * (1 - STOP_PCT), 2)
                if price <= stop:
                    print(f"  HARD STOP: {sym} @ ${price:.2f}")
                    r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                    stops_hit.append({'symbol': sym, 'price': price, 'pnl': pnl, 'type': 'hard stop -12%'})
                    buying_power += qty * price
                    continue

            held[sym] = {'qty': qty, 'cost': cost, 'price': price, 'pnl': pnl, 'stop': stop}
        except Exception as e:
            print(f"  Position error: {e}")

    return held, stops_hit, targets_hit, trailing_log, buying_power


# ── Slack ─────────────────────────────────────────────────────────────────

def slack_send(msg):
    try:
        r_ = requests.post(SLACK_WEBHOOK_URL, json={'text': msg}, timeout=15)
        r_.raise_for_status()
        print("Slack sent.")
    except Exception as e:
        print(f"Slack error: {e}")


def vi(verdict):
    return {'BULL': '🟢', 'BEAR': '🔴', 'NEUTRAL': '🟡'}.get(verdict, '⚪')


def build_trade_report(t):
    c  = t['committee']
    rm = c['risk_manager']
    ev_icon = {'POSITIVE': '✅', 'NEGATIVE': '❌', 'NEUTRAL': '⚖️'}.get(rm['ev'], '⚖️')
    lines = [
        f"*{t['symbol']}* — ${t['amount']:.2f} @ ${t['price']} | Target ${t['target']} | Stop ${t['stop']} | Conviction *{c['conviction']}/10*",
    ]
    if t.get('earnings_days') is not None and t['earnings_days'] <= 5:
        lines.append(f"  ⚠️ *Earnings in {t['earnings_days']} days — position halved*")
    lines += [
        f"  {vi(c['fundamental']['verdict'])} *Fundamental ({c['fundamental']['score']}/10):* {c['fundamental']['analysis']}",
        f"  {vi(c['macro']['verdict'])} *Macro ({c['macro']['score']}/10):* {c['macro']['analysis']}",
        f"  {vi(c['technical']['verdict'])} *Technical ({c['technical']['score']}/10):* {c['technical']['analysis']}",
        f"  {vi(c['innovation']['verdict'])} *Innovation ({c['innovation']['score']}/10):* {c['innovation']['analysis']}",
        f"  {ev_icon} *Risk Manager:* Bull {rm['bull_prob']}% → ${rm.get('bull_target','?')} / Bear {rm['bear_prob']}% → ${rm.get('bear_target','?')} | {rm['analysis']}",
        f"  🔴 *Devil's Advocate:* {c['devil_advocate']['main_attack']} _(failure prob: {c['devil_advocate']['failure_probability']})_",
        f"  ↩️ *Post-debate:* {c['reassessment']}",
        f"  📋 *Final thesis:* {c['final_thesis']}",
        f"  ⚠️ *Primary risk:* {c['primary_risk']}",
    ]
    return '\n'.join(lines)


# ── Login ─────────────────────────────────────────────────────────────────

def rh_login():
    import base64
    pickle_dir  = os.path.expanduser('~/.tokens')
    os.makedirs(pickle_dir, exist_ok=True)
    with open(os.path.join(pickle_dir, 'robinhood.pickle'), 'wb') as f:
        f.write(base64.b64decode(ROBINHOOD_SESSION))
    result = r.login(username=ROBINHOOD_EMAIL, password=ROBINHOOD_PASSWORD,
                     store_session=True, expiresIn=3600)
    if not result:
        slack_send(f"{SLACK_MENTION} ⚠️ *Bot login failed* — re-run setup_device_token.py and update ROBINHOOD_SESSION.")
        sys.exit(1)
    print("Robinhood login OK")


def get_buying_power():
    try:
        profile = r.account.load_account_profile(account_number=ACCOUNT_NUMBER)
        bp = float(profile.get('buying_power') or 0)
        if bp > 0:
            return bp
    except:
        pass
    port = r.profiles.load_portfolio_profile(account_number=ACCOUNT_NUMBER)
    return float(port.get('withdrawable_amount') or 0)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    today = date.today().isoformat()
    mode  = sys.argv[1] if len(sys.argv) > 1 else 'morning'
    print(f"[{today}] mode={mode}")

    label = "Morning Analysis" if mode == 'morning' else "EOD Recap"
    slack_send(f"{SLACK_MENTION} :robot_face: *{label} started* — {today} | Analysis running, report incoming...")

    rh_login()
    buying_power = get_buying_power()

    # Get market data first
    spy_data = get_stock_data('SPY')
    qqq_data = get_stock_data('QQQ')
    vix_data = get_stock_data('^VIX')
    spy_chg  = spy_data['day_chg'] if spy_data else 0.0
    qqq_chg  = qqq_data['day_chg'] if qqq_data else 0.0
    vix      = vix_data['price']   if vix_data else 20.0
    print(f"Market: SPY {spy_chg:+.2f}% QQQ {qqq_chg:+.2f}% VIX {vix:.1f}")

    # Sector rotation
    print("Fetching sector data...")
    sector_perf = get_sector_performance(spy_chg)
    fg_score    = calc_fear_greed(vix, spy_chg, qqq_chg, sector_perf)
    print(f"Fear & Greed: {fg_score} | Sectors: {len(sector_perf)} fetched")

    # Positions with full exit management
    held, stops_hit, targets_hit, trailing_log, buying_power = load_positions(buying_power)
    deployable  = buying_power * (1 - 0.20)
    equity      = buying_power + sum(h['price'] * h['qty'] for h in held.values())
    print(f"Cash: ${buying_power:.2f} | Deployable: ${deployable:.2f} | Equity: ${equity:.2f}")

    # ── EOD mode ──────────────────────────────────────────────────────────
    if mode == 'eod':
        lines = [f"{SLACK_MENTION} 📊 *EOD Recap — {today}*\n"]
        lines.append(f"*Market:* SPY {spy_chg:+.2f}% | QQQ {qqq_chg:+.2f}% | VIX {vix:.1f} | {fg_label(fg_score)}\n")
        all_sec = sorted(sector_perf.items(), key=lambda x: x[1]['rs'], reverse=True)
        lines.append('*Sectors:* ' + ' | '.join([f"{s}: {v['chg']:+.1f}%" for s, v in all_sec]))
        lines.append('')
        if held:
            lines.append('*Open positions:*')
            for sym, h in held.items():
                icon = '🟢' if h['pnl'] > 0 else '🔴'
                dollar_pnl = (h['price'] - h['cost']) * h['qty']
                to_stop = (h['price'] - h['stop']) / h['price'] * 100
                lines.append(
                    f"  {icon} *{sym}* | Entry ${h['cost']:.2f} → Now ${h['price']:.2f} | "
                    f"P&L: *{h['pnl']:+.1f}%* (${dollar_pnl:+.2f}) | "
                    f"Stop ${h['stop']:.2f} ({to_stop:.1f}% away)"
                )
        else:
            lines.append('No open positions.')
        if trailing_log:
            lines.append('\n*Trailing stops updated:*')
            for t in trailing_log:
                lines.append(f"  📍 {t}")
        if targets_hit:
            lines.append('\n*Take-profits executed:*')
            for t in targets_hit:
                lines.append(f"  🎯 {t['symbol']} @ ${t['price']:.2f} | P&L: {t['pnl']:+.1f}%")
        if stops_hit:
            lines.append('\n*Stops executed:*')
            for s in stops_hit:
                lines.append(f"  ⚠️ {s['symbol']} @ ${s['price']:.2f} | P&L: {s['pnl']:+.1f}% ({s['type']})")
        lines.append(f"\n*Cash:* ${buying_power:.2f} | *Total equity:* ${equity:.2f}")
        slack_send('\n'.join(lines))
        r.logout()
        return

    # ── Morning mode ──────────────────────────────────────────────────────
    vix_block  = vix > 40
    vix_reduce = vix > 30 and not vix_block

    # Screen candidates
    to_screen = [s for s in CANDIDATES if s not in held]
    print(f"Screening {len(to_screen)} candidates...")
    stock_data = {}
    for sym in to_screen:
        d = get_stock_data(sym)
        if d:
            stock_data[sym] = d
        time.sleep(0.2)

    qualified = sorted(
        [(sym, d) for sym, d in stock_data.items()
         if d['day_chg'] > spy_chg and vcp_score(d, spy_chg) >= 4],
        key=lambda x: x[1]['day_chg'] - spy_chg,
        reverse=True,
    )
    print(f"{len(qualified)} passed screen. Running full committee on top {min(6, len(qualified))}...")

    trades_done = []
    skipped     = []

    if not vix_block and deployable >= 10:
        for sym, data in qualified[:6]:
            print(f"  Deep data + committee: {sym}...")
            deep      = get_deep_data(sym, data['price'])
            committee = run_committee(sym, data, deep, spy_chg, qqq_chg, vix,
                                      fg_score, sector_perf, equity, held)
            if not committee:
                continue

            conviction = committee.get('conviction', 0)
            action     = committee.get('action', 'SKIP')
            dte        = committee.get('earnings_days')
            rs         = data['day_chg'] - spy_chg
            print(f"  {sym}: {conviction}/10 → {action}")

            if dte is not None and dte <= 2:
                print(f"  {sym}: BLOCKED — earnings in {dte} days")
                skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'earnings in {dte} days', 'rs': rs})
                continue

            if conviction >= 8 and action == 'BUY' and deployable >= 10:
                base_pct = {8: 0.15, 9: 0.175, 10: 0.20}.get(min(conviction, 10), 0.15)
                if vix_reduce:
                    base_pct *= 0.70
                if dte is not None and dte <= 5:
                    base_pct *= 0.50
                dollar_amt = round(min(buying_power * base_pct, deployable, equity * 0.25), 2)
                if dollar_amt < 5:
                    skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'insufficient funds', 'rs': rs})
                    continue
                try:
                    r.orders.order_buy_fractional_by_price(
                        sym, dollar_amt, account_number=ACCOUNT_NUMBER, jsonify=True
                    )
                    trades_done.append({
                        'symbol':        sym,
                        'amount':        dollar_amt,
                        'price':         data['price'],
                        'conviction':    conviction,
                        'stop':          round(data['price'] * (1 - STOP_PCT), 2),
                        'target':        round(data['price'] * 1.20, 2),
                        'rs':            rs,
                        'committee':     committee,
                        'earnings_days': dte,
                    })
                    deployable   -= dollar_amt
                    buying_power -= dollar_amt
                    print(f"  ✓ Bought {sym} ${dollar_amt:.2f}")
                except Exception as e:
                    print(f"  Order failed {sym}: {e}")
                    skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'order error: {e}', 'rs': rs})
            else:
                skipped.append({'symbol': sym, 'conviction': conviction,
                                'reason': f'conviction {conviction}/10 — below 8/10 threshold', 'rs': rs})

    # ── Slack message ─────────────────────────────────────────────────────
    lines = [f"{SLACK_MENTION} 📈 *Morning Brief — {today}*\n"]

    lines.append(f"*Market:* SPY {spy_chg:+.2f}% | QQQ {qqq_chg:+.2f}% | VIX {vix:.1f}")
    lines.append(f"*Fear & Greed:* {fg_label(fg_score)}")
    all_sectors = sorted(sector_perf.items(), key=lambda x: x[1]['rs'], reverse=True)
    sector_str = ' | '.join([f"{s}: {v['chg']:+.1f}%" for s, v in all_sectors])
    lines.append(f'*All sectors (vs SPY):* {sector_str}')
    if vix_block:
        lines.append('⛔ *VIX > 40 — no new positions today*')
    elif vix_reduce:
        lines.append('⚠️ VIX > 30 — sizes reduced 30%')
    lines.append('')

    if targets_hit:
        lines.append('*🎯 Take-profits executed:*')
        for t in targets_hit:
            lines.append(f"  {t['symbol']} @ ${t['price']:.2f} | P&L: *{t['pnl']:+.1f}%* ✅")
        lines.append('')

    if stops_hit:
        lines.append('*Auto-exits (stops):*')
        for s in stops_hit:
            lines.append(f"  ⚠️ {s['symbol']} @ ${s['price']:.2f} | P&L: {s['pnl']:+.1f}% ({s['type']})")
        lines.append('')

    if trailing_log:
        lines.append('*Trailing stops updated:*')
        for t in trailing_log:
            lines.append(f"  📍 {t}")
        lines.append('')

    if held:
        lines.append('*Open positions:*')
        for sym, h in held.items():
            icon = '🟢' if h['pnl'] > 0 else '🔴'
            dollar_pnl = (h['price'] - h['cost']) * h['qty']
            to_stop = (h['price'] - h['stop']) / h['price'] * 100
            to_target = (h['cost'] * 1.20 - h['price']) / h['price'] * 100
            lines.append(
                f"  {icon} *{sym}* | Entry ${h['cost']:.2f} → Now ${h['price']:.2f} | "
                f"P&L: *{h['pnl']:+.1f}%* (${dollar_pnl:+.2f}) | "
                f"Stop ${h['stop']:.2f} ({to_stop:.1f}% away) | "
                f"Target ${h['cost']*1.20:.2f} ({to_target:.1f}% away)"
            )
        lines.append('')

    if trades_done:
        lines.append(f'*Committee bought {len(trades_done)} position(s) today:*\n')
        for t in trades_done:
            lines.append(build_trade_report(t))
            lines.append('')
    elif vix_block:
        lines.append('*No trades — VIX above 40.*')
    elif deployable < 10:
        lines.append('*No trades — buying power below minimum.*')
    else:
        lines.append('*No trades — nothing cleared the full committee today.*')
        near = sorted([x for x in skipped if x['conviction'] >= 6],
                      key=lambda x: x['conviction'], reverse=True)[:4]
        if near:
            lines.append('Near misses (reviewed but not cleared):')
            for x in near:
                lines.append(f"  • {x['symbol']}: {x['conviction']}/10 — {x['reason']}")

    lines.append(f"\n*Cash:* ${buying_power:.2f} | *Total equity:* ${equity:.2f}")
    lines.append('_Override: sell directly in Robinhood app_')

    slack_send('\n'.join(lines))
    r.logout()
    print("Done.")


if __name__ == '__main__':
    main()
