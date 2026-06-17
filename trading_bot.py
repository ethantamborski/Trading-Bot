#!/usr/bin/env python3
"""
Autonomous trading bot — full institutional-grade analysis.
6-member committee + live news + options sentiment + sector rotation +
trailing stops + take-profit + Fear & Greed + MACD + Bollinger + ATR.
Pass 'morning', 'midday', or 'eod' as sys.argv[1] (default: morning).
'morning' and 'midday' both run full buy logic; 'midday' only differs in labels.
"""

import os, json, sys, time, math
from datetime import date, datetime

import robin_stocks.robinhood as r
import yfinance as yf
import requests
import anthropic
import gspread
from google.oauth2.service_account import Credentials

# ── Config ────────────────────────────────────────────────────────────────
ROBINHOOD_EMAIL    = os.environ['ROBINHOOD_EMAIL']
ROBINHOOD_PASSWORD = os.environ['ROBINHOOD_PASSWORD']
ROBINHOOD_SESSION  = os.environ['ROBINHOOD_SESSION']
SLACK_WEBHOOK_URL  = os.environ['SLACK_WEBHOOK_URL']
ANTHROPIC_API_KEY  = os.environ['ANTHROPIC_API_KEY']
GOOGLE_SHEET_ID    = os.environ.get('GOOGLE_SHEET_ID', '')
GCLOUD_KEY_PATH    = '/opt/trading-bot/gcloud.json'
SHEET_URL          = f'https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}'
SESSION_DATE       = os.environ.get('SESSION_DATE', '')  # YYYY-MM-DD when session was last created

ACCOUNT_NUMBER = '456166776'
SLACK_MENTION  = '<@U0B8ZNEB9N2>'
STOP_PCT       = 0.12

CASH_RESERVE          = 0.10
TRAIL_BREAKEVEN_AT    = 0.08   # trail stop to +0.5% when up 8%
TRAIL_PROFIT_AT       = 0.15   # trail stop to +5% when up 15%
TAKE_PROFIT_AT        = 0.20   # full auto-sell when up 20%

MIN_CONVICTION        = 6      # hard floor — never buy below this
ROTATE_INTO_CONVICTION = 8     # only rotate capital into 8/10+ ideas
ROTATE_LAGGARD_MAX_PNL = 3.0   # only rotate OUT of positions flat/red (< +3%)

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
    'COIN','MSTR','MARA','HOOD','SOFI','AFRM','UPST',
    # Consumer / social
    'SHOP','RBLX','RDDT','PINS','SNAP','TTD','ROKU','UBER','XYZ',
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
        hist      = hist.dropna(subset=['Close'])
        closes    = list(hist['Close'])
        volumes   = list(hist['Volume'].fillna(0))
        price     = float(closes[-1])
        prev      = float(closes[-2])
        if prev <= 0 or math.isnan(price) or math.isnan(prev):
            return None
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

    prompt = f"""You are a six-member investment committee analyzing an aggressive swing trade. Prioritize truth and probability accuracy. Genuine disagreement is required where warranted. Do not reach false consensus.

CONVICTION SCALE — use this calibration exactly:
  3-4: Poor setup. Multiple red flags, weak technicals, macro headwind. Clear SKIP.
  5-6: Below average. Some merit but risk/reward insufficient for this account. SKIP.
  7:   Good setup. Clean technicals, positive EV, at least one strong catalyst. BUY with standard sizing.
  8:   Strong setup. Multiple tailwinds aligning, clear breakout structure, above-average RS. BUY with larger sizing.
  9:   Exceptional. Everything lines up — catalyst, technicals, macro, volume. Rare. BUY aggressively.
  10:  Generational entry. Reserved for extraordinary setups only.
A legitimate BUY with 12% stop and 20% target on an aggressive account SHOULD score 7+. Do not anchor to institutional caution — this is an aggressive swing trading account.

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
            max_tokens=2000,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = resp.content[0].text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        import re as _re
        text = _re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', ' ', text)
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
            if cost <= 0 or qty <= 0:
                print(f"  Skipping {sym}: zero cost/qty from Robinhood")
                continue
            price = float((r.stocks.get_latest_price(sym) or [cost])[0])
            price = price if price > 0 else cost
            pnl   = (price - cost) / cost * 100

            if pnl >= TAKE_PROFIT_AT * 100:
                print(f"  TAKE PROFIT: {sym} +{pnl:.1f}%")
                r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                targets_hit.append({'symbol': sym, 'price': price, 'pnl': pnl, 'cost': cost, 'qty': qty})
                buying_power += qty * price
                continue

            if pnl >= TRAIL_PROFIT_AT * 100:
                effective_stop = round(cost * 1.05, 2)
                trailing_log.append(f"{sym} stop → +5% floor (${effective_stop})")
                if price <= effective_stop:
                    print(f"  TRAIL STOP (+5%): {sym}")
                    r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                    stops_hit.append({'symbol': sym, 'price': price, 'pnl': pnl, 'type': 'trail +5%', 'cost': cost, 'qty': qty})
                    buying_power += qty * price
                    continue
                stop = effective_stop

            elif pnl >= TRAIL_BREAKEVEN_AT * 100:
                effective_stop = round(cost * 1.005, 2)
                trailing_log.append(f"{sym} stop → breakeven (${effective_stop})")
                if price <= effective_stop:
                    print(f"  TRAIL STOP (breakeven): {sym}")
                    r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                    stops_hit.append({'symbol': sym, 'price': price, 'pnl': pnl, 'type': 'trail breakeven', 'cost': cost, 'qty': qty})
                    buying_power += qty * price
                    continue
                stop = effective_stop

            else:
                stop = round(cost * (1 - STOP_PCT), 2)
                if price <= stop:
                    print(f"  HARD STOP: {sym} @ ${price:.2f}")
                    r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                    stops_hit.append({'symbol': sym, 'price': price, 'pnl': pnl, 'type': 'hard stop -12%', 'cost': cost, 'qty': qty})
                    buying_power += qty * price
                    continue

            held[sym] = {'qty': qty, 'cost': cost, 'price': price if price > 0 else cost, 'pnl': pnl, 'stop': stop}
        except Exception as e:
            print(f"  Position error: {e}")

    return held, stops_hit, targets_hit, trailing_log, buying_power


# ── Sheets formatting helpers ─────────────────────────────────────────────

_C = {
    'blk':   '#080E0A', 'dkgrn': '#0C2A17', 'grn':   '#14532D',
    'mgrn':  '#1A7A40', 'lgrn':  '#A8D8B9', 'mint':  '#E8F5EE',
    'xmint': '#F2FAF6', 'white': '#FFFFFF',  'lgray': '#F4F6F4',
    'mgray': '#5A7A62', 'dgray': '#162219',  'border':'#B8D4C0',
    'pos':   '#0F7B3F', 'lpos':  '#D5F5E3',  'neg':   '#C0392B',
    'lneg':  '#FADBD8',
}


def _rgb(h):
    h = h.lstrip('#')
    return {'red': int(h[0:2],16)/255, 'green': int(h[2:4],16)/255, 'blue': int(h[4:6],16)/255}


def _rng(sid, r1, r2, c1, c2):
    return {'sheetId': sid, 'startRowIndex': r1, 'endRowIndex': r2,
            'startColumnIndex': c1, 'endColumnIndex': c2}


def _cell(sid, r1, r2, c1, c2, **kw):
    fmt, fields = {}, []
    if 'bg' in kw:
        fmt['backgroundColor'] = _rgb(kw['bg']); fields.append('backgroundColor')
    tf = {}
    if 'fg' in kw:   tf['foregroundColor'] = _rgb(kw['fg'])
    if kw.get('bold'):  tf['bold'] = True
    if kw.get('italic'): tf['italic'] = True
    if 'size' in kw: tf['fontSize'] = kw['size']
    if tf: fmt['textFormat'] = tf; fields.append('textFormat')
    if 'align' in kw:  fmt['horizontalAlignment'] = kw['align']; fields.append('horizontalAlignment')
    if 'valign' in kw: fmt['verticalAlignment']   = kw['valign']; fields.append('verticalAlignment')
    if 'wrap' in kw:   fmt['wrapStrategy'] = kw['wrap']; fields.append('wrapStrategy')
    return {'repeatCell': {
        'range': _rng(sid, r1, r2, c1, c2),
        'cell': {'userEnteredFormat': fmt},
        'fields': 'userEnteredFormat(' + ','.join(fields) + ')'
    }}


def _colw(sid, c, px):
    return {'updateDimensionProperties': {
        'range': {'sheetId': sid, 'dimension': 'COLUMNS', 'startIndex': c, 'endIndex': c+1},
        'properties': {'pixelSize': px}, 'fields': 'pixelSize'
    }}


def _rowh(sid, r, px):
    return {'updateDimensionProperties': {
        'range': {'sheetId': sid, 'dimension': 'ROWS', 'startIndex': r, 'endIndex': r+1},
        'properties': {'pixelSize': px}, 'fields': 'pixelSize'
    }}


def _freeze(sid, rows=1):
    return {'updateSheetProperties': {
        'properties': {'sheetId': sid, 'gridProperties': {'frozenRowCount': rows}},
        'fields': 'gridProperties.frozenRowCount'
    }}


def _borders(sid, r1, r2, c1, c2, color='#D5D8DC'):
    b = {'style': 'SOLID', 'colorStyle': {'rgbColor': _rgb(color)}}
    return {'updateBorders': {
        'range': _rng(sid, r1, r2, c1, c2),
        'top': b, 'bottom': b, 'left': b, 'right': b,
        'innerHorizontal': b, 'innerVertical': b
    }}


def _cond(sid, r1, r2, c1, c2, ctype, val, bg, fg=None, bold=False):
    fmt = {'backgroundColor': _rgb(bg)}
    if fg or bold:
        fmt['textFormat'] = {}
        if fg:   fmt['textFormat']['foregroundColor'] = _rgb(fg)
        if bold: fmt['textFormat']['bold'] = True
    return {'addConditionalFormatRule': {
        'rule': {
            'ranges': [_rng(sid, r1, r2, c1, c2)],
            'booleanRule': {
                'condition': {'type': ctype, 'values': [{'userEnteredValue': val}]},
                'format': fmt
            }
        }, 'index': 0
    }}


def _del_charts(sh, sheet_id):
    try:
        meta = sh.fetch_sheet_metadata()
        for s in meta.get('sheets', []):
            if s['properties']['sheetId'] == sheet_id:
                cids = [c['chartId'] for c in s.get('charts', [])]
                if cids:
                    sh.batch_update({'requests': [
                        {'deleteEmbeddedObject': {'objectId': cid}} for cid in cids
                    ]})
    except Exception as e:
        print(f"  Chart cleanup: {e}")


def _line_chart(sid, anchor_row, anchor_col, title, x_col, y_col, y_label, w=620, h=320):
    """Single-series line chart."""
    return _multi_line_chart(sid, anchor_row, anchor_col, title, x_col,
                             [y_col], [y_label], w, h)


def _multi_line_chart(sid, anchor_row, anchor_col, title, x_col, y_cols, y_labels, w=680, h=340):
    """Multi-series line chart."""
    series = [
        {'series': {'sourceRange': {'sources': [_rng(sid, 0, 2000, c, c+1)]}},
         'targetAxis': 'LEFT_AXIS'}
        for c in y_cols
    ]
    return {'addChart': {'chart': {
        'spec': {
            'title': title,
            'titleTextFormat': {'bold': True, 'fontSize': 11, 'fontFamily': 'Arial'},
            'basicChart': {
                'chartType': 'LINE',
                'legendPosition': 'BOTTOM_LEGEND',
                'axis': [
                    {'position': 'BOTTOM_AXIS', 'title': 'Date'},
                    {'position': 'LEFT_AXIS',   'title': y_labels[0] if len(y_labels) == 1 else ''},
                ],
                'domains': [{'domain': {'sourceRange': {'sources': [_rng(sid, 0, 2000, x_col, x_col+1)]}}}],
                'series': series,
                'headerCount': 1,
                'lineSmoothing': True,
            },
            'fontName': 'Arial',
            'backgroundColor': _rgb('#FFFFFF'),
        },
        'position': {'overlayPosition': {
            'anchorCell': {'sheetId': sid, 'rowIndex': anchor_row, 'columnIndex': anchor_col},
            'widthPixels': w, 'heightPixels': h
        }}
    }}}


def _bar_chart(sid, anchor_row, anchor_col, title, x_col, y_col, w=600, h=300):
    """Horizontal bar chart (symbol vs value)."""
    return {'addChart': {'chart': {
        'spec': {
            'title': title,
            'titleTextFormat': {'bold': True, 'fontSize': 11, 'fontFamily': 'Arial'},
            'basicChart': {
                'chartType': 'BAR',
                'legendPosition': 'NO_LEGEND',
                'axis': [
                    {'position': 'BOTTOM_AXIS', 'title': 'P&L ($)'},
                    {'position': 'LEFT_AXIS',   'title': 'Trade'},
                ],
                'domains': [{'domain': {'sourceRange': {'sources': [_rng(sid, 0, 2000, x_col, x_col+1)]}}}],
                'series': [{'series': {'sourceRange': {'sources': [_rng(sid, 0, 2000, y_col, y_col+1)]}},
                            'targetAxis': 'BOTTOM_AXIS'}],
                'headerCount': 1,
            },
            'fontName': 'Arial',
            'backgroundColor': _rgb('#FFFFFF'),
        },
        'position': {'overlayPosition': {
            'anchorCell': {'sheetId': sid, 'rowIndex': anchor_row, 'columnIndex': anchor_col},
            'widthPixels': w, 'heightPixels': h
        }}
    }}}


def get_prev_equity(sh, today):
    """Return the most recent equity value from a PREVIOUS day (not today)."""
    try:
        ws   = sh.worksheet('Performance')
        vals = ws.get_all_values()
        for row in reversed(vals[1:]):
            if len(row) >= 2 and row[0] != today:
                return float(row[1])
    except Exception:
        pass
    return None


def update_risk_monitor(sh, today, held, equity):
    try:
        try:
            ws = sh.worksheet('Risk Monitor')
            ws.clear()
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title='Risk Monitor', rows=200, cols=13)

        hdrs = ['Symbol','% of Portfolio','Entry','Current','P&L %','P&L $',
                'Stop','To Stop %','Target','To Target %','R/R Ratio','Days Held']
        rows = [hdrs, [f'Last Updated: {today}', '', '', '', '', '', '', '', '', '', '', '']]

        # Try reading Trade Journal for entry dates
        entry_dates = {}
        try:
            tj = sh.worksheet('Trade Journal')
            tj_vals = tj.get_all_values()
            for row in tj_vals[1:]:
                if len(row) >= 2:
                    entry_dates[row[1]] = row[0]  # symbol -> date
        except Exception:
            pass

        if held:
            from datetime import date as _date
            for sym, h in held.items():
                px       = h['price'] if h['price'] > 0 else h['cost']
                pct_port = px * h['qty'] / equity * 100 if equity > 0 else 0
                dpnl     = (px - h['cost']) * h['qty']
                tostop   = (px - h['stop']) / px * 100
                totgt    = (h['cost'] * 1.20 - px) / px * 100
                rr       = abs(totgt / tostop) if tostop != 0 else 0
                # days held
                days_held = '—'
                if sym in entry_dates:
                    try:
                        entry_d = _date.fromisoformat(entry_dates[sym])
                        days_held = str((_date.today() - entry_d).days)
                    except Exception:
                        pass
                rows.append([
                    sym, f'{pct_port:.1f}%', f'${h["cost"]:.2f}', f'${px:.2f}',
                    f'{h["pnl"]:+.1f}%', f'${dpnl:+.2f}',
                    f'${h["stop"]:.2f}', f'{tostop:.1f}%',
                    f'${h["cost"]*1.20:.2f}', f'{totgt:.1f}%',
                    f'{rr:.1f}x', days_held
                ])
        else:
            rows.append(['No open positions', '', '', '', '', '', '', '', '', '', '', ''])

        # Summary block
        max_loss = sum((h['stop'] - h['price']) * h['qty'] for h in held.values()) if held else 0
        at_risk  = sum(h['price'] * h['qty'] for h in held.values()) if held else 0
        rows += [
            [''],
            ['PORTFOLIO RISK', '', '', '', '', '', '', '', '', '', '', ''],
            ['Max loss (all stops)', f'${max_loss:.2f}', '', '', '', '', '', '', '', '', '', ''],
            ['Capital at risk', f'${at_risk:.2f}', '', '', '', '', '', '', '', '', '', ''],
        ]

        ws.update(rows, 'A1')
        print("Risk Monitor updated.")
    except Exception as e:
        print(f"Risk Monitor error: {e}")


def update_market_pulse(sh, today, mode, spy_chg, qqq_chg, iwm_chg, vix, fg_score):
    try:
        hdrs = ['Date','Mode','SPY %','QQQ %','IWM %','VIX','F&G Score','F&G Label']
        ws = ensure_tab(sh, 'Market Pulse', hdrs)
        # Store % and VIX as raw numbers so Sheets can chart them
        ws.append_row([today, mode,
                       round(spy_chg, 2), round(qqq_chg, 2), round(iwm_chg, 2),
                       round(vix, 1), fg_score, fg_label(fg_score)],
                      value_input_option='USER_ENTERED')
        print("Market Pulse appended.")
    except Exception as e:
        print(f"Market Pulse error: {e}")


def update_sector_history(sh, today, sector_perf):
    try:
        SECTOR_ORDER = ['Technology','Healthcare','Financials','Energy',
                        'Consumer Disc','Industrials','Communication','Materials']
        hdrs = ['Date'] + SECTOR_ORDER
        ws = ensure_tab(sh, 'Sector History', hdrs)
        row = [today] + [round(sector_perf.get(s, {}).get('chg', 0), 2) for s in SECTOR_ORDER]
        ws.append_row(row, value_input_option='USER_ENTERED')
        print("Sector History appended.")
    except Exception as e:
        print(f"Sector History error: {e}")


def update_closed_trades(sh, today, stops_hit, targets_hit):
    try:
        if not stops_hit and not targets_hit:
            return
        hdrs = ['Date','Symbol','Exit Type','Entry Price','Exit Price','Qty','P&L %','P&L $']
        ws = ensure_tab(sh, 'Closed Trades', hdrs)
        for t in targets_hit:
            pnl_d = (t['price'] - t['cost']) * t['qty']
            ws.append_row([today, t['symbol'], 'target +20%',
                           round(t['cost'], 2), round(t['price'], 2), round(t['qty'], 4),
                           round(t['pnl'], 2), round(pnl_d, 2)],
                          value_input_option='USER_ENTERED')
        for s in stops_hit:
            pnl_d = (s['price'] - s['cost']) * s['qty']
            ws.append_row([today, s['symbol'], s['type'],
                           round(s['cost'], 2), round(s['price'], 2), round(s['qty'], 4),
                           round(s['pnl'], 2), round(pnl_d, 2)],
                          value_input_option='USER_ENTERED')
        print(f"Closed Trades: {len(stops_hit)+len(targets_hit)} exits logged.")
    except Exception as e:
        print(f"Closed Trades error: {e}")


def update_analytics(sh, today):
    try:
        try:
            ws = sh.worksheet('Analytics')
            ws.clear()
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title='Analytics', rows=100, cols=6)

        # Read Closed Trades
        closed = []
        try:
            ct_vals = sh.worksheet('Closed Trades').get_all_values()
            for row in ct_vals[1:]:
                if len(row) >= 8:
                    try:
                        closed.append({'symbol': row[1], 'exit_type': row[2],
                                       'pnl_pct': float(row[6]), 'pnl_d': float(row[7])})
                    except Exception:
                        pass
        except Exception:
            pass

        # Read Trade Journal for open trades and conviction
        open_count = 0
        avg_conv   = None
        try:
            tj_vals = sh.worksheet('Trade Journal').get_all_values()
            open_count = max(0, len(tj_vals) - 1)
            convs = []
            for row in tj_vals[1:]:
                if len(row) > 5:
                    try:
                        convs.append(float(row[5].split('/')[0]))
                    except Exception:
                        pass
            if convs:
                avg_conv = sum(convs) / len(convs)
        except Exception:
            pass

        wins   = [t for t in closed if t['pnl_d'] > 0]
        losses = [t for t in closed if t['pnl_d'] <= 0]
        tc     = len(closed)
        wr     = len(wins) / tc * 100 if tc else None
        aw     = sum(t['pnl_d'] for t in wins)  / len(wins)  if wins   else None
        al     = sum(t['pnl_d'] for t in losses) / len(losses) if losses else None
        pf     = abs(sum(t['pnl_d'] for t in wins)) / abs(sum(t['pnl_d'] for t in losses)) \
                 if wins and losses and sum(t['pnl_d'] for t in losses) != 0 else None
        total_pnl = sum(t['pnl_d'] for t in closed)
        best  = max(closed, key=lambda t: t['pnl_d']) if closed else None
        worst = min(closed, key=lambda t: t['pnl_d']) if closed else None
        tgts  = sum(1 for t in closed if 'target' in t['exit_type'])
        stps  = sum(1 for t in closed if 'stop'   in t['exit_type'])

        NA = '— awaiting data'
        rows = [
            ['  TRADE ANALYTICS', '', f'Updated: {today}'],
            [''],
            ['  PERFORMANCE SUMMARY'],
            ['Total Trades Taken',    str(open_count + tc)],
            ['Currently Open',        str(open_count)],
            ['Closed Trades',         str(tc)],
            ['Win Rate',              f'{wr:.1f}%'         if wr  is not None else NA],
            ['Profit Factor',         f'{pf:.2f}x'         if pf  is not None else NA],
            ['Avg Conviction Score',  f'{avg_conv:.1f}/10' if avg_conv         else NA],
            [''],
            ['  P&L BREAKDOWN'],
            ['Total Closed P&L',      f'${total_pnl:+.2f}' if closed else NA],
            ['Avg Winning Trade',     f'${aw:+.2f}'         if aw   is not None else NA],
            ['Avg Losing Trade',      f'${al:+.2f}'         if al   is not None else NA],
            ['Targets Hit (+20%)',    str(tgts)],
            ['Stops Hit',             str(stps)],
            ['Best Trade',            f"{best['symbol']}  {best['pnl_pct']:+.1f}%"  if best  else NA],
            ['Worst Trade',           f"{worst['symbol']} {worst['pnl_pct']:+.1f}%" if worst else NA],
        ]
        ws.update(rows, 'A1')
        print("Analytics updated.")
    except Exception as e:
        print(f"Analytics error: {e}")


def apply_formatting(sh, n_pos):
    """Apply green/black/white dashboard formatting to all sheets."""
    try:
        meta = sh.fetch_sheet_metadata()
        ids = {s['properties']['title']: s['properties']['sheetId']
               for s in meta.get('sheets', [])}
    except Exception as e:
        print(f"Formatting meta error: {e}"); return

    did   = ids.get('Dashboard')
    dlid  = ids.get('Daily Log')
    tjid  = ids.get('Trade Journal')
    pid   = ids.get('Performance')
    rmid  = ids.get('Risk Monitor')
    mpid  = ids.get('Market Pulse')
    shid  = ids.get('Sector History')
    ctid  = ids.get('Closed Trades')
    anid  = ids.get('Analytics')
    C     = _C
    rq    = []

    # ── Dashboard ─────────────────────────────────────────────────────────
    if did is not None:
        pos_rows = max(1, n_pos)
        # Row layout:
        # 0: title, 1: subtitle, 2: spacer, 3: KEY METRICS header
        # 4: metric labels, 5: metric values, 6: spacer
        # 7: MARKET OVERVIEW header, 8-12: market rows (SPY/QQQ/IWM/VIX/F&G)
        # 13: spacer, 14: SECTOR ROTATION header, 15-22: 8 sector rows
        # 23: spacer, 24: OPEN POSITIONS header, 25 to 25+pos_rows-1: position rows
        # pstart = 26 + pos_rows: PORTFOLIO SUMMARY header, pstart+1 to pstart+3: data
        pstart = 26 + pos_rows

        rq += [
            # Row 0: title banner
            _cell(did, 0, 1, 0, 9, bg=C['blk'], fg=C['white'], bold=True, size=14, valign='MIDDLE'),
            _rowh(did, 0, 44),
            # Row 1: subtitle bar
            _cell(did, 1, 2, 0, 9, bg=C['dkgrn'], fg=C['lgrn'], size=9, valign='MIDDLE'),
            _rowh(did, 1, 20),
            # Row 2: spacer
            _cell(did, 2, 3, 0, 9, bg=C['mint']),
            _rowh(did, 2, 5),
            # Row 3: KEY METRICS header
            _cell(did, 3, 4, 0, 9, bg=C['grn'], fg=C['white'], bold=True, size=10, valign='MIDDLE'),
            _rowh(did, 3, 28),
            # Row 4: metric labels
            _cell(did, 4, 5, 0, 9, bg=C['dkgrn'], fg=C['lgrn'], bold=True, size=9, valign='MIDDLE'),
            _rowh(did, 4, 22),
            # Row 5: metric values
            _cell(did, 5, 6, 0, 9, bg=C['dgray'], fg=C['white'], bold=True, size=12, valign='MIDDLE'),
            _rowh(did, 5, 36),
            # Row 6: spacer
            _cell(did, 6, 7, 0, 9, bg=C['mint']),
            _rowh(did, 6, 5),
            # Row 7: MARKET OVERVIEW header
            _cell(did, 7, 8, 0, 9, bg=C['grn'], fg=C['white'], bold=True, size=10, valign='MIDDLE'),
            _rowh(did, 7, 28),
            # Rows 8-12: SPY, QQQ, IWM, VIX, F&G
            _cell(did, 8, 13, 0, 1, fg=C['dgray'], bold=True),
            _cell(did, 8, 13, 1, 9, fg=C['dgray']),
            _cell(did, 8, 9,  0, 9, bg=C['white']),
            _cell(did, 9, 10, 0, 9, bg=C['xmint']),
            _cell(did, 10, 11, 0, 9, bg=C['white']),
            _cell(did, 11, 12, 0, 9, bg=C['xmint']),
            _cell(did, 12, 13, 0, 9, bg=C['white']),
            _borders(did, 8, 13, 0, 2, C['border']),
            # Row 13: spacer
            _cell(did, 13, 14, 0, 9, bg=C['mint']),
            _rowh(did, 13, 5),
            # Row 14: SECTOR ROTATION header
            _cell(did, 14, 15, 0, 9, bg=C['dkgrn'], fg=C['white'], bold=True, size=10, valign='MIDDLE'),
            _cell(did, 14, 15, 1, 3, fg=C['lgrn'], size=9, align='RIGHT'),
            _rowh(did, 14, 28),
            # Rows 15-22: 8 sectors
            _cell(did, 15, 23, 0, 1, fg=C['dgray'], bold=True),
            _cell(did, 15, 23, 1, 3, fg=C['dgray'], align='RIGHT'),
            _borders(did, 15, 23, 0, 3, C['border']),
            _cond(did, 15, 23, 2, 3, 'TEXT_CONTAINS', '+', C['lpos'], C['pos']),
            _cond(did, 15, 23, 2, 3, 'TEXT_CONTAINS', '-', C['lneg'], C['neg']),
        ]
        for i in range(15, 23):
            rq.append(_cell(did, i, i+1, 0, 9, bg=(C['white'] if i % 2 == 1 else C['xmint'])))

        rq += [
            # Row 23: spacer
            _cell(did, 23, 24, 0, 9, bg=C['mint']),
            _rowh(did, 23, 5),
            # Row 24: OPEN POSITIONS header
            _cell(did, 24, 25, 0, 9, bg=C['blk'], fg=C['white'], bold=True, size=10, valign='MIDDLE'),
            _cell(did, 24, 25, 1, 9, fg=C['lgrn'], size=9, align='CENTER'),
            _rowh(did, 24, 28),
            # Position data rows 25 to 25+pos_rows-1
            _cell(did, 25, 25+pos_rows, 0, 1, fg=C['dgray'], bold=True),
            _cell(did, 25, 25+pos_rows, 1, 9, fg=C['dgray'], align='CENTER'),
            _borders(did, 25, 25+pos_rows, 0, 9, C['border']),
            _cond(did, 25, 25+pos_rows, 3, 5, 'TEXT_CONTAINS', '+', C['lpos'], C['pos'], True),
            _cond(did, 25, 25+pos_rows, 3, 5, 'TEXT_CONTAINS', '-', C['lneg'], C['neg'], True),
        ]
        for i in range(25, 25 + pos_rows):
            rq.append(_cell(did, i, i+1, 0, 9, bg=(C['white'] if i % 2 == 1 else C['xmint'])))

        rq += [
            # Spacer after positions
            _cell(did, 25+pos_rows, 26+pos_rows, 0, 9, bg=C['mint']),
            _rowh(did, 25+pos_rows, 5),
            # Portfolio summary header
            _cell(did, pstart, pstart+1, 0, 9, bg=C['grn'], fg=C['white'], bold=True, size=10),
            _rowh(did, pstart, 28),
            _cell(did, pstart+1, pstart+4, 0, 1, fg=C['dgray'], bold=True),
            _cell(did, pstart+1, pstart+4, 1, 9, fg=C['dgray'], align='RIGHT'),
            _borders(did, pstart+1, pstart+4, 0, 2, C['border']),
        ]

        # Column widths
        rq += [
            _colw(did, 0, 165), _colw(did, 1, 100), _colw(did, 2, 90),
            _colw(did, 3, 80),  _colw(did, 4, 80),  _colw(did, 5, 90),
            _colw(did, 6, 80),  _colw(did, 7, 90),  _colw(did, 8, 90),
        ]

    # ── Daily Log ─────────────────────────────────────────────────────────
    if dlid is not None:
        rq += [
            _cell(dlid, 0, 1, 0, 14, bg=C['dkgrn'], fg=C['white'], bold=True, size=10, align='CENTER', valign='MIDDLE'),
            _rowh(dlid, 0, 28),
            _freeze(dlid, rows=1),
            _cell(dlid, 1, 2000, 0, 14, fg=C['dgray'], size=9),
            _borders(dlid, 0, 2000, 0, 14, C['border']),
            _cond(dlid, 1, 2000, 2, 5, 'TEXT_CONTAINS', '-', C['lneg'], C['neg']),
            _cond(dlid, 1, 2000, 2, 5, 'TEXT_CONTAINS', '+', C['lpos'], C['pos']),
            _colw(dlid, 0, 100), _colw(dlid, 1, 75),  _colw(dlid, 2, 75),
            _colw(dlid, 3, 75),  _colw(dlid, 4, 75),  _colw(dlid, 5, 60),
            _colw(dlid, 6, 140), _colw(dlid, 7, 95),  _colw(dlid, 8, 80),
            _colw(dlid, 9, 85),  _colw(dlid, 10, 90), _colw(dlid, 11, 95),
            _colw(dlid, 12, 115), _colw(dlid, 13, 115),
        ]

    # ── Trade Journal (formerly Trade Log) ────────────────────────────────
    if tjid is not None:
        rq += [
            _cell(tjid, 0, 1, 0, 15, bg=C['dkgrn'], fg=C['white'], bold=True, size=10, align='CENTER', valign='MIDDLE'),
            _rowh(tjid, 0, 28),
            _freeze(tjid, rows=1),
            _cell(tjid, 1, 2000, 0, 15, fg=C['dgray'], size=9),
            _cell(tjid, 1, 2000, 13, 15, wrap='WRAP'),
            _borders(tjid, 0, 2000, 0, 15, C['border']),
            _colw(tjid, 0, 100), _colw(tjid, 1, 65),  _colw(tjid, 2, 60),
            _colw(tjid, 3, 75),  _colw(tjid, 4, 80),  _colw(tjid, 5, 90),
            _colw(tjid, 6, 155), _colw(tjid, 7, 140), _colw(tjid, 8, 140),
            _colw(tjid, 9, 140), _colw(tjid, 10, 90), _colw(tjid, 11, 75),
            _colw(tjid, 12, 80), _colw(tjid, 13, 290), _colw(tjid, 14, 210),
        ]

    # ── Performance ───────────────────────────────────────────────────────
    if pid is not None:
        rq += [
            _cell(pid, 0, 1, 0, 5, bg=C['dkgrn'], fg=C['white'], bold=True, size=10, align='CENTER', valign='MIDDLE'),
            _rowh(pid, 0, 28),
            _freeze(pid, rows=1),
            _cell(pid, 1, 2000, 0, 5, fg=C['dgray'], size=9),
            _cell(pid, 1, 2000, 1, 3, align='RIGHT'),
            _borders(pid, 0, 2000, 0, 5, C['border']),
            _colw(pid, 0, 100), _colw(pid, 1, 120),
            _colw(pid, 2, 110), _colw(pid, 3, 100), _colw(pid, 4, 100),
        ]

    # ── Risk Monitor ──────────────────────────────────────────────────────
    if rmid is not None:
        rq += [
            _cell(rmid, 0, 1, 0, 12, bg=C['dkgrn'], fg=C['white'], bold=True, size=10, align='CENTER', valign='MIDDLE'),
            _rowh(rmid, 0, 28),
            _freeze(rmid, rows=1),
            _cell(rmid, 1, 2000, 0, 12, fg=C['dgray'], size=9),
            _borders(rmid, 0, 2000, 0, 12, C['border']),
            _cond(rmid, 2, 200, 4, 5, 'TEXT_CONTAINS', '+', C['lpos'], C['pos'], True),
            _cond(rmid, 2, 200, 4, 5, 'TEXT_CONTAINS', '-', C['lneg'], C['neg'], True),
            _cond(rmid, 2, 200, 5, 6, 'TEXT_CONTAINS', '+', C['lpos'], C['pos'], True),
            _cond(rmid, 2, 200, 5, 6, 'TEXT_CONTAINS', '-', C['lneg'], C['neg'], True),
            _colw(rmid, 0, 90),  _colw(rmid, 1, 110), _colw(rmid, 2, 80),
            _colw(rmid, 3, 80),  _colw(rmid, 4, 75),  _colw(rmid, 5, 80),
            _colw(rmid, 6, 80),  _colw(rmid, 7, 80),  _colw(rmid, 8, 85),
            _colw(rmid, 9, 80),  _colw(rmid, 10, 80), _colw(rmid, 11, 85),
        ]
        for i in range(2, 50):
            rq.append(_cell(rmid, i, i+1, 0, 12, bg=(C['white'] if i % 2 == 0 else C['xmint'])))

    # ── Market Pulse ──────────────────────────────────────────────────────
    if mpid is not None:
        rq += [
            _cell(mpid, 0, 1, 0, 8, bg=C['dkgrn'], fg=C['white'], bold=True, size=10, align='CENTER', valign='MIDDLE'),
            _rowh(mpid, 0, 28),
            _freeze(mpid, rows=1),
            _cell(mpid, 1, 2000, 0, 8, fg=C['dgray'], size=9),
            _borders(mpid, 0, 2000, 0, 8, C['border']),
            _cond(mpid, 1, 2000, 2, 5, 'TEXT_CONTAINS', '+', C['lpos'], C['pos']),
            _cond(mpid, 1, 2000, 2, 5, 'TEXT_CONTAINS', '-', C['lneg'], C['neg']),
            _colw(mpid, 0, 100), _colw(mpid, 1, 70),  _colw(mpid, 2, 75),
            _colw(mpid, 3, 75),  _colw(mpid, 4, 75),  _colw(mpid, 5, 65),
            _colw(mpid, 6, 90),  _colw(mpid, 7, 140),
        ]
        for i in range(1, 500):
            rq.append(_cell(mpid, i, i+1, 0, 8, bg=(C['white'] if i % 2 == 1 else C['xmint'])))

    # ── Sector History ────────────────────────────────────────────────────
    if shid is not None:
        rq += [
            _cell(shid, 0, 1, 0, 9, bg=C['dkgrn'], fg=C['white'], bold=True, size=10, align='CENTER', valign='MIDDLE'),
            _rowh(shid, 0, 28),
            _freeze(shid, rows=1),
            _cell(shid, 1, 2000, 0, 9, fg=C['dgray'], size=9),
            _borders(shid, 0, 2000, 0, 9, C['border']),
            _cond(shid, 1, 2000, 1, 9, 'NUMBER_GREATER_THAN_EQ', '0', C['lpos'], C['pos']),
            _cond(shid, 1, 2000, 1, 9, 'NUMBER_LESS',             '0', C['lneg'], C['neg']),
            _colw(shid, 0, 100),
        ]
        for i in range(1, 9):
            rq.append(_colw(shid, i, 105))

    # ── Closed Trades ─────────────────────────────────────────────────────
    if ctid is not None:
        rq += [
            _cell(ctid, 0, 1, 0, 8, bg=C['dkgrn'], fg=C['white'], bold=True, size=10, align='CENTER', valign='MIDDLE'),
            _rowh(ctid, 0, 28),
            _freeze(ctid, rows=1),
            _cell(ctid, 1, 2000, 0, 8, fg=C['dgray'], size=9),
            _borders(ctid, 0, 2000, 0, 8, C['border']),
            _cond(ctid, 1, 2000, 6, 7, 'TEXT_CONTAINS', '+', C['lpos'], C['pos'], True),
            _cond(ctid, 1, 2000, 6, 7, 'TEXT_CONTAINS', '-', C['lneg'], C['neg'], True),
            _cond(ctid, 1, 2000, 7, 8, 'TEXT_CONTAINS', '+', C['lpos'], C['pos'], True),
            _cond(ctid, 1, 2000, 7, 8, 'TEXT_CONTAINS', '-', C['lneg'], C['neg'], True),
            _colw(ctid, 0, 100), _colw(ctid, 1, 70),  _colw(ctid, 2, 130),
            _colw(ctid, 3, 85),  _colw(ctid, 4, 85),  _colw(ctid, 5, 65),
            _colw(ctid, 6, 75),  _colw(ctid, 7, 85),
        ]
        for i in range(1, 500):
            rq.append(_cell(ctid, i, i+1, 0, 8, bg=(C['white'] if i % 2 == 1 else C['xmint'])))

    # ── Analytics ─────────────────────────────────────────────────────────
    if anid is not None:
        rq += [
            _cell(anid, 0, 1, 0, 3, bg=C['blk'], fg=C['white'], bold=True, size=13, valign='MIDDLE'),
            _rowh(anid, 0, 36),
            _cell(anid, 2, 3, 0, 3, bg=C['grn'], fg=C['white'], bold=True),
            _cell(anid, 10, 11, 0, 3, bg=C['grn'], fg=C['white'], bold=True),
            _cell(anid, 1, 100, 0, 1, bold=True, fg=C['dgray']),
            _cell(anid, 1, 100, 1, 2, align='RIGHT', fg=C['dgray']),
            _colw(anid, 0, 210), _colw(anid, 1, 160),
        ]

    if rq:
        try:
            sh.batch_update({'requests': rq})
            print("Formatting applied.")
        except Exception as e:
            print(f"Formatting batch error: {e}")

    # ── Charts (one batch_update per tab to isolate failures) ────────────────
    if pid is not None:
        _del_charts(sh, pid)
        try:
            sh.batch_update({'requests': [
                _line_chart(pid, 2, 6, 'Portfolio Equity Curve', 0, 1, 'Total Equity ($)', w=660, h=360),
            ]})
            print("Performance chart added.")
        except Exception as e:
            print(f"Performance chart error: {e}")

    if dlid is not None:
        _del_charts(sh, dlid)
        try:
            sh.batch_update({'requests': [
                _line_chart(dlid, 2, 15, 'Portfolio Equity Over Time', 0, 13, 'Total Equity ($)', w=580, h=300),
            ]})
            print("Daily Log chart added.")
        except Exception as e:
            print(f"Daily Log chart error: {e}")

    if mpid is not None:
        _del_charts(sh, mpid)
        try:
            sh.batch_update({'requests': [
                _multi_line_chart(mpid, 2, 9,
                    'Market Performance — SPY / QQQ / IWM', 0, [2, 3, 4],
                    ['SPY %', 'QQQ %', 'IWM %'], w=660, h=310),
                _line_chart(mpid, 22, 9,
                    'VIX — Volatility Index', 0, 5, 'VIX', w=660, h=230),
                _line_chart(mpid, 38, 9,
                    'Fear & Greed Index', 0, 6, 'Score (0–100)', w=660, h=230),
            ]})
            print("Market Pulse charts added.")
        except Exception as e:
            print(f"Market Pulse chart error: {e}")

    if shid is not None:
        _del_charts(sh, shid)
        try:
            sh.batch_update({'requests': [
                _multi_line_chart(shid, 2, 10,
                    'Sector Rotation — Daily Change (%)', 0,
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    ['Technology', 'Healthcare', 'Financials', 'Energy',
                     'Consumer Disc', 'Industrials', 'Communication', 'Materials'],
                    w=760, h=420),
            ]})
            print("Sector History chart added.")
        except Exception as e:
            print(f"Sector History chart error: {e}")

    if ctid is not None:
        _del_charts(sh, ctid)
        try:
            sh.batch_update({'requests': [
                _bar_chart(ctid, 2, 9, 'P&L Per Trade ($)', 1, 7, w=600, h=320),
            ]})
            print("Closed Trades chart added.")
        except Exception as e:
            print(f"Closed Trades chart error: {e}")


# ── Slack ─────────────────────────────────────────────────────────────────

def get_sheet():
    if not GOOGLE_SHEET_ID or not os.path.exists(GCLOUD_KEY_PATH):
        return None
    try:
        creds = Credentials.from_service_account_file(
            GCLOUD_KEY_PATH,
            scopes=['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']
        )
        return gspread.authorize(creds).open_by_key(GOOGLE_SHEET_ID)
    except Exception as e:
        print(f"Sheets connection error: {e}")
        return None


def ensure_tab(sh, title, headers):
    """Get or create a tab. If headers changed, clear and reset. Always ensure enough columns."""
    needed_cols = len(headers) + 6
    try:
        ws = sh.worksheet(title)
        current = ws.row_values(1)
        if current != headers:
            ws.clear()
            if ws.col_count < needed_cols:
                ws.resize(rows=2000, cols=needed_cols)
            ws.append_row(headers, value_input_option='USER_ENTERED')
        elif ws.col_count < needed_cols:
            ws.resize(rows=2000, cols=needed_cols)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=2000, cols=needed_cols)
        ws.append_row(headers, value_input_option='USER_ENTERED')
    return ws


def update_sheets(mode, today, spy_chg, qqq_chg, iwm_chg, vix, fg_score, sector_perf,
                  held, trades_done, buying_power, equity, stops_hit, targets_hit):
    sh = get_sheet()
    if not sh:
        print("Sheets not configured, skipping.")
        return

    prev_equity = get_prev_equity(sh, today)
    daily_pnl   = equity - prev_equity if prev_equity is not None else 0.0

    try:
        # ── Dashboard (full refresh every run) ────────────────────────────
        try:
            dash = sh.worksheet('Dashboard')
            dash.clear()
        except gspread.exceptions.WorksheetNotFound:
            dash = sh.add_worksheet(title='Dashboard', rows=100, cols=10)

        rows = [
            ['  TRADING BOT DASHBOARD'],
            [f'  {today}   ·   {mode.upper()}   ·   Updated: {datetime.now().strftime("%H:%M")} ET'],
            [''],
            ['  KEY METRICS'],
            ['Total Equity', 'Cash', 'Daily P&L', '# Positions'],
            [f'${equity:.2f}', f'${buying_power:.2f}', f'${daily_pnl:+.2f}', str(len(held))],
            [''],
            ['  MARKET OVERVIEW'],
            ['SPY', f'{spy_chg:+.2f}%'],
            ['QQQ', f'{qqq_chg:+.2f}%'],
            ['IWM', f'{iwm_chg:+.2f}%'],
            ['VIX', f'{vix:.1f}'],
            ['Fear & Greed', fg_label(fg_score)],
            [''],
            ['  SECTOR ROTATION', 'Change', 'vs SPY'],
        ]
        for sec, d in sorted(sector_perf.items(), key=lambda x: x[1]['rs'], reverse=True):
            rows.append([sec, f"{d['chg']:+.2f}%", f"{d['rs']:+.2f}%"])
        rows.append([''])
        rows.append(['  OPEN POSITIONS', 'Entry', 'Current', 'P&L %', 'P&L $', 'Stop', 'To Stop', 'Target', 'To Target'])
        for sym, h in held.items():
            px     = h['price'] if h['price'] > 0 else h['cost']
            dpnl   = (px - h['cost']) * h['qty']
            tostop = (px - h['stop']) / px * 100
            totgt  = (h['cost'] * 1.20 - px) / px * 100
            rows.append([sym, f"${h['cost']:.2f}", f"${px:.2f}",
                         f"{h['pnl']:+.1f}%", f"${dpnl:+.2f}",
                         f"${h['stop']:.2f}", f"{tostop:.1f}%",
                         f"${h['cost']*1.20:.2f}", f"{totgt:.1f}%"])
        if not held:
            rows.append(['No open positions'])
        rows += [[''], ['  PORTFOLIO SUMMARY'],
                 ['Cash', f'${buying_power:.2f}'],
                 ['Total Equity', f'${equity:.2f}'],
                 ['# Positions', str(len(held))]]
        dash.update(rows, 'A1')
        print("Dashboard updated.")
    except Exception as e:
        print(f"Dashboard error: {e}")

    try:
        # ── Daily Log (one row per run) ───────────────────────────────────
        hdrs = ['Date','Mode','SPY%','QQQ%','IWM%','VIX','Fear&Greed','# Positions',
                '# Trades','Stops Hit','Targets Hit','Daily P&L','Cash','Total Equity']
        ws = ensure_tab(sh, 'Daily Log', hdrs)
        ws.append_row([today, mode, f'{spy_chg:+.2f}%', f'{qqq_chg:+.2f}%',
                       f'{iwm_chg:+.2f}%', f'{vix:.1f}', fg_label(fg_score),
                       len(held), len(trades_done),
                       len(stops_hit), len(targets_hit),
                       round(daily_pnl, 2), round(buying_power, 2), round(equity, 2)],
                      value_input_option='RAW')
        print("Daily log appended.")
    except Exception as e:
        print(f"Daily log error: {e}")

    try:
        # ── Trade Journal (one row per trade) ─────────────────────────────
        if trades_done:
            hdrs = ['Date','Symbol','Action','Price','Amount','Conviction',
                    'Fundamental','Macro','Technical','Innovation','EV',
                    'Stop','Target','Final Thesis','Primary Risk']
            ws = ensure_tab(sh, 'Trade Journal', hdrs)
            for t in trades_done:
                c = t['committee']
                ws.append_row([
                    today, t['symbol'], 'BUY', f"${t['price']:.2f}", f"${t['amount']:.2f}",
                    f"{c['conviction']}/10",
                    f"{c['fundamental']['verdict']} ({c['fundamental']['score']}/10)",
                    f"{c['macro']['verdict']} ({c['macro']['score']}/10)",
                    f"{c['technical']['verdict']} ({c['technical']['score']}/10)",
                    f"{c['innovation']['verdict']} ({c['innovation']['score']}/10)",
                    c['risk_manager']['ev'],
                    f"${t['stop']:.2f}", f"${t['target']:.2f}",
                    c.get('final_thesis',''), c.get('primary_risk','')
                ], value_input_option='USER_ENTERED')
            print(f"Trade journal: {len(trades_done)} trades logged.")
    except Exception as e:
        print(f"Trade journal error: {e}")

    try:
        # ── Performance (daily equity tracking) ───────────────────────────
        hdrs = ['Date','Total Equity','Cash','# Positions','Trades Today']
        ws = ensure_tab(sh, 'Performance', hdrs)
        ws.append_row([today, round(equity, 2), round(buying_power, 2),
                       len(held), len(trades_done)],
                      value_input_option='USER_ENTERED')
        print("Performance log appended.")
    except Exception as e:
        print(f"Performance log error: {e}")

    update_risk_monitor(sh, today, held, equity)
    update_market_pulse(sh, today, mode, spy_chg, qqq_chg, iwm_chg, vix, fg_score)
    update_sector_history(sh, today, sector_perf)
    if stops_hit or targets_hit:
        update_closed_trades(sh, today, stops_hit, targets_hit)
    update_analytics(sh, today)

    apply_formatting(sh, len(held))


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


# ── Session expiry warning ────────────────────────────────────────────────

def check_session_expiry():
    if not SESSION_DATE:
        return
    try:
        from datetime import date as _date
        created  = _date.fromisoformat(SESSION_DATE)
        expires  = created.replace(year=created.year, month=created.month, day=created.day)
        from datetime import timedelta
        expires  = created + timedelta(days=30)
        days_left = (expires - _date.today()).days
        if days_left <= 1:
            slack_send(
                f"{SLACK_MENTION} 🔑 *Robinhood session expires {'TODAY' if days_left <= 0 else 'TOMORROW'}* — renew it now or the bot will stop trading.\n\n"
                f"*Here's exactly what to do (takes 2 minutes):*\n"
                f"1️⃣  Open Terminal on your Mac\n"
                f"2️⃣  Run: `cd ~/Documents/Claude/trading/github-trading-bot && python3 setup_device_token.py`\n"
                f"3️⃣  Enter your Robinhood email and password when prompted (approve any SMS code)\n"
                f"4️⃣  It saves a file called `session.txt` — open it and copy the entire contents\n"
                f"5️⃣  SSH into the server: `ssh root@167.99.239.217`\n"
                f"6️⃣  Run: `python3 -c \"import os; val=input('Paste session: '); lines=open('/opt/trading-bot/.env').read().split('\\n'); lines=[l if not l.startswith('ROBINHOOD_SESSION') else f'ROBINHOOD_SESSION={{val}}' for l in lines]; open('/opt/trading-bot/.env','w').write('\\n'.join(lines))\"`\n"
                f"7️⃣  Paste the session value and hit Enter\n"
                f"8️⃣  Update the date: `sed -i 's/SESSION_DATE=.*/SESSION_DATE=$(date +%Y-%m-%d)/' /opt/trading-bot/.env`\n"
                f"✅  Done — bot will log in normally at next run."
            )
    except Exception as e:
        print(f"Session expiry check error: {e}")


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

    start_label = {'morning': 'Morning Analysis', 'midday': 'Midday Analysis',
                   'eod': 'EOD Recap'}.get(mode, 'Morning Analysis')
    brief = {'morning': 'Morning Brief', 'midday': 'Midday Brief',
             'eod': 'EOD Recap'}.get(mode, 'Morning Brief')
    slack_send(f"{SLACK_MENTION} :robot_face: *{start_label} started* — {today} | Analysis running, report incoming...")

    if mode == 'morning':
        check_session_expiry()

    rh_login()
    buying_power = get_buying_power()

    # Get market data first
    spy_data = get_stock_data('SPY')
    qqq_data = get_stock_data('QQQ')
    iwm_data = get_stock_data('IWM')
    vix_data = get_stock_data('^VIX')
    spy_chg  = spy_data['day_chg'] if spy_data else 0.0
    qqq_chg  = qqq_data['day_chg'] if qqq_data else 0.0
    iwm_chg  = iwm_data['day_chg'] if iwm_data else 0.0
    vix      = vix_data['price']   if vix_data else 20.0
    print(f"Market: SPY {spy_chg:+.2f}% QQQ {qqq_chg:+.2f}% IWM {iwm_chg:+.2f}% VIX {vix:.1f}")

    # Sector rotation
    print("Fetching sector data...")
    sector_perf = get_sector_performance(spy_chg)
    fg_score    = calc_fear_greed(vix, spy_chg, qqq_chg, sector_perf)
    print(f"Fear & Greed: {fg_score} | Sectors: {len(sector_perf)} fetched")

    # Positions with full exit management
    held, stops_hit, targets_hit, trailing_log, buying_power = load_positions(buying_power)
    deployable  = buying_power * (1 - CASH_RESERVE)
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
        update_sheets('eod', today, spy_chg, qqq_chg, iwm_chg, vix, fg_score, sector_perf,
                      held, [], buying_power, equity, stops_hit, targets_hit)
        eod_lines = [f"{SLACK_MENTION} 📊 *EOD Recap ready — {today}* | <{SHEET_URL}|Open Dashboard>"]
        if targets_hit:
            for t in targets_hit:
                eod_lines.append(f"  🎯 *TAKE PROFIT:* {t['symbol']} @ ${t['price']:.2f} | {t['pnl']:+.1f}%")
        if stops_hit:
            for s in stops_hit:
                eod_lines.append(f"  ⚠️ *STOP HIT:* {s['symbol']} @ ${s['price']:.2f} | {s['pnl']:+.1f}%")
        eod_lines.append(f"  Cash: ${buying_power:.2f} | Equity: ${equity:.2f}")
        slack_send('\n'.join(eod_lines))
        r.logout()
        return

    # ── Morning / Midday mode (full buy logic) ─────────────────────────────
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

    market_up = spy_chg > 0
    qualified = sorted(
        [(sym, d) for sym, d in stock_data.items()
         if d['day_chg'] > spy_chg and vcp_score(d, spy_chg) >= 2],
        key=lambda x: x[1]['day_chg'] - spy_chg,
        reverse=True,
    )
    print(f"{len(qualified)} passed screen (SPY {spy_chg:+.2f}%). Running full committee on top {min(10, len(qualified))}...")

    trades_done = []
    skipped     = []

    if not vix_block and deployable >= 10:
        for sym, data in qualified[:10]:
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

            if conviction >= MIN_CONVICTION and action == 'BUY' and deployable >= 10:
                base_pct = {6: 0.25, 7: 0.30, 8: 0.35, 9: 0.40, 10: 0.45}.get(min(conviction, 10), 0.25)
                if vix_reduce:
                    base_pct *= 0.70
                if dte is not None and dte <= 5:
                    base_pct *= 0.50
                dollar_amt = round(min(buying_power * base_pct, deployable, equity * 0.30), 2)
                if dollar_amt < 1:
                    skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'insufficient funds', 'rs': rs, 'action': action, 'data': data, 'committee': committee})
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
                    skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'order error: {e}', 'rs': rs, 'action': action, 'data': data, 'committee': committee})
            else:
                skipped.append({'symbol': sym, 'conviction': conviction,
                                'reason': f'conviction {conviction}/10 — below {MIN_CONVICTION}/10 threshold', 'rs': rs, 'action': action, 'data': data, 'committee': committee})

    # ── Capital rotation: sell a flat/red laggard to fund a strong unfunded pick ──
    rotated_out = []
    if not vix_block and held:
        strong_unfunded = sorted(
            [x for x in skipped
             if x.get('action') == 'BUY' and x.get('conviction', 0) >= ROTATE_INTO_CONVICTION
             and 'data' in x and x['symbol'] not in {t['symbol'] for t in trades_done}],
            key=lambda x: x['conviction'], reverse=True
        )
        if strong_unfunded:
            laggard_sym, laggard = min(held.items(), key=lambda kv: kv[1]['pnl'])
            pick = strong_unfunded[0]
            # only rotate out a position that is flat/red (not a winner we're letting run)
            if laggard['pnl'] < ROTATE_LAGGARD_MAX_PNL:
                try:
                    r.orders.order_sell_market(laggard_sym, laggard['qty'],
                                               account_number=ACCOUNT_NUMBER, jsonify=True)
                    proceeds = laggard['qty'] * laggard['price']
                    buying_power += proceeds
                    deployable   += proceeds
                    rotated_out.append({'symbol': laggard_sym, 'pnl': laggard['pnl'],
                                        'proceeds': proceeds, 'into': pick['symbol'],
                                        'conviction': pick['conviction']})
                    del held[laggard_sym]
                    print(f"  ↻ ROTATE: sold {laggard_sym} ({laggard['pnl']:+.1f}%) → funding {pick['symbol']} ({pick['conviction']}/10)")

                    sym  = pick['symbol']
                    data = pick['data']
                    dollar_amt = round(min(buying_power * 0.30, deployable, equity * 0.30), 2)
                    if dollar_amt >= 1:
                        r.orders.order_buy_fractional_by_price(
                            sym, dollar_amt, account_number=ACCOUNT_NUMBER, jsonify=True
                        )
                        rs  = data['day_chg'] - spy_chg
                        dte = pick.get('earnings_days')
                        trades_done.append({
                            'symbol':        sym,
                            'amount':        dollar_amt,
                            'price':         data['price'],
                            'conviction':    pick['conviction'],
                            'stop':          round(data['price'] * (1 - STOP_PCT), 2),
                            'target':        round(data['price'] * 1.20, 2),
                            'rs':            rs,
                            'committee':     pick['committee'],
                            'earnings_days': dte,
                        })
                        deployable   -= dollar_amt
                        buying_power -= dollar_amt
                        print(f"  ✓ Rotated into {sym} ${dollar_amt:.2f}")
                except Exception as e:
                    print(f"  Rotation failed: {e}")

    # ── Slack message ─────────────────────────────────────────────────────
    lines = [f"{SLACK_MENTION} 📈 *{brief} — {today}*\n"]

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

    if rotated_out:
        for ro in rotated_out:
            lines.append(f"↻ *Rotated:* sold {ro['symbol']} ({ro['pnl']:+.1f}%) → into {ro['into']} ({ro['conviction']}/10)\n")

    if trades_done:
        lines.append(f'*Committee bought {len(trades_done)} position(s) today:*\n')
        for t in trades_done:
            lines.append(build_trade_report(t))
            lines.append('')
    elif vix_block:
        lines.append('*No trades — VIX above 40 (extreme fear, sitting out).*')
    elif deployable < 10:
        lines.append('*No trades — buying power below minimum.*')
    else:
        lines.append('*No trades — nothing cleared the full committee today.*')
        near = sorted([x for x in skipped if x['conviction'] >= 5],
                      key=lambda x: x['conviction'], reverse=True)[:4]
        if near:
            lines.append('Near misses (reviewed but not cleared):')
            for x in near:
                lines.append(f"  • {x['symbol']}: {x['conviction']}/10 — {x['reason']}")

    lines.append(f"\n*Cash:* ${buying_power:.2f} | *Total equity:* ${equity:.2f}")
    lines.append('_Override: sell directly in Robinhood app_')

    # Write everything to Google Sheets
    update_sheets(mode, today, spy_chg, qqq_chg, iwm_chg, vix, fg_score, sector_perf,
                  held, trades_done, buying_power, equity, stops_hit, targets_hit)

    # Slack — minimal ping with link + any critical alerts
    ping_lines = [f"{SLACK_MENTION} 📊 *{brief} ready — {today}* | <{SHEET_URL}|Open Dashboard>"]
    if targets_hit:
        for t in targets_hit:
            ping_lines.append(f"  🎯 *TAKE PROFIT:* {t['symbol']} @ ${t['price']:.2f} | P&L: *{t['pnl']:+.1f}%*")
    if stops_hit:
        for s in stops_hit:
            ping_lines.append(f"  ⚠️ *STOP HIT:* {s['symbol']} @ ${s['price']:.2f} | P&L: {s['pnl']:+.1f}%")
    if trades_done:
        syms = ', '.join([f"{t['symbol']} ({t['committee']['conviction']}/10)" for t in trades_done])
        ping_lines.append(f"  💰 *Bought:* {syms}")
    else:
        ping_lines.append(f"  No new trades today.")
    ping_lines.append(f"  Cash: ${buying_power:.2f} | Equity: ${equity:.2f}")
    slack_send('\n'.join(ping_lines))
    r.logout()
    print("Done.")


if __name__ == '__main__':
    main()
