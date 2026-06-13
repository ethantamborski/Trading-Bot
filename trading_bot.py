#!/usr/bin/env python3
"""
Autonomous trading bot — full 6-member investment committee, earnings protection,
VCP/CANSLIM screening, conviction-based sizing. Runs on GitHub Actions at 9am + 4pm ET.
Pass 'morning' or 'eod' as sys.argv[1] (default: morning).
"""

import os, json, sys, time
from datetime import date, datetime

import robin_stocks.robinhood as r
import yfinance as yf
from slack_sdk import WebClient
import requests
import anthropic

# ── Config (from GitHub Secrets) ──────────────────────────────────────────
ROBINHOOD_EMAIL    = os.environ['ROBINHOOD_EMAIL']
ROBINHOOD_PASSWORD = os.environ['ROBINHOOD_PASSWORD']
ROBINHOOD_SESSION  = os.environ['ROBINHOOD_SESSION']
SLACK_WEBHOOK_URL  = os.environ['SLACK_WEBHOOK_URL']
ANTHROPIC_API_KEY  = os.environ['ANTHROPIC_API_KEY']

ACCOUNT_NUMBER = '456166776'
SLACK_MENTION  = '<@U0B8ZNEB9N2>'
STOP_PCT       = 0.12
CASH_RESERVE   = 0.20

CANDIDATES = [
    'CRWD', 'NVDA', 'ARM',  'RKLB', 'PLTR', 'MSTR', 'HOOD', 'IONQ',
    'CRCL', 'APP',  'SOFI', 'SOUN', 'COIN', 'AMD',  'SMCI', 'META',
    'GOOGL','AVGO', 'TSM',  'AAPL', 'MSFT', 'SHOP', 'RBLX', 'MARA',
    'RDDT', 'PANW', 'ZS',   'DDOG', 'SNOW', 'XOM',
]

ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Market data ───────────────────────────────────────────────────────────

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    return 100 - 100 / (1 + ag / al) if al else 100.0


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
        rsi       = calc_rsi(closes[-30:] if len(closes) >= 30 else closes)
        avg_vol   = sum(volumes[-20:]) / 20
        vol_ratio = volumes[-1] / avg_vol if avg_vol else 1.0
        high52    = max(hist['High'])
        return {
            'symbol':        symbol,
            'price':         round(price, 2),
            'day_chg':       round((price - prev) / prev * 100, 2),
            'ma20':          round(ma20, 2),
            'ma50':          round(ma50, 2),
            'above_ma20':    price > ma20,
            'above_ma50':    price > ma50,
            'rsi':           round(rsi, 1),
            'vol_ratio':     round(vol_ratio, 2),
            'pct_from_high': round((price - high52) / high52 * 100, 1),
        }
    except Exception as e:
        print(f"  Data error {symbol}: {e}")
        return None


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


def days_to_earnings(symbol):
    try:
        ticker = yf.Ticker(symbol)
        cal    = ticker.calendar
        if cal is None:
            return None
        # Handle both DataFrame and dict formats across yfinance versions
        if hasattr(cal, 'columns'):
            if 'Earnings Date' in cal.columns:
                ed = cal['Earnings Date'].iloc[0]
            else:
                return None
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


# ── Full 6-member investment committee ────────────────────────────────────

def run_committee(symbol, data, spy_chg, qqq_chg, vix, account_equity, held):
    rs    = data['day_chg'] - spy_chg
    vcp   = vcp_score(data, spy_chg)
    dte   = days_to_earnings(symbol)
    pos_ctx = ', '.join([f"{s} ({h['pnl']:+.1f}%)" for s, h in held.items()]) or 'None'

    earnings_note = ''
    if dte is not None:
        if dte <= 2:
            earnings_note = f'⚠️ EARNINGS IN {dte} DAYS — extremely high risk, avoid.'
        elif dte <= 5:
            earnings_note = f'⚠️ Earnings in {dte} days — reduce position size significantly.'

    market_regime = (
        'RISK-ON (bullish)'    if spy_chg > 0.5 and vix < 18 else
        'RISK-OFF (bearish)'   if spy_chg < -0.5 or vix > 28 else
        'NEUTRAL / MIXED'
    )

    prompt = f"""You are a six-member institutional investment decision committee. Analyze this potential swing trade as if billions of dollars are at risk. Prioritize truth, probability accuracy, and intellectual honesty over confidence or speed. Do NOT reach consensus artificially — disagreement is required where warranted.

═══════════════════════════════════════
MARKET CONDITIONS
═══════════════════════════════════════
Market regime: {market_regime}
SPY: {spy_chg:+.2f}% | QQQ: {qqq_chg:+.2f}% | VIX: {vix:.1f}

═══════════════════════════════════════
CANDIDATE: {symbol}
═══════════════════════════════════════
Price: ${data['price']} | Day change: {data['day_chg']:+.2f}%
Relative strength vs SPY: {rs:+.2f}%
RSI(14): {data['rsi']} | Above MA20: {data['above_ma20']} | Above MA50: {data['above_ma50']}
Volume: {data['vol_ratio']}x avg | Distance from 52w high: {data['pct_from_high']}%
VCP/CANSLIM score: {vcp}/6
{earnings_note}

═══════════════════════════════════════
ACCOUNT CONTEXT
═══════════════════════════════════════
Account equity: ${account_equity:.2f}
Open positions: {pos_ctx}
Parameters: 12% hard stop, 20%+ target, R/R ≥ 1.5:1, 1-4 week hold
Risk tolerance: Fully aggressive, maximum ROI

═══════════════════════════════════════
COMMITTEE INSTRUCTIONS
═══════════════════════════════════════

MEMBER 1 — FUNDAMENTAL ANALYST
Assess business quality, competitive moat, valuation, revenue/earnings growth trajectory, margins, balance sheet strength. Is this a fundamentally strong business at this price? Be specific to {symbol}.

MEMBER 2 — MACROECONOMIC STRATEGIST
How do current macro conditions (Fed policy, interest rates, sector rotation, global risks, dollar strength) affect this specific trade? Is the macro environment tailwind or headwind for {symbol}?

MEMBER 3 — TECHNICAL & FLOW ANALYST
Analyze the price action, trend structure, breakout quality, volume confirmation, RSI momentum, MA positioning, and institutional accumulation signals. Is the chart constructive for a long entry NOW?

MEMBER 4 — DEVIL'S ADVOCATE (CRITICAL — must aggressively attack)
Your ONLY job is to find every possible flaw, hidden risk, false narrative, and reason this trade fails. Attack the thesis hard. What is the market already pricing in? What could go catastrophically wrong? Why might the bulls be wrong?

MEMBER 5 — INNOVATION & INDUSTRY SPECIALIST
Is {symbol} positioned in a high-growth sector with durable tailwinds? Is the disruption thesis real and not yet fully priced in? Assess the 1-3 year industry positioning.

MEMBER 6 — PROBABILITY & RISK MANAGER
Run a probability-weighted scenario analysis. What is the expected value of this trade? Model bull/base/bear cases with probabilities. Does the reward justify the risk at this exact entry?

After all members analyze independently, they reassess in light of each other's arguments (especially the Devil's Advocate). Then produce the final committee verdict.

═══════════════════════════════════════
OUTPUT — respond in this exact JSON only, no markdown:
═══════════════════════════════════════
{{
  "fundamental":    {{"verdict": "BULL|BEAR|NEUTRAL", "analysis": "<2 sentences>", "score": <1-10>}},
  "macro":          {{"verdict": "BULL|BEAR|NEUTRAL", "analysis": "<2 sentences>", "score": <1-10>}},
  "technical":      {{"verdict": "BULL|BEAR|NEUTRAL", "analysis": "<2 sentences>", "score": <1-10>}},
  "devil_advocate": {{"main_attack": "<2 sentences of hardest criticism>", "failure_probability": "<% estimate>"}},
  "innovation":     {{"verdict": "BULL|BEAR|NEUTRAL", "analysis": "<2 sentences>", "score": <1-10>}},
  "risk_manager":   {{"ev": "POSITIVE|NEGATIVE|NEUTRAL", "bull_prob": <0-100>, "base_prob": <0-100>, "bear_prob": <0-100>, "analysis": "<2 sentences>"}},
  "reassessment":   "<1 sentence on what changed after devil's advocate attack>",
  "conviction":     <final 1-10 after full committee debate>,
  "action":         "BUY" or "SKIP",
  "final_thesis":   "<3 sentences: why buy now, what makes this setup exceptional, key catalyst>",
  "primary_risk":   "<the single biggest threat that could invalidate this trade>"
}}"""

    try:
        resp = ai_client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=900,
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


# ── Slack ─────────────────────────────────────────────────────────────────

def slack_send(msg):
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={'text': msg}, timeout=15)
        resp.raise_for_status()
        print("Slack sent.")
    except Exception as e:
        print(f"Slack error: {e}")


# ── Login ─────────────────────────────────────────────────────────────────

def rh_login():
    import base64
    pickle_dir  = os.path.expanduser('~/.tokens')
    pickle_path = os.path.join(pickle_dir, 'robinhood.pickle')
    os.makedirs(pickle_dir, exist_ok=True)
    with open(pickle_path, 'wb') as f:
        f.write(base64.b64decode(ROBINHOOD_SESSION))
    result = r.login(
        username=ROBINHOOD_EMAIL,
        password=ROBINHOOD_PASSWORD,
        store_session=True,
        expiresIn=3600,
    )
    if not result:
        slack_send(f"{SLACK_MENTION} ⚠️ *Trading bot login failed* — re-run setup_device_token.py and update ROBINHOOD_SESSION secret.")
        sys.exit(1)
    print("Robinhood login OK")


# ── Account helpers ───────────────────────────────────────────────────────

def get_buying_power():
    try:
        profile = r.account.load_account_profile(account_number=ACCOUNT_NUMBER)
        bp = float(profile.get('buying_power') or 0)
        if bp > 0:
            return bp
    except Exception as e:
        print(f"Profile error: {e}")
    port = r.profiles.load_portfolio_profile(account_number=ACCOUNT_NUMBER)
    return float(port.get('withdrawable_amount') or 0)


def load_positions():
    held      = {}
    stops_hit = []
    raw       = r.account.get_open_stock_positions(account_number=ACCOUNT_NUMBER) or []
    for pos in raw:
        try:
            instr = r.stocks.get_instrument_by_url(pos['instrument'])
            sym   = instr['symbol']
            qty   = float(pos['quantity'])
            cost  = float(pos['average_buy_price'])
            price = float((r.stocks.get_latest_price(sym) or [cost])[0])
            pnl   = (price - cost) / cost * 100
            stop  = round(cost * (1 - STOP_PCT), 2)
            held[sym] = {'qty': qty, 'cost': cost, 'price': price, 'pnl': pnl, 'stop': stop}
            if price <= stop:
                print(f"  STOP TRIGGERED: {sym} @ ${price:.2f}")
                r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                stops_hit.append({'symbol': sym, 'price': price, 'pnl': pnl})
                del held[sym]
        except Exception as e:
            print(f"  Position error: {e}")
    return held, stops_hit


# ── Slack report builder ──────────────────────────────────────────────────

def verdict_icon(v):
    return {'BULL': '🟢', 'BEAR': '🔴', 'NEUTRAL': '🟡'}.get(v, '⚪')

def build_trade_report(t):
    c  = t['committee']
    ev = {'POSITIVE': '✅', 'NEGATIVE': '❌', 'NEUTRAL': '⚖️'}.get(c['risk_manager']['ev'], '⚖️')
    lines = [
        f"*{t['symbol']}* — ${t['amount']:.2f} @ ${t['price']} | Target ${t['target']} | Stop ${t['stop']} | Conviction *{c['conviction']}/10*",
        f"  {verdict_icon(c['fundamental']['verdict'])} Fundamental: {c['fundamental']['analysis']}",
        f"  {verdict_icon(c['macro']['verdict'])} Macro: {c['macro']['analysis']}",
        f"  {verdict_icon(c['technical']['verdict'])} Technical: {c['technical']['analysis']}",
        f"  {verdict_icon(c['innovation']['verdict'])} Innovation: {c['innovation']['analysis']}",
        f"  {ev} Risk Manager: {c['risk_manager']['analysis']}",
        f"  🔴 Devil's Advocate: {c['devil_advocate']['main_attack']} _(failure probability: {c['devil_advocate']['failure_probability']})_",
        f"  ↩️ Reassessment: {c['reassessment']}",
        f"  📋 *Final thesis:* {c['final_thesis']}",
        f"  ⚠️ *Primary risk:* {c['primary_risk']}",
    ]
    if t.get('earnings_days') is not None and t['earnings_days'] <= 5:
        lines.insert(1, f"  ⚠️ *Earnings in {t['earnings_days']} days — sized down*")
    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    today = date.today().isoformat()
    mode  = sys.argv[1] if len(sys.argv) > 1 else 'morning'
    print(f"[{today}] mode={mode}")

    rh_login()
    buying_power = get_buying_power()
    held, stops_hit = load_positions()
    deployable = buying_power * (1 - CASH_RESERVE)
    print(f"Cash: ${buying_power:.2f} | Deployable: ${deployable:.2f}")

    spy_data = get_stock_data('SPY')
    qqq_data = get_stock_data('QQQ')
    vix_data = get_stock_data('^VIX')
    spy_chg  = spy_data['day_chg'] if spy_data else 0.0
    qqq_chg  = qqq_data['day_chg'] if qqq_data else 0.0
    vix      = vix_data['price']   if vix_data else 20.0
    print(f"Market: SPY {spy_chg:+.2f}% QQQ {qqq_chg:+.2f}% VIX {vix:.1f}")

    # ── EOD mode ──────────────────────────────────────────────────────────
    if mode == 'eod':
        lines = [f"{SLACK_MENTION} 📊 *EOD Recap — {today}*\n"]
        lines.append(f"*Market:* SPY {spy_chg:+.2f}% | QQQ {qqq_chg:+.2f}% | VIX {vix:.1f}\n")
        if held:
            lines.append('*Open positions:*')
            for sym, h in held.items():
                icon = '🟢' if h['pnl'] > 0 else '🔴'
                lines.append(f"  {icon} {sym}: ${h['price']:.2f} ({h['pnl']:+.1f}%) | Stop ${h['stop']:.2f}")
        else:
            lines.append('No open positions.')
        if stops_hit:
            lines.append('\n*Auto-sold (stops hit):*')
            for s in stops_hit:
                lines.append(f"  ⚠️ {s['symbol']} @ ${s['price']:.2f} | P&L: {s['pnl']:+.1f}%")
        lines.append(f"\n*Cash:* ${buying_power:.2f}")
        slack_send('\n'.join(lines))
        r.logout()
        return

    # ── Morning mode ──────────────────────────────────────────────────────
    vix_block  = vix > 40
    vix_reduce = vix > 30 and not vix_block
    account_equity = buying_power + sum(h['price'] * h['qty'] for h in held.values())

    # Screen candidates
    to_screen = [s for s in CANDIDATES if s not in held]
    print(f"Screening {len(to_screen)} candidates...")
    stock_data = {}
    for sym in to_screen:
        d = get_stock_data(sym)
        if d:
            stock_data[sym] = d
        time.sleep(0.25)

    qualified = sorted(
        [(sym, d) for sym, d in stock_data.items()
         if d['day_chg'] > spy_chg and vcp_score(d, spy_chg) >= 4],
        key=lambda x: x[1]['day_chg'] - spy_chg,
        reverse=True,
    )
    print(f"{len(qualified)} passed screen. Running full committee on top {min(6, len(qualified))}...")

    trades_done   = []
    skipped       = []

    if not vix_block and deployable >= 10:
        for sym, data in qualified[:6]:
            print(f"  Committee analyzing {sym}...")
            committee = run_committee(sym, data, spy_chg, qqq_chg, vix, account_equity, held)
            if not committee:
                continue

            conviction = committee.get('conviction', 0)
            action     = committee.get('action', 'SKIP')
            dte        = committee.get('earnings_days')
            rs         = data['day_chg'] - spy_chg
            print(f"  {sym}: {conviction}/10 → {action}")

            # Block trades within 2 days of earnings
            if dte is not None and dte <= 2:
                print(f"  {sym}: BLOCKED — earnings in {dte} days")
                skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'earnings in {dte} days', 'rs': rs})
                continue

            if conviction >= 8 and action == 'BUY' and deployable >= 10:
                base_pct = {8: 0.15, 9: 0.175, 10: 0.20}.get(min(conviction, 10), 0.15)
                if vix_reduce:
                    base_pct *= 0.70
                # Reduce by half if earnings within 5 days
                if dte is not None and dte <= 5:
                    base_pct *= 0.50
                dollar_amt = round(min(buying_power * base_pct, deployable, account_equity * 0.25), 2)
                if dollar_amt < 5:
                    skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'insufficient funds (${dollar_amt:.2f})', 'rs': rs})
                    continue
                try:
                    r.orders.order_buy_fractional_by_price(
                        sym, dollar_amt, account_number=ACCOUNT_NUMBER, jsonify=True
                    )
                    trades_done.append({
                        'symbol':       sym,
                        'amount':       dollar_amt,
                        'price':        data['price'],
                        'conviction':   conviction,
                        'stop':         round(data['price'] * (1 - STOP_PCT), 2),
                        'target':       round(data['price'] * 1.20, 2),
                        'vcp':          vcp_score(data, spy_chg),
                        'rs':           rs,
                        'committee':    committee,
                        'earnings_days': dte,
                    })
                    deployable   -= dollar_amt
                    buying_power -= dollar_amt
                    print(f"  ✓ Bought {sym} ${dollar_amt:.2f}")
                except Exception as e:
                    print(f"  Order failed {sym}: {e}")
                    skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'order failed: {e}', 'rs': rs})
            else:
                skipped.append({'symbol': sym, 'conviction': conviction, 'reason': f'conviction {conviction}/10 below threshold', 'rs': rs})

    # ── Build Slack message ───────────────────────────────────────────────
    lines = [f"{SLACK_MENTION} 📈 *Morning Brief — {today}*\n"]

    lines.append(f"*Market:* SPY {spy_chg:+.2f}% | QQQ {qqq_chg:+.2f}% | VIX {vix:.1f}")
    if vix_block:
        lines.append('⛔ *VIX > 40 — no new positions. Managing existing only.*')
    elif vix_reduce:
        lines.append('⚠️ VIX > 30 — position sizes reduced 30%')
    lines.append('')

    if stops_hit:
        lines.append('*Auto-sold (stops hit):*')
        for s in stops_hit:
            lines.append(f"  ⚠️ {s['symbol']} @ ${s['price']:.2f} | P&L: {s['pnl']:+.1f}%")
        lines.append('')

    if held:
        lines.append('*Open positions:*')
        for sym, h in held.items():
            icon = '🟢' if h['pnl'] > 0 else '🔴'
            lines.append(f"  {icon} {sym}: ${h['price']:.2f} ({h['pnl']:+.1f}%) | Stop ${h['stop']:.2f}")
        lines.append('')

    if trades_done:
        lines.append(f'*Committee executed {len(trades_done)} trade(s) today:*\n')
        for t in trades_done:
            lines.append(build_trade_report(t))
            lines.append('')
    elif vix_block:
        lines.append('*No trades — VIX above 40.*')
    elif deployable < 10:
        lines.append('*No trades — buying power below minimum.*')
    else:
        lines.append('*No trades — no setups cleared the full committee review.*')
        high_skipped = sorted([x for x in skipped if x['conviction'] >= 6],
                              key=lambda x: x['conviction'], reverse=True)[:3]
        if high_skipped:
            lines.append('Near misses:')
            for x in high_skipped:
                lines.append(f"  • {x['symbol']}: {x['conviction']}/10 — {x['reason']}")

    lines.append(f"*Cash remaining:* ${buying_power:.2f} | *Total equity:* ${account_equity:.2f}")
    lines.append('_To override: sell directly in the Robinhood app_')

    slack_send('\n'.join(lines))
    r.logout()
    print("Done.")


if __name__ == '__main__':
    main()
