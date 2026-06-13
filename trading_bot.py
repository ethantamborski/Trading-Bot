#!/usr/bin/env python3
"""
Autonomous trading bot — runs on GitHub Actions at 9am + 4pm ET, Mon-Fri.
Pass 'morning' or 'eod' as sys.argv[1] (default: morning).
"""

import os, json, sys, time
from datetime import date

import requests
import robin_stocks.robinhood as r
import yfinance as yf
import anthropic

# ── Config (from GitHub Secrets) ──────────────────────────────────────────
ROBINHOOD_EMAIL     = os.environ['ROBINHOOD_EMAIL']
ROBINHOOD_PASSWORD  = os.environ['ROBINHOOD_PASSWORD']
ROBINHOOD_SESSION   = os.environ['ROBINHOOD_SESSION']    # base64-encoded pickle
SLACK_WEBHOOK_URL   = os.environ['SLACK_WEBHOOK_URL']
ANTHROPIC_API_KEY   = os.environ['ANTHROPIC_API_KEY']

ACCOUNT_NUMBER = '456166776'
SLACK_MENTION  = '<@U0B8ZNEB9N2>'
STOP_PCT       = 0.12    # 12% hard stop
CASH_RESERVE   = 0.20    # keep 20% cash at all times

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


# ── AI conviction ─────────────────────────────────────────────────────────

def get_ai_conviction(symbol, data, spy_chg, qqq_chg, vix):
    rs  = data['day_chg'] - spy_chg
    vcp = vcp_score(data, spy_chg)
    prompt = (
        f"You are an elite momentum swing trader. Score this setup for a 1-4 week trade.\n\n"
        f"{symbol}: ${data['price']} | Day: {data['day_chg']:+.2f}% | RS vs SPY: {rs:+.2f}%\n"
        f"RSI: {data['rsi']} | Above MA20: {data['above_ma20']} | Above MA50: {data['above_ma50']}\n"
        f"Volume: {data['vol_ratio']}x avg | From 52w high: {data['pct_from_high']}% | VCP: {vcp}/6\n"
        f"Market: SPY {spy_chg:+.2f}% QQQ {qqq_chg:+.2f}% VIX {vix:.1f}\n\n"
        f"Rules: stop 12% below entry, target 20%+ (R/R >= 1.5:1), aggressive risk ok.\n\n"
        f'Return ONLY raw JSON — no markdown, no code fences:\n'
        f'{{"conviction": <int 1-10>, "action": "BUY" or "SKIP", "thesis": "<one sentence>", "risk": "<main risk>"}}'
    )
    try:
        resp = ai_client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=150,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = resp.content[0].text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"  AI error {symbol}: {e}")
        return {'conviction': 0, 'action': 'SKIP', 'thesis': 'AI error', 'risk': ''}


# ── Slack ─────────────────────────────────────────────────────────────────

def slack_send(msg):
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json={'text': msg}, timeout=10)
        r.raise_for_status()
        print("Slack sent.")
    except Exception as e:
        print(f"Slack error: {e}")


# ── Login ─────────────────────────────────────────────────────────────────

def rh_login():
    # Restore session pickle so login doesn't prompt for SMS
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
        slack_send(f"{SLACK_MENTION} ⚠️ *Trading bot login failed* — session may have expired. Re-run setup_device_token.py and update ROBINHOOD_SESSION secret.")
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
    # fallback
    port = r.profiles.load_portfolio_profile(account_number=ACCOUNT_NUMBER)
    return float(port.get('withdrawable_amount') or 0)


def load_positions():
    held = {}
    stops_hit = []
    raw = r.account.get_open_stock_positions(account_number=ACCOUNT_NUMBER) or []
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
                print(f"  STOP TRIGGERED: {sym} @ ${price:.2f} (stop ${stop:.2f})")
                r.orders.order_sell_market(sym, qty, account_number=ACCOUNT_NUMBER, jsonify=True)
                stops_hit.append({'symbol': sym, 'price': price, 'pnl': pnl})
                del held[sym]
        except Exception as e:
            print(f"  Position error: {e}")
    return held, stops_hit


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    today = date.today().isoformat()
    mode  = sys.argv[1] if len(sys.argv) > 1 else 'morning'
    print(f"[{today}] mode={mode}")

    rh_login()
    buying_power = get_buying_power()
    held, stops_hit = load_positions()

    # Refund buying power for stopped-out positions
    for s in stops_hit:
        buying_power += s['price'] * next(
            (h['qty'] for sym, h in held.items() if sym == s['symbol']), 0
        )
    deployable = buying_power * (1 - CASH_RESERVE)
    print(f"Cash: ${buying_power:.2f} | Deployable: ${deployable:.2f}")

    # Market overview
    spy_data = get_stock_data('SPY')
    qqq_data = get_stock_data('QQQ')
    vix_data = get_stock_data('^VIX')
    spy_chg  = spy_data['day_chg'] if spy_data else 0.0
    qqq_chg  = qqq_data['day_chg'] if qqq_data else 0.0
    vix      = vix_data['price']   if vix_data else 20.0
    print(f"Market: SPY {spy_chg:+.2f}% QQQ {qqq_chg:+.2f}% VIX {vix:.1f}")

    # ── EOD mode: just report, no new trades ─────────────────────────────
    if mode == 'eod':
        lines = [f"{SLACK_MENTION} 📊 *EOD Recap — {today}*\n"]
        lines.append(f"*Market close:* SPY {spy_chg:+.2f}% | QQQ {qqq_chg:+.2f}%\n")
        if held:
            lines.append('*Open positions:*')
            for sym, h in held.items():
                icon = '🟢' if h['pnl'] > 0 else '🔴'
                lines.append(f"  {icon} {sym}: ${h['price']:.2f} ({h['pnl']:+.1f}%) | Stop ${h['stop']:.2f}")
        else:
            lines.append('No open positions.')
        if stops_hit:
            lines.append('\n*Auto-sold today (stops hit):*')
            for s in stops_hit:
                lines.append(f"  ⚠️ {s['symbol']} @ ${s['price']:.2f} | P&L: {s['pnl']:+.1f}%")
        lines.append(f"\n*Cash:* ${buying_power:.2f}")
        slack_send('\n'.join(lines))
        r.logout()
        return

    # ── Morning mode: screen + trade ─────────────────────────────────────
    vix_block  = vix > 40
    vix_reduce = vix > 30 and not vix_block

    to_screen = [s for s in CANDIDATES if s not in held]
    print(f"Screening {len(to_screen)} candidates...")
    stock_data = {}
    for sym in to_screen:
        d = get_stock_data(sym)
        if d:
            stock_data[sym] = d
        time.sleep(0.25)

    # Filter: outperform SPY + VCP >= 4
    qualified = sorted(
        [(sym, d) for sym, d in stock_data.items()
         if d['day_chg'] > spy_chg and vcp_score(d, spy_chg) >= 4],
        key=lambda x: x[1]['day_chg'] - spy_chg,
        reverse=True,
    )
    print(f"{len(qualified)} passed screen. Analyzing top {min(8, len(qualified))}...")

    trades_done = []
    reviewed    = []

    if not vix_block and deployable >= 10:
        for sym, data in qualified[:8]:
            verdict    = get_ai_conviction(sym, data, spy_chg, qqq_chg, vix)
            conviction = verdict.get('conviction', 0)
            action     = verdict.get('action', 'SKIP')
            rs         = data['day_chg'] - spy_chg
            reviewed.append({'symbol': sym, 'conviction': conviction, 'rs': rs, 'action': action})
            print(f"  {sym}: {conviction}/10 → {action}")

            if conviction >= 8 and action == 'BUY' and deployable >= 10:
                base_pct = {8: 0.15, 9: 0.175, 10: 0.20}.get(min(conviction, 10), 0.15)
                if vix_reduce:
                    base_pct *= 0.70
                dollar_amt = round(min(buying_power * base_pct, deployable, buying_power * 0.25), 2)
                if dollar_amt < 5:
                    reviewed[-1]['skip_reason'] = f'insufficient funds (${dollar_amt:.2f})'
                    continue
                try:
                    r.orders.order_buy_fractional_by_price(
                        sym, dollar_amt, account_number=ACCOUNT_NUMBER, jsonify=True
                    )
                    trades_done.append({
                        'symbol': sym, 'amount': dollar_amt, 'price': data['price'],
                        'conviction': conviction, 'thesis': verdict.get('thesis', ''),
                        'risk': verdict.get('risk', ''), 'rs': rs,
                        'stop': round(data['price'] * (1 - STOP_PCT), 2),
                        'target': round(data['price'] * 1.20, 2),
                        'vcp': vcp_score(data, spy_chg),
                    })
                    deployable   -= dollar_amt
                    buying_power -= dollar_amt
                    print(f"  ✓ Bought {sym} ${dollar_amt:.2f}")
                except Exception as e:
                    print(f"  Order failed {sym}: {e}")
                    reviewed[-1]['skip_reason'] = f'order failed: {e}'

    # ── Build Slack message ───────────────────────────────────────────────
    lines = [f"{SLACK_MENTION} 📈 *Morning Brief — {today}*\n"]
    lines.append(f"*Market:* SPY {spy_chg:+.2f}% | QQQ {qqq_chg:+.2f}% | VIX {vix:.1f}")
    if vix_block:
        lines.append('⚠️ *VIX > 40 — no new positions taken*')
    elif vix_reduce:
        lines.append('⚠️ VIX > 30 — sizes reduced 30%')
    lines.append('')

    if stops_hit:
        lines.append('*Auto-sold (stop hit):*')
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
        lines.append('*Bought today:*')
        for t in trades_done:
            lines.append(
                f"  • *{t['symbol']}* ${t['amount']:.2f} @ ${t['price']}"
                f" | Target ${t['target']} | Stop ${t['stop']} | {t['conviction']}/10"
            )
        lines.append('\n*Why:*')
        for t in trades_done:
            lines.append(f"  • {t['symbol']}: {t['thesis']} _(risk: {t['risk']})_")
    elif vix_block:
        lines.append('*No trades — VIX above 40.*')
    elif deployable < 10:
        lines.append('*No trades — buying power below minimum.*')
    else:
        lines.append('*No trades today — nothing hit conviction threshold.*')
        near_misses = sorted(
            [x for x in reviewed if x['conviction'] >= 6],
            key=lambda x: x['conviction'], reverse=True
        )[:3]
        if near_misses:
            lines.append('Near misses:')
            for x in near_misses:
                lines.append(f"  • {x['symbol']}: {x['conviction']}/10 (RS {x['rs']:+.2f}%)")

    lines.append(f"\n*Cash remaining:* ${buying_power:.2f}")
    lines.append('_To override: sell directly in the Robinhood app_')

    slack_send('\n'.join(lines))
    r.logout()
    print("Done.")


if __name__ == '__main__':
    main()
