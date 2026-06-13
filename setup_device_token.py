#!/usr/bin/env python3
"""
One-time local setup — logs into Robinhood and exports the session as a
base64 string you paste into GitHub Secrets as ROBINHOOD_SESSION.

Usage:
    python3 setup_device_token.py
"""
import os, base64, getpass

try:
    import robin_stocks.robinhood as r
except ImportError:
    print("Install robin-stocks first:  pip3 install robin-stocks")
    raise SystemExit(1)

print("=== Robinhood Session Setup ===\n")
print("You'll log in once here. If Robinhood sends an SMS code, enter it.")
print("Afterward, the bot can log in from GitHub Actions with no SMS needed.\n")

email    = input("Robinhood email: ").strip()
password = getpass.getpass("Robinhood password: ")

print("\nLogging in...")
r.login(email, password, expiresIn=86400 * 30, store_session=True)

pickle_path = os.path.expanduser('~/.tokens/robinhood.pickle')
if not os.path.exists(pickle_path):
    print(f"\n❌ Session file not found at {pickle_path}")
    print("Login may have failed. Try again.")
    raise SystemExit(1)

with open(pickle_path, 'rb') as f:
    session_b64 = base64.b64encode(f.read()).decode()

r.logout()

print("\n✅ Session saved!\n")
print("=" * 60)
print("Add these 5 secrets to your GitHub repo:")
print("  Settings → Secrets and variables → Actions → New repository secret\n")
print(f"  ROBINHOOD_EMAIL     →  {email}")
print(f"  ROBINHOOD_PASSWORD  →  [password you just entered]")
print(f"  ROBINHOOD_SESSION   →  {session_b64}")
print(f"  SLACK_WEBHOOK_URL   →  https://hooks.slack.com/... (from Slack Incoming Webhooks)")
print(f"  ANTHROPIC_API_KEY   →  sk-ant-... (from console.anthropic.com)")
print("=" * 60)
print("\nNote: if the bot ever stops logging in (~30 days), re-run this script")
print("and update the ROBINHOOD_SESSION secret.")
