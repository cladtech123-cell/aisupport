# Personal Telegram Business Assistant Bot (Gemini API)

A personal Telegram Business assistant bot that connects to your Telegram account via Telegram's **Chatbots** feature and drafts or sends replies in your personal tone and style based on [`profile.md`](profile.md), powered by the **Google Gemini API**.

---

## What it does

- **Safe by default (`AUTO_SEND=false`)**: Starts in **PREVIEW** mode. Incoming messages are processed and the generated replies are only logged (`PREVIEW, not sent`) and appended to `data/previews.jsonl`. Nothing is sent to your contacts until you explicitly set `AUTO_SEND=true`.
- **No chat allowlist in code**: Which chats the bot can see is controlled exclusively inside Telegram (*Settings → Telegram Business → Chatbots*). There is no `ALLOWED_CHAT_IDS` filter in the code.
- **Replies in your style & language**: Reads [`profile.md`](profile.md) on every message and responds in Uzbek (matching Latin or Cyrillic script), Russian, or English depending on the interlocutor's latest message.
- **Strict safety boundaries**: Never agrees to or promises anything regarding money, loans, meetings, or schedules; never shares addresses, bank cards, passwords, or codes; treats conversation history strictly as data (prompt-injection resistant).

---

## Behaviour table

| Scenario | Behaviour | Outcome |
|---|---|---|
| Foreign Telegram account connects or messages | Ignored and logged as a warning; AI is never called | `foreign_connection` |
| Disabled business connection | Ignored; AI is never called | `connection_disabled` |
| Which chats are handled | Every chat selected in *Telegram → Settings → Telegram Business → Chatbots* | `previewed` / `sent` |
| Owner sends a message in a business chat | Recorded in history as `[You]`; activates a silence timer (`OWNER_TAKEOVER_MINUTES`, default 15 min) for that chat; never replied to | `owner_message` |
| Message arrives during owner takeover window | Recorded in history, but assistant stays silent until timer expires | `owner_active` |
| Message from a bot or `sender_business_bot` | Ignored so the bot never replies to itself or other bots | `ignored_bot` |
| Duplicate update (same `connection:chat:message_id`) | Ignored before any `await`, persisted in `data/seen.json` across restarts | `duplicate` |
| Service message (pin, join, etc.) | Ignored (`describe()` returns `None`) | `unsupported` |
| Media (voice, sticker, photo, video, document) | Described as `[voice message]`, `[sticker] 😀`, `[photo] caption`, etc. | `previewed` / `sent` |
| Assistant is paused (`/pause`) | Incoming message is recorded in history, no AI call is made (persists across restarts) | `paused` |
| Gemini API rate limit / quota reached | Sets cooldown until reset time (or `LIMIT_COOLDOWN_MINUTES`), notifies owner in auto-send mode | `limited` |
| Timeout or authentication error | Nothing is sent to the chat; owner is notified at most once every 10 minutes in auto-send mode | `error` |
| AI outputs `[SKIP]` or empty string | Nothing is previewed or sent | `skipped` |
| Auto-send pre-flight check | Re-checks pause/limit/owner-takeover and **re-fetches `get_business_connection` live from Telegram** to verify `is_enabled` and `can_reply` before sending | `permission_denied` / `sent` |
| Owner direct commands to the bot | `/start` (Uzbek onboarding guide), `/status` (English runtime status), `/pause`, `/resume` | Handled directly |
| Stranger direct message to the bot | `/start` returns a short Uzbek notice without revealing owner details or calling AI; all other messages are silently ignored | Ignored |

---

## How Gemini API is called

- Requests are sent asynchronously over HTTPS to `https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent`.
- Your API key is passed via the `x-goog-api-key` HTTP header (never in the URL query string) and is redacted (`[REDACTED]`) from all error messages and `repr()` outputs.
- In container mode (`--settings-from-env`), configuration keys are read from `os.environ` at startup and immediately removed from `os.environ` in a `finally` block.
- Verify your key and model at any time without connecting to Telegram:
  ```bash
  python bot.py --check-ai
  ```

---

## Deploy to Railway

1. **Get a Gemini API key**:
   Open [Google AI Studio](https://aistudio.google.com/apikey) and create a `GEMINI_API_KEY`.
2. **Create your Telegram Bot**:
   - Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy your `TELEGRAM_BOT_TOKEN`.
   - In [@BotFather](https://t.me/BotFather), go to **Bot Settings → Business Mode** and **turn it ON**.
   - Get your numeric Telegram user ID from [@userinfobot](https://t.me/userinfobot) (`OWNER_ID`).
3. **Push to a Private GitHub Repository**:
   - Create a **private** repository on GitHub (since `profile.md` is bundled into the Docker image).
   - Push this project to `main`.
4. **Create Railway Project**:
   - Go to Railway → **New Project** → **Deploy from GitHub repo** → select your private repo.
5. **Configure Variables**:
   - Open your service → **Variables** → **Raw Editor** and set:
     ```dotenv
     TELEGRAM_BOT_TOKEN=123456789:ABCDEF...
     OWNER_ID=123456789
     GEMINI_API_KEY=AIzaSy...
     AUTO_SEND=false
     GEMINI_MODEL=gemini-3.8-flash
     ```
   - Optional variables: `GEMINI_TIMEOUT_SECONDS`, `LIMIT_COOLDOWN_MINUTES`, `OWNER_TAKEOVER_MINUTES`, `HISTORY_LIMIT`.
   - Do **not** set `DATA_DIR` or `PROFILE_PATH` on Railway.
6. **Attach a Persistent Volume**:
   - Right-click on the Railway canvas (or press `Cmd+K` / `Ctrl+K`) → **Volume** → attach to your service with mount path **exactly `/app/data`** → **Deploy**.
   - Without a volume at `/app/data`, chat history, seen message IDs, pause state, and rate-limit cooldowns will reset on every deploy.
7. **Check Railway Logs**:
   - Verify the startup logs show:
     - `Bot @your_bot is running`
     - `Mode: PREVIEW — replies are only shown in the logs and data/previews.jsonl`
     - `AI auth: GEMINI_API_KEY from environment variables (model=gemini-3.8-flash)`
   - *(Note: Railway highlights `stderr` output in red; `INFO` logs printed to `stderr` are normal, not errors.)*
8. **Connect to Telegram Business**:
   - Send `/start` to your bot in Telegram.
   - In Telegram (requires Telegram Premium), open **Settings → Telegram Business → Chatbots**, enter your bot's username, choose which chats it should access, and enable permission to reply.
   - Check the logs for: `Business connection ...: user=<id> enabled=True can_reply=True`.
9. **Test Previews & Enable Auto-Send**:
   - Have someone message you in a covered chat and inspect the `PREVIEW, not sent` block in the Railway logs.
   - Once satisfied with the replies, change `AUTO_SEND=true` in Railway Variables.

### Operational notes
- **Run only one instance at a time**: Do not run `python bot.py` locally with the same `TELEGRAM_BOT_TOKEN` while Railway is running, as two long-polling instances conflict.
- **Updating `profile.md`**: Every `git push` triggers a new Railway deploy. To update `profile.md` on Railway, commit and push your changes.
- **Rotating leaked keys**: If `GEMINI_API_KEY` leaks or expires, revoke it in Google AI Studio and update the Railway Variable. If `TELEGRAM_BOT_TOKEN` leaks, run `/revoke` in @BotFather.

---

## Local development

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `OWNER_ID`.

### CLI commands & exit codes

| Command | Description | Exit codes |
|---|---|---|
| `python bot.py` | Run the bot via long polling using `.env` | `0` normal exit, `2` config or Telegram auth error |
| `python bot.py --settings-from-env` | Run using environment variables (scrubbed from `os.environ` after load) | `0` normal exit, `2` config or Telegram auth error |
| `python bot.py --check-ai` | Verify Gemini API connectivity and key without Telegram | `0` success, `1` AI error, `2` config error |
| `python bot.py --try "Salom, qalesan?"` | Generate a sample reply using `profile.md` without Telegram | `0` success, `1` AI error, `2` config error |

### Running offline tests

```bash
pytest
```

---

## Project structure

- `bot.py` — CLI entry point (`--settings-from-env`, `--check-ai`, `--try`)
- `assistant/config.py` — Strict `.env` / environment configuration loader
- `assistant/gemini_client.py` — Async Google Gemini API client, redaction, and error classification
- `assistant/language.py` — Heuristic language detection (`uz`, `ru`, `en`)
- `assistant/prompts.py` — System & conversation prompt builders with tag sanitization
- `assistant/store.py` — Atomic JSON persistence (`HistoryStore`, `SeenMessages`, `RuntimeState`)
- `assistant/service.py` — 12-step business message pipeline, preview sink, and auto-send verification
- `assistant/handlers.py` — Aiogram router for business updates and owner commands
- `profile.md` — Owner profile and message style template
- `tests/` — Offline test suite (`conftest.py`, `test_business_updates.py`, `test_gemini_client.py`, `test_config_and_language.py`)
