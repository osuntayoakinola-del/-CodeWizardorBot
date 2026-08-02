"""
CodeWizard Telegram Bot
An AI-powered coding & writing assistant for Telegram, backed by the Anthropic API.

Run locally:
    export TELEGRAM_BOT_TOKEN="..."
    export ANTHROPIC_API_KEY="..."
    python bot.py

Deployed on Railway, these two env vars are set in the project's Variables tab.
"""

import logging
import os
from collections import defaultdict

from anthropic import Anthropic, APIError
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Swap models freely: e.g. "claude-haiku-4-5-20251001" for a cheaper/faster bot,
# "claude-opus-4-8" for the strongest reasoning.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# How many past messages (user+assistant turns) to keep per chat, so context
# doesn't grow unbounded and blow past the API's token limits.
MAX_HISTORY_MESSAGES = 20

SYSTEM_PROMPT = (
    "You are CodeWizard, an AI coding and writing assistant living inside a "
    "Telegram bot. You help users write, debug, explain, and refactor code in "
    "any language, and you help with general writing tasks (emails, docs, "
    "copy, brainstorming). Keep replies focused and practical. Format code "
    "using triple-backtick Markdown code blocks with a language tag. Keep "
    "prose concise — this is a chat interface, not a document."
)

TELEGRAM_MAX_LEN = 4000  # stay under Telegram's 4096-char hard limit

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("codewizard-bot")

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# In-memory per-chat conversation history: {chat_id: [{"role": ..., "content": ...}, ...]}
# NOTE: this resets whenever the process restarts (e.g. redeploy). For durable
# history across restarts, swap this for a small database (Railway offers a
# free Postgres/Redis plugin) — see README.
conversations: dict[int, list[dict]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_for_telegram(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Split long text into Telegram-safe chunks, breaking on newlines where possible."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:]
    return chunks


async def ask_claude(chat_id: int, user_text: str) -> str:
    """Send the user's message plus recent history to Claude and return the reply text."""
    history = conversations[chat_id]
    history.append({"role": "user", "content": user_text})
    # Trim to the last N messages
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=history,
        )
    except APIError as e:
        logger.exception("Anthropic API error")
        # Don't poison history with a failed turn
        history.pop()
        return f"⚠️ Sorry, I hit an error talking to the AI backend: {e}"

    reply_text = "".join(block.text for block in response.content if block.type == "text")
    history.append({"role": "assistant", "content": reply_text})
    return reply_text


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hi, I'm *CodeWizard* — your AI coding & writing assistant.\n\n"
        "Send me a coding question, a bug, a snippet to review, or a writing "
        "task, and I'll help.\n\n"
        "Commands:\n"
        "/reset — clear our conversation history\n"
        "/help — show this message",
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    conversations.pop(chat_id, None)
    await update.message.reply_text("🧹 Conversation history cleared.")


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    reply_text = await ask_claude(chat_id, user_text)

    for chunk in split_for_telegram(reply_text):
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            # Fall back to plain text if Markdown parsing fails on odd formatting
            await update.message.reply_text(chunk)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception", exc_info=context.error)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("CodeWizard bot starting (model=%s)...", MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
