import os
import re
import sqlite3
import time
import asyncio
from io import BytesIO
from datetime import datetime, timedelta, UTC
from pathlib import Path
from collections import defaultdict, deque

import discord
from discord.ext import commands
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient


# =========================
# Load env
# =========================
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

TOKEN = (os.getenv("DISCORD_TOKEN") or "").strip()
PREFIX = (os.getenv("PREFIX") or "!").strip()

OWNER_ID = int(os.getenv("OWNER_ID") or "0")

MODLOG_CHANNEL_ID = int(os.getenv("MODLOG_CHANNEL_ID") or "0")
WELCOME_CHANNEL_ID = int(os.getenv("WELCOME_CHANNEL_ID") or "0")
LEAVE_CHANNEL_ID = int(os.getenv("LEAVE_CHANNEL_ID") or "0")
DELETED_LOG_CHANNEL_ID = int(os.getenv("DELETED_LOG_CHANNEL_ID") or "0")
AUTO_ROLE_ID = int(os.getenv("AUTO_ROLE_ID") or "0")

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
HF_MODEL = (os.getenv("HF_MODEL")
            or "meta-llama/Meta-Llama-3-8B-Instruct").strip()
HF_IMAGE_MODEL = (os.getenv("HF_IMAGE_MODEL")
                  or "stabilityai/stable-diffusion-xl-base-1.0").strip()

AI_COOLDOWN_S = int(os.getenv("AI_COOLDOWN_S") or "4")
IMG_COOLDOWN_S = int(os.getenv("IMG_COOLDOWN_S") or "8")
SELFROLE_MAX_OPTIONS = int(os.getenv("SELFROLE_MAX_OPTIONS") or "25")

if not TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN in .env")

print("Loaded .env from:", ENV_PATH)
print("PREFIX:", PREFIX)
print("OWNER_ID:", OWNER_ID)
print("MODLOG_CHANNEL_ID:", MODLOG_CHANNEL_ID)
print("WELCOME_CHANNEL_ID:", WELCOME_CHANNEL_ID)
print("LEAVE_CHANNEL_ID:", LEAVE_CHANNEL_ID)
print("DELETED_LOG_CHANNEL_ID:", DELETED_LOG_CHANNEL_ID)
print("AUTO_ROLE_ID:", AUTO_ROLE_ID)
print("HF_MODEL:", HF_MODEL)
print("HF_IMAGE_MODEL:", HF_IMAGE_MODEL)
print("HF enabled:", bool(HF_TOKEN))


# =========================
# Intents
# =========================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # join/leave + invite tracking

bot = commands.Bot(command_prefix=PREFIX, intents=intents)


# =========================
# HF client (AI + Image)
# =========================
hf = AsyncInferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None


# =========================
# Persona for AI chat
# =========================
SYSTEM_PERSONA = f"""
You are root.

You speak like a real, thoughtful human — not an assistant, not a bot.
Owner/creator is Rasad (Discord user id: {OWNER_ID if OWNER_ID else "unknown"}).

Style:
- Calm, sharp, friendly.
- Never say “As an AI…”.
- No roleplay stage directions like "(nods)".

Rules:
- If asked for illegal/harmful stuff, refuse briefly and offer a safe alternative.
- Keep replies concise unless the user asks for details.
""".strip()


def utc_now() -> datetime:
    """Timezone-aware UTC now (avoids deprecated utcnow())."""
    return datetime.now(UTC)


# =========================
# Commands list helper
# =========================
def commands_text(prefix: str) -> str:
    owner_line = f"Owner: <@{OWNER_ID}>\n\n" if OWNER_ID else "\n"
    return (
        "**root — commands**\n"
        + owner_line +
        "**AI / Image**\n"
        f"- `{prefix}img <prompt>` → generate an image\n"
        f"- type `root:` + your message OR mention the bot to chat\n"
        f"- type `root` to see this command list\n\n"
        "**Self roles**\n"
        f"- `{prefix}selfroles` → self-role menu commands\n"
        f"- `{prefix}selfrolesetup #channel @Role1 @Role2 ...` → create a dropdown\n\n"
        "**Moderation**\n"
        f"- `{prefix}kick @user <reason>`\n"
        f"- `{prefix}ban @user <reason>`\n"
        f"- `{prefix}unban name#1234`\n"
        f"- `{prefix}mute @user <10m|2h|1d> <reason>`\n"
        f"- `{prefix}unmute @user <reason>`\n"
        f"- `{prefix}purge [1-200]` (default 5)\n"
        f"- `{prefix}purgeuser @user [amount]` (default 20)\n"
        f"- `{prefix}warn @user <reason>`\n"
        f"- `{prefix}warnings @user`\n"
        f"- `{prefix}clearwarnings @user`\n"
        f"- `{prefix}echo @user <message>` → mention + say\n\n"
        "**Setup (admin)**\n"
        f"- `{prefix}setwelcome #channel`\n"
        f"- `{prefix}setleave #channel`\n"
        f"- `{prefix}setmodlog #channel`\n"
        f"- `{prefix}setdeletedlog #channel`\n"
        f"- `{prefix}setautorole @role`\n\n"
        "**Other**\n"
        f"- `{prefix}commands` (aliases: `{prefix}helpme`, `{prefix}cmds`)\n"
        f"- `{prefix}owner`\n"
    )


# =========================
# AI memory + cooldown
# =========================
MAX_TURNS = 10
ai_memory = defaultdict(lambda: deque(maxlen=MAX_TURNS))
_last_ai_call = {}   # user_id -> last ts
_last_img_call = {}  # user_id -> last ts


def split_for_discord(text: str, limit: int = 1900) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


# =========================
# SQLite (warnings + invite stats + selfroles)
# =========================
DB_PATH = Path(__file__).with_name("modbot.db")


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS warnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guild_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            moderator_id INTEGER NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS invite_stats (
            guild_id INTEGER NOT NULL,
            inviter_id INTEGER NOT NULL,
            joins INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (guild_id, inviter_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS selfrole_menus (
            guild_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            PRIMARY KEY (guild_id, message_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS selfrole_items (
            guild_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            role_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            PRIMARY KEY (guild_id, message_id, role_id)
        )
    """)
    conn.commit()
    return conn


# =========================
# Helpers
# =========================
def parse_duration(s: str) -> timedelta | None:
    m = re.fullmatch(r"(\d+)([smhd])", s.strip().lower())
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    return {"s": timedelta(seconds=n), "m": timedelta(minutes=n), "h": timedelta(hours=n), "d": timedelta(days=n)}.get(unit)


def fmt_user(u: discord.abc.User) -> str:
    return f"{u} (`{u.id}`)"


def get_text_channel(guild: discord.Guild, channel_id: int) -> discord.TextChannel | None:
    if channel_id == 0:
        return None
    ch = guild.get_channel(channel_id)
    return ch if isinstance(ch, discord.TextChannel) else None


async def send_embed(channel: discord.TextChannel | None, title: str, fields: list[tuple[str, str]], color: int):
    if not channel:
        return
    embed = discord.Embed(title=title, color=color, timestamp=utc_now())
    for n, v in fields:
        embed.add_field(name=n, value=v, inline=False)
    await channel.send(embed=embed)


async def modlog(guild: discord.Guild, title: str, fields: list[tuple[str, str]], color: int = 0x2F3136):
    ch = get_text_channel(guild, MODLOG_CHANNEL_ID)
    await send_embed(ch, title, fields, color)


def inc_invite_joins(guild_id: int, inviter_id: int) -> int:
    conn = db()
    conn.execute(
        "INSERT INTO invite_stats (guild_id, inviter_id, joins) VALUES (?, ?, 1) "
        "ON CONFLICT(guild_id, inviter_id) DO UPDATE SET joins = joins + 1",
        (guild_id, inviter_id),
    )
    conn.commit()
    row = conn.execute(
        "SELECT joins FROM invite_stats WHERE guild_id=? AND inviter_id=?",
        (guild_id, inviter_id)
    ).fetchone()
    conn.close()
    return int(row[0]) if row else 1


# =========================
# Invite tracking cache
# =========================
invite_cache: dict[int, dict[str, int]] = {}


async def refresh_invites_for_guild(guild: discord.Guild):
    try:
        invites = await guild.invites()
        invite_cache[guild.id] = {inv.code: (inv.uses or 0) for inv in invites}
    except discord.Forbidden:
        invite_cache[guild.id] = {}
    except discord.HTTPException:
        pass


async def detect_used_invite(guild: discord.Guild):
    try:
        current = await guild.invites()
    except (discord.Forbidden, discord.HTTPException):
        return None, None

    old = invite_cache.get(guild.id, {})
    used = None
    for inv in current:
        if (inv.uses or 0) > old.get(inv.code, 0):
            used = inv
            break

    invite_cache[guild.id] = {inv.code: (inv.uses or 0) for inv in current}
    return (used.code if used else None), used


# =========================
# AI trigger rules
# =========================
def should_ai_respond(message: discord.Message) -> bool:
    if message.author.bot:
        return False
    mentioned = bot.user in message.mentions if bot.user else False
    prefix = message.content.lower().startswith("root:")
    return mentioned or prefix


def extract_ai_text(message: discord.Message) -> str:
    text = message.content
    if bot.user:
        text = text.replace(f"<@{bot.user.id}>",
                            "").replace(f"<@!{bot.user.id}>", "")
    if text.lower().startswith("root:"):
        text = text.split(":", 1)[1]
    return text.strip()


async def hf_chat(channel_id: int, user_text: str) -> str:
    if not hf:
        return "AI isn't configured. Add HF_TOKEN in .env."

    msgs = [{"role": "system", "content": SYSTEM_PERSONA}]
    for role, content in ai_memory[channel_id]:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text})

    for attempt in range(4):
        try:
            out = await hf.chat.completions.create(
                model=HF_MODEL,
                messages=msgs,
                temperature=0.8,
                top_p=0.9,
                max_tokens=450,
            )
            reply = (out.choices[0].message.content or "").strip()
            if reply:
                ai_memory[channel_id].append(("user", user_text))
                ai_memory[channel_id].append(("assistant", reply))
            return reply or "hmm—say that again a different way?"
        except Exception as e:
            s = repr(e).lower()
            if "429" in s or "too many requests" in s or "rate" in s or "exhaust" in s:
                await asyncio.sleep(2 + attempt * 2)
                continue
            raise
    return "hit a rate limit. try again in a bit."


async def hf_image_bytes(prompt: str) -> bytes:
    if not hf:
        raise RuntimeError("HF not configured")

    last_err = None
    for attempt in range(5):
        try:
            img = await hf.text_to_image(prompt=prompt, model=HF_IMAGE_MODEL)

            if isinstance(img, (bytes, bytearray)):
                return bytes(img)

            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        except StopIteration as e:
            last_err = e
            await asyncio.sleep(2 + attempt * 2)
            continue

        except Exception as e:
            last_err = e
            s = repr(e).lower()
            if "429" in s or "too many requests" in s or "rate" in s or "busy" in s or "overload" in s:
                await asyncio.sleep(2 + attempt * 2)
                continue
            raise

    raise RuntimeError(
        f"Image generation failed after retries: {repr(last_err)}")


# =========================
# Self roles (dropdown menus)
# =========================
def save_selfrole_menu(guild_id: int, channel_id: int, message_id: int, title: str, roles: list[discord.Role]):
    conn = db()
    conn.execute(
        "INSERT OR REPLACE INTO selfrole_menus (guild_id, channel_id, message_id, title) VALUES (?, ?, ?, ?)",
        (guild_id, channel_id, message_id, title)
    )
    conn.execute("DELETE FROM selfrole_items WHERE guild_id=? AND message_id=?",
                 (guild_id, message_id))
    for r in roles[:SELFROLE_MAX_OPTIONS]:
        conn.execute(
            "INSERT OR REPLACE INTO selfrole_items (guild_id, message_id, role_id, label) VALUES (?, ?, ?, ?)",
            (guild_id, message_id, r.id, r.name)
        )
    conn.commit()
    conn.close()


def load_selfrole_menus():
    conn = db()
    rows = conn.execute(
        "SELECT guild_id, channel_id, message_id, title FROM selfrole_menus").fetchall()
    conn.close()
    return [(int(a), int(b), int(c), str(d)) for (a, b, c, d) in rows]


def load_selfrole_items(guild_id: int, message_id: int):
    conn = db()
    rows = conn.execute(
        "SELECT role_id, label FROM selfrole_items WHERE guild_id=? AND message_id=? ORDER BY label ASC",
        (guild_id, message_id)
    ).fetchall()
    conn.close()
    return [(int(rid), str(lbl)) for (rid, lbl) in rows]


class SelfRoleView(discord.ui.View):
    def __init__(self, guild_id: int, message_id: int, title: str, items: list[tuple[int, str]]):
        super().__init__(timeout=None)
        self.guild_id = guild_id
        self.message_id = message_id
        self.title = title

        options = [
            discord.SelectOption(label=label[:100], value=str(role_id))
            for role_id, label in items[:SELFROLE_MAX_OPTIONS]
        ]

        select = discord.ui.Select(
            placeholder="Pick roles (toggle on/off)…",
            min_values=0,
            max_values=min(len(options), 25),
            options=options,
            custom_id=f"selfroles:{guild_id}:{message_id}",
        )
        select.callback = self.on_select  # type: ignore
        self.add_item(select)

    async def on_select(self, interaction: discord.Interaction):
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return

        member: discord.Member = interaction.user
        selected = {int(v) for v in (
            interaction.data.get("values") or [])}  # type: ignore

        items = load_selfrole_items(interaction.guild.id, self.message_id)
        menu_role_ids = [rid for (rid, _lbl) in items]

        to_add = []
        to_remove = []

        for rid in menu_role_ids:
            role = interaction.guild.get_role(rid)
            if not role:
                continue
            if rid in selected and role not in member.roles:
                to_add.append(role)
            if rid not in selected and role in member.roles:
                to_remove.append(role)

        try:
            if to_add:
                await member.add_roles(*to_add, reason="Self-roles selection")
            if to_remove:
                await member.remove_roles(*to_remove, reason="Self-roles selection")
        except discord.Forbidden:
            await interaction.response.send_message(
                "I can't edit roles (missing permissions / role hierarchy).",
                ephemeral=True,
            )
            return
        except discord.HTTPException:
            await interaction.response.send_message(
                "Discord API error while updating roles.", ephemeral=True
            )
            return

        added = ", ".join(r.name for r in to_add) or "none"
        removed = ", ".join(r.name for r in to_remove) or "none"
        await interaction.response.send_message(
            f"✅ Updated roles.\nAdded: **{added}**\nRemoved: **{removed}**",
            ephemeral=True,
        )


# =========================
# Events
# =========================
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")

    for g in bot.guilds:
        await refresh_invites_for_guild(g)
    print("Invite cache initialized.")

    # Restore self-role dropdown menus (persistent views)
    try:
        restored = 0
        for guild_id, _channel_id, message_id, title in load_selfrole_menus():
            items = load_selfrole_items(guild_id, message_id)
            if not items:
                continue
            bot.add_view(SelfRoleView(guild_id, message_id, title, items))
            restored += 1
        if restored:
            print(f"Restored {restored} self-role menu view(s).")
    except Exception as e:
        print("Selfrole restore error:", repr(e))


@bot.event
async def on_invite_create(invite: discord.Invite):
    if invite.guild:
        await refresh_invites_for_guild(invite.guild)


@bot.event
async def on_invite_delete(invite: discord.Invite):
    if invite.guild:
        await refresh_invites_for_guild(invite.guild)


@bot.event
async def on_member_join(member: discord.Member):
    guild = member.guild
    welcome_ch = get_text_channel(guild, WELCOME_CHANNEL_ID)

    # Auto-role
    role_status = "Skipped (AUTO_ROLE_ID not set)"
    if AUTO_ROLE_ID:
        role = guild.get_role(AUTO_ROLE_ID)
        if not role:
            role_status = "Failed (role not found)"
        else:
            try:
                await member.add_roles(role, reason="Auto-role on join")
                role_status = f"Assigned {role.name}"
            except discord.Forbidden:
                role_status = "Failed (missing Manage Roles / role hierarchy)"
            except discord.HTTPException:
                role_status = "Failed (HTTP error)"

    # Invite tracking
    used_code, used_inv = await detect_used_invite(guild)
    inviter_text = "Unknown"
    invite_code_text = "Unknown"
    total_invites_text = "Unknown"

    if used_inv and used_inv.inviter:
        inviter = used_inv.inviter
        inviter_text = f"{inviter} ({inviter.mention})"
        invite_code_text = used_inv.code
        total_invites_text = str(inc_invite_joins(guild.id, inviter.id))

        await modlog(guild, "Member Joined (Invite Tracked)", [
            ("Member", fmt_user(member)),
            ("Invited By", fmt_user(inviter)),
            ("Invite Code", invite_code_text),
            ("Total Invites (tracked)", total_invites_text),
            ("Auto-role", role_status),
        ], 0x57F287)
    else:
        await modlog(guild, "Member Joined", [
            ("Member", fmt_user(member)),
            ("Invite", "Could not determine (missing perms/vanity/offline/cache)"),
            ("Auto-role", role_status),
        ], 0x57F287)

    if welcome_ch:
        await welcome_ch.send(
            f"Welcome {member.mention} 👋\n"
            f"Invited by: **{inviter_text}**\n"
            f"Invite code: `{invite_code_text}` • Total invites (tracked): **{total_invites_text}**\n"
            f"Role: **{role_status}**"
        )


@bot.event
async def on_member_remove(member: discord.Member):
    guild = member.guild
    leave_ch = get_text_channel(guild, LEAVE_CHANNEL_ID) or get_text_channel(
        guild, WELCOME_CHANNEL_ID)
    if leave_ch:
        await leave_ch.send(f"{member.name} left the server.")
    await modlog(guild, "Member Left", [("Member", fmt_user(member))], 0xED4245)


@bot.event
async def on_message_delete(message: discord.Message):
    if message.author and message.author.bot:
        return
    if not message.guild:
        return

    log_ch = get_text_channel(message.guild, DELETED_LOG_CHANNEL_ID)
    if not log_ch:
        return

    content = message.content if message.content else "(content not available — not cached)"
    author = message.author

    fields = [
        ("User", f"{author} ({author.id})" if author else "Unknown"),
        ("Channel",
         f"#{getattr(message.channel, 'name', 'unknown')} ({message.channel.id})"),
        ("Content", content[:1000]),
    ]
    if message.attachments:
        fields.append(("Attachments", "\n".join(
            a.url for a in message.attachments)[:1000]))

    await send_embed(log_ch, "🗑️ Message Deleted", fields, 0xED4245)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Commands still work
    await bot.process_commands(message)

    # If someone says "root" -> show commands
    msg = (message.content or "").strip().lower()
    if msg in ("root", "root?", "root help", "root commands", "root cmd", "root cmds", "root help me"):
        await message.channel.send(commands_text(PREFIX))
        return

    # AI respond only on mention or "root:" prefix
    if not should_ai_respond(message):
        return

    # cooldown per user
    now = time.time()
    uid = message.author.id
    if uid in _last_ai_call and (now - _last_ai_call[uid]) < AI_COOLDOWN_S:
        return
    _last_ai_call[uid] = now

    user_text = extract_ai_text(message)
    if not user_text:
        return

    async with message.channel.typing():
        try:
            reply = await hf_chat(message.channel.id, user_text)
            for part in split_for_discord(reply):
                await message.channel.send(part)
        except Exception as e:
            print("AI error:", repr(e))
            await message.channel.send("hit an API error. try again in a bit.")


# =========================
# Commands
# =========================
@bot.command(name="commands", aliases=["helpme", "cmds"])
async def commands_cmd(ctx: commands.Context):
    await ctx.reply(commands_text(PREFIX))


@bot.command(name="owner")
async def owner_cmd(ctx: commands.Context):
    if OWNER_ID:
        await ctx.reply(f"👑 My owner is <@{OWNER_ID}>.")
    else:
        await ctx.reply("Owner isn't set. Add `OWNER_ID=` in .env.")


@bot.command(name="echo")
@commands.has_permissions(manage_messages=True)
async def echo(ctx: commands.Context, member: discord.Member, *, message: str):
    """
    Usage:
      !echo @user Hey
    Sends:
      @user Hey
    """
    message = (message or "").strip()
    if not message:
        await ctx.reply(f"usage: `{PREFIX}echo @user your message`")
        return

    # delete the command message to keep chat clean (optional)
    try:
        await ctx.message.delete()
    except Exception:
        pass

    await ctx.send(f"{member.mention} {message}")

    if ctx.guild:
        await modlog(ctx.guild, "Echo", [
            ("Moderator", fmt_user(ctx.author)),
            ("Target", fmt_user(member)),
            ("Message", message[:900]),
            ("Channel", ctx.channel.mention),
        ], 0x5865F2)


@bot.command(name="img")
async def img(ctx: commands.Context, *, prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        await ctx.reply(f"usage: `{PREFIX}img your prompt here`")
        return
    if not hf:
        await ctx.reply("Image gen isn't configured. Add HF_TOKEN in .env.")
        return

    now = time.time()
    uid = ctx.author.id
    if uid in _last_img_call and (now - _last_img_call[uid]) < IMG_COOLDOWN_S:
        wait = int(IMG_COOLDOWN_S - (now - _last_img_call[uid]))
        await ctx.reply(f"slow down — try again in {wait}s.")
        return
    _last_img_call[uid] = now

    async with ctx.typing():
        try:
            img_bytes = await hf_image_bytes(prompt)
            bio = BytesIO(img_bytes)
            bio.seek(0)
            await ctx.reply(
                content=f"prompt: {prompt}",
                file=discord.File(fp=bio, filename="image.png"),
            )
        except Exception as e:
            print("HF image error:", repr(e))
            await ctx.reply("hit an error generating that image (rate limit / model busy / model gated). try again in a bit.")


# =========================
# Self roles commands
# =========================
@bot.group(name="selfroles", invoke_without_command=True)
async def selfroles(ctx: commands.Context):
    await ctx.reply(
        "**Self roles**\n"
        f"- `{PREFIX}selfrolesetup #channel @Role1 @Role2 ...` → create a dropdown\n"
        f"- `{PREFIX}selfroleslist` → list menus\n"
        f"- `{PREFIX}selfrolesremove <message_id>` → remove a menu\n\n"
        "Users pick roles using the dropdown (no commands needed for them)."
    )


@bot.command(name="selfrolesetup")
@commands.has_permissions(administrator=True)
async def selfrolesetup(ctx: commands.Context, channel: discord.TextChannel, *roles: discord.Role):
    roles = list(roles)
    if not roles:
        await ctx.reply(f"usage: `{PREFIX}selfrolesetup #channel @Role1 @Role2 ...`")
        return

    if len(roles) > SELFROLE_MAX_OPTIONS:
        roles = roles[:SELFROLE_MAX_OPTIONS]

    title = "Choose your roles"
    msg = await channel.send(f"**{title}**\nUse the menu below to toggle roles.")

    save_selfrole_menu(ctx.guild.id, channel.id, msg.id, title, roles)
    items = [(r.id, r.name) for r in roles]
    view = SelfRoleView(ctx.guild.id, msg.id, title, items)
    await msg.edit(view=view)

    bot.add_view(view)
    await ctx.reply(f"✅ Self-role menu created in {channel.mention} (message_id: `{msg.id}`).")


@bot.command(name="selfroleslist")
@commands.has_permissions(administrator=True)
async def selfroleslist(ctx: commands.Context):
    conn = db()
    rows = conn.execute(
        "SELECT channel_id, message_id, title FROM selfrole_menus WHERE guild_id=? ORDER BY message_id DESC LIMIT 15",
        (ctx.guild.id,)
    ).fetchall()
    conn.close()

    if not rows:
        await ctx.reply("No self-role menus found.")
        return

    lines = [
        f"- `{mid}` in <#{int(cid)}> — **{title}**" for (cid, mid, title) in rows]
    await ctx.reply("**Self-role menus:**\n" + "\n".join(lines))


@bot.command(name="selfrolesremove")
@commands.has_permissions(administrator=True)
async def selfrolesremove(ctx: commands.Context, message_id: int):
    conn = db()
    row = conn.execute(
        "SELECT channel_id FROM selfrole_menus WHERE guild_id=? AND message_id=?",
        (ctx.guild.id, message_id)
    ).fetchone()

    if not row:
        conn.close()
        await ctx.reply("❌ I can't find a self-role menu with that message_id.")
        return

    channel_id = int(row[0])
    conn.execute("DELETE FROM selfrole_items WHERE guild_id=? AND message_id=?",
                 (ctx.guild.id, message_id))
    conn.execute("DELETE FROM selfrole_menus WHERE guild_id=? AND message_id=?",
                 (ctx.guild.id, message_id))
    conn.commit()
    conn.close()

    # Try removing the dropdown from the original message
    ch = ctx.guild.get_channel(channel_id)
    if isinstance(ch, discord.TextChannel):
        try:
            m = await ch.fetch_message(message_id)
            await m.edit(view=None)
        except Exception:
            pass

    await ctx.reply("✅ Self-role menu removed.")


# =========================
# Moderation commands
# =========================
@bot.command()
@commands.has_permissions(kick_members=True)
async def kick(ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided"):
    await member.kick(reason=f"{ctx.author}: {reason}")
    await ctx.reply(f"✅ Kicked {member.mention}.")
    await modlog(ctx.guild, "Kick", [
        ("User", fmt_user(member)),
        ("Moderator", fmt_user(ctx.author)),
        ("Reason", reason),
        ("Channel", ctx.channel.mention),
    ], 0xF0B429)


@bot.command()
@commands.has_permissions(ban_members=True)
async def ban(ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided"):
    try:
        await member.ban(reason=f"{ctx.author}: {reason}", delete_message_seconds=0)
    except TypeError:
        await member.ban(reason=f"{ctx.author}: {reason}", delete_message_days=0)

    await ctx.reply(f"✅ Banned {member.mention}.")
    await modlog(ctx.guild, "Ban", [
        ("User", fmt_user(member)),
        ("Moderator", fmt_user(ctx.author)),
        ("Reason", reason),
        ("Channel", ctx.channel.mention),
    ], 0xED4245)


@bot.command()
@commands.has_permissions(ban_members=True)
async def unban(ctx: commands.Context, *, user_tag: str):
    bans = [entry async for entry in ctx.guild.bans(limit=2000)]
    target = None
    for entry in bans:
        if str(entry.user).lower() == user_tag.lower():
            target = entry.user
            break
    if not target:
        await ctx.reply("❌ Couldn't find that user in bans. Use `name#1234` exactly.")
        return
    await ctx.guild.unban(target, reason=f"{ctx.author}: unban")
    await ctx.reply(f"✅ Unbanned {target}.")
    await modlog(ctx.guild, "Unban", [
        ("User", fmt_user(target)),
        ("Moderator", fmt_user(ctx.author)),
        ("Channel", ctx.channel.mention),
    ], 0x57F287)


@bot.command(name="mute")
@commands.has_permissions(moderate_members=True)
async def mute(ctx: commands.Context, member: discord.Member, duration: str, *, reason: str = "No reason provided"):
    delta = parse_duration(duration)
    if not delta:
        await ctx.reply("❌ Invalid duration. Use like `10m`, `2h`, `1d`.")
        return
    until = utc_now() + delta
    await member.timeout(until, reason=f"{ctx.author}: {reason}")
    await ctx.reply(f"✅ Timed out {member.mention} for `{duration}`.")
    await modlog(ctx.guild, "Timeout (Mute)", [
        ("User", fmt_user(member)),
        ("Moderator", fmt_user(ctx.author)),
        ("Duration", duration),
        ("Reason", reason),
        ("Channel", ctx.channel.mention),
    ], 0xFAA61A)


@bot.command(name="unmute")
@commands.has_permissions(moderate_members=True)
async def unmute(ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided"):
    await member.timeout(None, reason=f"{ctx.author}: {reason}")
    await ctx.reply(f"✅ Removed timeout from {member.mention}.")
    await modlog(ctx.guild, "Timeout Removed", [
        ("User", fmt_user(member)),
        ("Moderator", fmt_user(ctx.author)),
        ("Reason", reason),
        ("Channel", ctx.channel.mention),
    ], 0x57F287)


@bot.command()
@commands.has_permissions(manage_messages=True)
async def purge(ctx: commands.Context, amount: int = 5):
    if amount < 1 or amount > 200:
        await ctx.reply("❌ Amount must be between 1 and 200.")
        return
    deleted = await ctx.channel.purge(limit=amount + 1)
    await ctx.send(f"🧹 Deleted {max(len(deleted)-1, 0)} messages.", delete_after=3)
    await modlog(ctx.guild, "Purge", [
        ("Moderator", fmt_user(ctx.author)),
        ("Channel", ctx.channel.mention),
        ("Amount", str(amount)),
    ], 0x5865F2)


@bot.command(name="purgeuser", aliases=["purgeu", "clearuser"])
@commands.has_permissions(manage_messages=True)
async def purgeuser(ctx: commands.Context, member: discord.Member, amount: int = 20):
    if amount < 1 or amount > 200:
        await ctx.reply("❌ Amount must be between 1 and 200.")
        return

    def check(m: discord.Message) -> bool:
        return m.author.id == member.id

    deleted = await ctx.channel.purge(limit=amount + 50, check=check)
    await ctx.send(f"🧹 Deleted {len(deleted)} messages from {member.mention}.", delete_after=4)
    await modlog(ctx.guild, "Purge User", [
        ("Moderator", fmt_user(ctx.author)),
        ("Target", fmt_user(member)),
        ("Channel", ctx.channel.mention),
        ("Deleted", str(len(deleted))),
    ], 0x5865F2)


@bot.command()
@commands.has_permissions(moderate_members=True)
async def warn(ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided"):
    conn = db()
    conn.execute(
        "INSERT INTO warnings (guild_id, user_id, moderator_id, reason, created_at) VALUES (?, ?, ?, ?, ?)",
        (ctx.guild.id, member.id, ctx.author.id, reason, utc_now().isoformat())
    )
    conn.commit()
    conn.close()

    await ctx.reply(f"⚠️ Warned {member.mention}: {reason}")
    await modlog(ctx.guild, "Warn", [
        ("User", fmt_user(member)),
        ("Moderator", fmt_user(ctx.author)),
        ("Reason", reason),
        ("Channel", ctx.channel.mention),
    ], 0xFEE75C)


@bot.command()
@commands.has_permissions(moderate_members=True)
async def warnings(ctx: commands.Context, member: discord.Member):
    conn = db()
    rows = conn.execute(
        "SELECT id, moderator_id, reason, created_at FROM warnings WHERE guild_id=? AND user_id=? ORDER BY id DESC LIMIT 10",
        (ctx.guild.id, member.id)
    ).fetchall()
    conn.close()

    if not rows:
        await ctx.reply(f"{member.mention} has no warnings.")
        return

    lines = []
    for wid, mod_id, reason, created_at in rows:
        lines.append(
            f"**#{wid}** • <@{mod_id}> • {created_at.split('T')[0]} • {reason}")

    embed = discord.Embed(
        title=f"Warnings for {member}", description="\n".join(lines), color=0xFEE75C)
    await ctx.reply(embed=embed)


@bot.command()
@commands.has_permissions(administrator=True)
async def clearwarnings(ctx: commands.Context, member: discord.Member):
    conn = db()
    conn.execute("DELETE FROM warnings WHERE guild_id=? AND user_id=?",
                 (ctx.guild.id, member.id))
    conn.commit()
    conn.close()

    await ctx.reply(f"✅ Cleared warnings for {member.mention}.")
    await modlog(ctx.guild, "Clear Warnings", [
        ("User", fmt_user(member)),
        ("Moderator", fmt_user(ctx.author)),
        ("Channel", ctx.channel.mention),
    ], 0x57F287)


# =========================
# Setup commands (admin)
# =========================
@bot.command()
@commands.has_permissions(administrator=True)
async def setwelcome(ctx: commands.Context, channel: discord.TextChannel):
    global WELCOME_CHANNEL_ID
    WELCOME_CHANNEL_ID = channel.id
    await ctx.reply(f"Welcome channel set to {channel.mention}. Put in .env:\n`WELCOME_CHANNEL_ID={channel.id}`")


@bot.command()
@commands.has_permissions(administrator=True)
async def setleave(ctx: commands.Context, channel: discord.TextChannel):
    global LEAVE_CHANNEL_ID
    LEAVE_CHANNEL_ID = channel.id
    await ctx.reply(f"Leave channel set to {channel.mention}. Put in .env:\n`LEAVE_CHANNEL_ID={channel.id}`")


@bot.command()
@commands.has_permissions(administrator=True)
async def setmodlog(ctx: commands.Context, channel: discord.TextChannel):
    global MODLOG_CHANNEL_ID
    MODLOG_CHANNEL_ID = channel.id
    await ctx.reply(f"Modlog channel set to {channel.mention}. Put in .env:\n`MODLOG_CHANNEL_ID={channel.id}`")


@bot.command()
@commands.has_permissions(administrator=True)
async def setdeletedlog(ctx: commands.Context, channel: discord.TextChannel):
    global DELETED_LOG_CHANNEL_ID
    DELETED_LOG_CHANNEL_ID = channel.id
    await ctx.reply(f"Deleted-log channel set to {channel.mention}. Put in .env:\n`DELETED_LOG_CHANNEL_ID={channel.id}`")


@bot.command()
@commands.has_permissions(administrator=True)
async def setautorole(ctx: commands.Context, role: discord.Role):
    global AUTO_ROLE_ID
    AUTO_ROLE_ID = role.id
    await ctx.reply(f"Auto-role set to **{role.name}**. Put in .env:\n`AUTO_ROLE_ID={role.id}`")


# =========================
# START BOT
# =========================
bot.run(TOKEN)
