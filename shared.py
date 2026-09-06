"""Shared configuration and mutable runtime state for Pawwake."""

import logging
import os

from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# ============================================================
# 配置项 —— 全部从环境变量读取，部署时在云平台面板里设置
# ============================================================

# 你的 API Key（OpenRouter / OpenAI / 其他兼容服务）
API_KEY = os.getenv("API_KEY", "")

# API 地址（改这个就能切换不同的 LLM 服务商）
# OpenRouter: https://openrouter.ai/api/v1/chat/completions
# OpenAI:     https://api.openai.com/v1/chat/completions
# 本地 Ollama: http://localhost:11434/v1/chat/completions
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 默认模型（如果客户端没指定就用这个）
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")

# 网关端口
PORT = int(os.getenv("PORT", "8080"))

# 网关访问密钥（强烈建议设置！）
# 设置后所有非公开 API 端点都需要 X-Gateway-Key 请求头。
# Dashboard 使用独立密码登录，不读取或保存网关密钥。
# 不设置则跳过 API 鉴权（兼容旧部署，仅建议内网环境使用）。
GATEWAY_SECRET = os.getenv("GATEWAY_SECRET", "")

# Dashboard 浏览器登录。SESSION_SECRET 需为稳定的 32 字符以上随机值，
# 用于签发 HttpOnly 会话 Cookie；修改它会让已有登录全部失效。
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
DASHBOARD_SESSION_SECONDS = 12 * 60 * 60
DASHBOARD_SESSION_COOKIE = "__Host-pawwake_dashboard_session"

# 数据库总开关只从部署环境读取。关闭时三个数据库能力开关统一失效，
# 恢复需修改环境变量并重启进程。
DATABASE_ENABLED = os.getenv("DATABASE_ENABLED", "true").lower() == "true"

# 记忆系统开关（只控制记忆能力；Dashboard 配置与分区缓存独立）
MEMORY_ENABLED = DATABASE_ENABLED and os.getenv("MEMORY_ENABLED", "false").lower() == "true"

# 每次注入的最大记忆条数
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))

# 同一记忆在分区 session 中的自动去重窗口。0 = 不保留 seen 去重状态。
MEMORY_SEEN_TTL_HOURS = max(
    0.0,
    float(os.getenv("MEMORY_SEEN_TTL_HOURS", "6")),
)

# 分区模式自动注入的最大历史对话片段数。raw API 可逐请求覆盖。
MAX_CONVERSATIONS_INJECT = int(os.getenv("MAX_CONVERSATIONS_INJECT", "3"))

# 同一历史片段在分区模式中的自动去重窗口。0 = 不保留 seen 去重状态。
CONVERSATION_SEEN_TTL_HOURS = max(
    0.0,
    float(os.getenv("CONVERSATION_SEEN_TTL_HOURS", "6")),
)

# 记忆提取间隔（0 = 禁用自动提取，1 = 每轮提取，N = 每 N 轮提取一次）
MEMORY_EXTRACT_INTERVAL = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "1"))

# 分区缓存
CACHE_PARTITION_ENABLED = DATABASE_ENABLED and os.getenv("CACHE_PARTITION_ENABLED", "false").lower() == "true"
CACHE_PARTITION_X = int(os.getenv("CACHE_PARTITION_X", "15"))
CACHE_SUMMARY_MODEL = os.getenv("CACHE_SUMMARY_MODEL", "")  # 留空=不生成摘要，轮转时A区直接滑出（纯轮转模式）
CACHE_SUMMARY_MAX_TOKENS = int(os.getenv("CACHE_SUMMARY_MAX_TOKENS", "2000"))  # 摘要输出上限，跟记忆提取的 MEMORY_MAX_TOKENS 各管各的。失败日志拿它当分母
CACHE_PARTITION_TRIGGER = os.getenv("CACHE_PARTITION_TRIGGER", "rounds")  # rounds=按轮次 | time=按时间窗口
CACHE_PARTITION_WINDOW = int(os.getenv("CACHE_PARTITION_WINDOW", "30"))  # 时间窗口（分钟），仅 trigger=time 时生效
CACHE_TTL = os.getenv("CACHE_TTL", "5m")  # 缓存TTL：5m(默认) | 1h。1h写入费2x(5m是1.25x)读都0.1x，消息间隔常超5分钟的慢聊场景1h更划算
PARTITION_SESSION_ID = os.getenv("PARTITION_SESSION_ID", "")


def make_cache_control() -> dict:
    """构造cache_control块。CACHE_TTL=1h时显式带ttl字段，其余值不带（上游默认5m）"""
    if CACHE_TTL == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}

def get_active_session_id() -> str:
    return PARTITION_SESSION_ID


def memory_injection_enabled() -> bool:
    return MEMORY_ENABLED and MAX_MEMORIES_INJECT > 0


def conversation_persistence_enabled() -> bool:
    """记忆、分区或对话召回任一开启时都需要保留历史。"""
    return DATABASE_ENABLED and (
        MEMORY_ENABLED or CACHE_PARTITION_ENABLED or CONVERSATION_RECALL_ENABLED
    )


def _api_failure(message: str, **extra) -> dict:
    """记录当前意外异常，对外只返回固定文案。"""
    logger.exception("API operation failed: %s", message)
    return {"error": message, **extra}

# 时区偏移（小时），用于记忆注入时的日期显示，默认 UTC+8
TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "8"))

# 数据库检索与向量能力同样属于可热更新运行态，统一从 shared 读取。
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))
MEMORY_VECTOR_ENABLED = os.getenv("MEMORY_VECTOR_ENABLED", "false").lower() == "true"
CONVERSATION_RECALL_ENABLED = DATABASE_ENABLED and os.getenv(
    "CONVERSATION_RECALL_ENABLED", "false"
).lower() == "true"
CONVERSATION_MIN_SCORE_THRESHOLD = float(
    os.getenv("CONVERSATION_MIN_SCORE_THRESHOLD", "0.7")
)
CONVERSATION_HW_KEYWORD = float(os.getenv("CONVERSATION_HW_KEYWORD", "0.45"))
CONVERSATION_HW_SEMANTIC = float(os.getenv("CONVERSATION_HW_SEMANTIC", "0.35"))
CONVERSATION_HW_RECENCY = float(os.getenv("CONVERSATION_HW_RECENCY", "0.2"))
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.5"))
WEIGHT_IMPORTANCE = float(os.getenv("WEIGHT_IMPORTANCE", "0.3"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.2"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.15"))
MEMORY_HW_KEYWORD = float(os.getenv("MEMORY_HW_KEYWORD", "0.35"))
MEMORY_HW_SEMANTIC = float(os.getenv("MEMORY_HW_SEMANTIC", "0.35"))
MEMORY_HW_IMPORTANCE = float(os.getenv("MEMORY_HW_IMPORTANCE", "0.15"))
MEMORY_HW_RECENCY = float(os.getenv("MEMORY_HW_RECENCY", "0.15"))
MEMORY_SEMANTIC_THRESHOLD = float(os.getenv("MEMORY_SEMANTIC_THRESHOLD", "0.5"))

# 非分区模式没有稳定 session 历史，保留进程内提取计数。
_nonpartition_round_counter = 0

# 强制流式传输（部分客户端不发stream=true导致thinking数据丢失，开启后强制所有请求走流式）
FORCE_STREAM = os.getenv("FORCE_STREAM", "false").lower() == "true"

# 推理/思维链参数（部分客户端走网关时不会自动添加reasoning参数，导致上游不返回thinking数据）
# 设为 low/medium/high 会在转发请求时注入 reasoning_effort 参数
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")

# 记忆模型专用 API Key（不设则回退到主 API_KEY）
# 适用于中转站按模型分组、不同模型需要不同 Key 的场景
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", "")

SETTINGS_TYPES = {
    "API_BASE_URL": str,
    "API_KEY": str,
    "DEFAULT_MODEL": str,
    "MEMORY_API_KEY": str,
    "MEMORY_ENABLED": lambda value: _parse_bool(value),
    "MAX_MEMORIES_INJECT": int,
    "MEMORY_SEEN_TTL_HOURS": lambda value: max(0.0, float(value)),
    "MAX_CONVERSATIONS_INJECT": int,
    "CONVERSATION_SEEN_TTL_HOURS": lambda value: max(0.0, float(value)),
    "MEMORY_EXTRACT_INTERVAL": int,
    "CACHE_PARTITION_ENABLED": lambda value: _parse_bool(value),
    "CACHE_PARTITION_X": int,
    "CACHE_PARTITION_TRIGGER": str,
    "CACHE_PARTITION_WINDOW": int,
    "CACHE_SUMMARY_MODEL": str,
    "CACHE_TTL": str,
    "FORCE_STREAM": lambda value: _parse_bool(value),
    "REASONING_EFFORT": str,
    "EMBEDDING_API_KEY": str,
    "EMBEDDING_BASE_URL": str,
    "EMBEDDING_MODEL": str,
    "EMBEDDING_DIM": int,
    "MIN_SCORE_THRESHOLD": float,
    "MEMORY_VECTOR_ENABLED": lambda value: _parse_bool(value),
    "CONVERSATION_RECALL_ENABLED": lambda value: _parse_bool(value),
    "CONVERSATION_MIN_SCORE_THRESHOLD": float,
    "CONVERSATION_HW_KEYWORD": float,
    "CONVERSATION_HW_SEMANTIC": float,
    "CONVERSATION_HW_RECENCY": float,
    "MEMORY_HW_KEYWORD": float,
    "MEMORY_HW_SEMANTIC": float,
    "MEMORY_HW_IMPORTANCE": float,
    "MEMORY_HW_RECENCY": float,
    "MEMORY_SEMANTIC_THRESHOLD": float,
}
SETTINGS_ALLOW_EMPTY = {"CACHE_SUMMARY_MODEL", "MEMORY_API_KEY"}


def get_memory_api_key() -> str:
    return MEMORY_API_KEY or API_KEY

def sync_memory_extractor_config():
    """把配置推给 memory_extractor：它在 import 时就把这几个读成了自己的模块级全局，
    之后改 os.environ 或 main 的 globals 都够不着，面板换了模型/换了 key，提取那边
    还拿启动时那份跑。数据库恢复完和面板保存完各调一次。
    MEMORY_MODEL 落回 DEFAULT_MEMORY_MODEL 而不是当前值：面板上这项写着"留空用默认"，
    清空写进的是空串，落回当前值会让热更新沿用旧模型、重启后却变默认，前后分裂。"""
    import memory_extractor as _me_mod
    _me_mod.API_KEY = API_KEY
    _me_mod.API_BASE_URL = API_BASE_URL
    _me_mod.MEMORY_API_KEY = MEMORY_API_KEY
    _me_mod.MEMORY_MODEL = os.environ.get("MEMORY_MODEL") or _me_mod.DEFAULT_MEMORY_MODEL

# 额外的请求头（有些 API 需要，比如 OpenRouter 需要 Referer）
EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://github.com/garan0613/pawwake")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "Pawwake")


# ============================================================
# 人设加载
# ============================================================

def load_system_prompt():
    """从 system_prompt.txt 文件读取人设内容"""
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    print("ℹ️  未找到 system_prompt.txt 或文件为空，将不注入 system prompt")
    return ""


SYSTEM_PROMPT = load_system_prompt()
_DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT  # 保留文件原始版本
if SYSTEM_PROMPT:
    print(f"✅ 人设已加载，长度：{len(SYSTEM_PROMPT)} 字符")
else:
    print("ℹ️  无人设，纯转发模式")

# System Prompt 缓存（支持设置面板热更新）
_cached_system_prompt = None
_cached_system_prompt_loaded = False

async def get_system_prompt() -> str:
    """获取 system prompt（数据库优先，fallback 到文件）"""
    global _cached_system_prompt, _cached_system_prompt_loaded
    if not DATABASE_ENABLED:
        return _DEFAULT_SYSTEM_PROMPT or ""
    if _cached_system_prompt_loaded:
        return _cached_system_prompt or ""
    try:
        from db import core as db_core
        db_prompt = await db_core.get_gateway_config("systemPrompt", "")
        if db_prompt:
            _cached_system_prompt = db_prompt
            prompt_source = "Dashboard DB"
        else:
            _cached_system_prompt = _DEFAULT_SYSTEM_PROMPT
            prompt_source = "system_prompt.txt 默认值"
        _cached_system_prompt_loaded = True
        print(f"📝 System Prompt 已解析: source={prompt_source}, length={len(_cached_system_prompt or '')}")
        return _cached_system_prompt or ""
    except Exception as e:
        # DB 短暂故障时用文件默认值完成当前请求，不缓存失败结果，
        # 让后续请求能在 DB 恢复后自动重试 Dashboard 人设。
        print(f"⚠️  读取 Dashboard System Prompt 失败，本次使用文件默认值: {e}")
        return _DEFAULT_SYSTEM_PROMPT or ""

def invalidate_system_prompt_cache():
    """清除 system prompt 缓存（设置面板更新后调用）"""
    global _cached_system_prompt, _cached_system_prompt_loaded
    _cached_system_prompt = None
    _cached_system_prompt_loaded = False




def _parse_bool(val, fallback=False) -> bool:
    """解析布尔值（兼容字符串/布尔/None）"""
    if val is None:
        return fallback
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


templates = Jinja2Templates(directory="templates")
