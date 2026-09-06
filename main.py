"""Pawwake application assembly and process entrypoint.

Module map:
  shared.py              runtime configuration, prompt cache, templates
  auth.py                header and Dashboard cookie authentication
  partition_engine.py    partitioned context assembly and summaries
  memory_pipeline.py     memory injection, recall, and extraction
  routes/chat.py         health and OpenAI-compatible chat
  routes/dashboard.py    Dashboard pages, models, and settings
  routes/memories.py     memory CRUD, import, consolidation, and backfill
  routes/conversations.py conversation management and recall
  routes/partition.py    partition state and thread management
  db/*                   PostgreSQL core, search, conversation, memory domains
"""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

import auth
import shared
from db import core as db_core
from db import search as db_search
from db import conversations as db_conversations
from db import memories as db_memories
from routes.chat import router as chat_router
from routes.dashboard import router as dashboard_router, settings_router
from routes.memories import bootstrap_router as memories_bootstrap_router
from routes.memories import maintenance_router as memories_maintenance_router
from routes.memories import router as memories_router
from routes.conversations import router as conversations_router
from routes.partition import router as partition_router

# ============================================================
# 应用生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化持久化配置；记忆关闭时仍保留 Dashboard 与对话线状态。"""
    if "MEMORY_EXTRACT_ENABLED" in os.environ:
        print(
            "⚠️  MEMORY_EXTRACT_ENABLED 已移除并被忽略；"
            "请改用 MEMORY_EXTRACT_INTERVAL=0 停止自动提取，"
            "MAX_MEMORIES_INJECT=0 停止自动注入"
        )
    if not shared.DATABASE_ENABLED:
        print("ℹ️  数据库总闸已关闭；跳过数据库初始化并进入纯转发模式")
        yield
        return

    try:
        await db_core.init_tables()
        await db_conversations.ensure_token_usage_table()

        # 从数据库恢复面板配置。这一步不能受 MEMORY_ENABLED 控制，
        # 否则 Dashboard 写入 false 后将无法重新开启记忆。
        try:
            db_cfg = await db_core.get_all_gateway_config()
            if db_cfg:
                # 显式空值也要恢复，否则重启时环境默认会覆盖面板上的清空操作。
                restored = []
                for key, val in db_cfg.items():
                    if key in shared.SETTINGS_TYPES:
                        if not val and key not in shared.SETTINGS_ALLOW_EMPTY:
                            continue
                        setattr(shared, key, shared.SETTINGS_TYPES[key](val or ""))
                        restored.append(key + ("(显式空)" if not val else ""))
                    elif key == "MEMORY_MODEL":
                        if not val:
                            os.environ["MEMORY_MODEL"] = ""
                            restored.append(key + "(显式空)")
                        else:
                            os.environ["MEMORY_MODEL"] = str(val)
                            restored.append(key)
                if restored:
                    shared.sync_memory_extractor_config()
                    print(f"🔄 从数据库恢复 {len(restored)} 项面板配置: {', '.join(restored)}")
        except Exception as e:
            print(f"[warning] 恢复面板配置失败: {e}")

        if shared.MEMORY_ENABLED:
            count = await db_memories.get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")
        else:
            print("ℹ️  记忆系统已关闭；Dashboard 配置与对话线状态仍可用")

        if shared.CONVERSATION_RECALL_ENABLED:
            updated_tsv = await db_search.rebuild_content_tsv()
            started = db_search.kick_embedding_backfill()
            print(
                f"🔎 对话召回已启动: TSV补齐{updated_tsv}条, "
                f"向量补算={'已唤醒' if started else '无需重复唤醒'}"
            )

        # 活跃对话线属于分区缓存状态，必须独立于记忆开关恢复。
        db_sid = await db_core.get_gateway_config("partition_session_id", "")
        if db_sid:
            shared.PARTITION_SESSION_ID = db_sid
            print(f"🔗 活跃对话线(DB): {shared.PARTITION_SESSION_ID}")
        elif shared.PARTITION_SESSION_ID:
            await db_core.set_gateway_config("partition_session_id", shared.PARTITION_SESSION_ID)
            print(f"🔗 活跃对话线(ENV→DB): {shared.PARTITION_SESSION_ID}")
        if shared.CACHE_PARTITION_ENABLED:
            print(f"🔒 分区缓存已启用: X={shared.CACHE_PARTITION_X}, 摘要模型={shared.CACHE_SUMMARY_MODEL or '（未配置，纯轮转模式）'}")
    except Exception as e:
        print(f"⚠️  数据库初始化失败: {e}")
        if shared.CACHE_PARTITION_ENABLED:
            print("⚠️  分区缓存需要数据库；聊天请求将返回 503，避免丢失历史")
        else:
            print("⚠️  持久化配置与记忆能力暂停；网关继续纯转发")
    
    yield
    
    await db_core.close_pool()




app = FastAPI(title="Pawwake", version="4.1.1", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.middleware("http")(auth.gateway_auth_middleware)

# Centralized router registration. Preserve this order in route snapshots.
app.include_router(chat_router)
database_dependencies = [Depends(auth.require_database_enabled)]
app.include_router(memories_bootstrap_router, dependencies=database_dependencies)
app.include_router(dashboard_router)
app.include_router(memories_router, dependencies=database_dependencies)
app.include_router(conversations_router, dependencies=database_dependencies)
app.include_router(partition_router, dependencies=database_dependencies)
app.include_router(memories_maintenance_router, dependencies=database_dependencies)
app.include_router(settings_router)

# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Pawwake 启动中... 端口 {shared.PORT}")
    print(f"📝 人设长度：{len(shared.SYSTEM_PROMPT)} 字符")
    print(f"🤖 默认模型：{shared.DEFAULT_MODEL}")
    print(f"🔗 API 地址：{shared.API_BASE_URL}")
    print(f"🗄️  数据库总闸：{'开启' if shared.DATABASE_ENABLED else '关闭'}")
    print(f"💬 对话持久化：{'开启' if shared.conversation_persistence_enabled() else '关闭'}")
    print(f"🧠 记忆系统：{'开启' if shared.MEMORY_ENABLED else '关闭'}")
    if shared.MEMORY_ENABLED:
        print(f"📚 记忆自动注入：{'禁用' if shared.MAX_MEMORIES_INJECT <= 0 else f'每轮最多 {shared.MAX_MEMORIES_INJECT} 条'}")
    print(f"🔄 记忆提取间隔：{'禁用' if shared.MEMORY_EXTRACT_INTERVAL == 0 else '每轮提取' if shared.MEMORY_EXTRACT_INTERVAL == 1 else f'每 {shared.MEMORY_EXTRACT_INTERVAL} 轮提取一次'}")
    if shared.CACHE_PARTITION_ENABLED:
        print(f"🔒 分区缓存：开启 (X={shared.CACHE_PARTITION_X}, session={shared.PARTITION_SESSION_ID or '未设置'})")
    if shared.FORCE_STREAM:
        print(f"⚡ 强制流式传输：开启")
    if shared.REASONING_EFFORT:
        print(f"🧠 推理参数注入：{shared.REASONING_EFFORT}")
    uvicorn.run(app, host="0.0.0.0", port=shared.PORT)
