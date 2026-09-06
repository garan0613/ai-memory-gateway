"""
数据库模块 —— 负责所有跟 PostgreSQL 打交道的事情
==============================================
包括：
- 创建表结构
- 存储对话记录
- 存储/检索记忆（带中文分词和加权排序）
"""

import os
import re
import json
import logging
from typing import Optional, List
from datetime import datetime, date, timedelta, timezone as dt_timezone

import asyncpg

import shared

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

HAS_PGVECTOR = False  # 在init_tables时检测


# ============================================================
# 连接池管理
# ============================================================

_pool: Optional[asyncpg.Pool] = None


class BrokenMergeReferencesError(ValueError):
    """备份前发现 merged_from 引用了不存在的记忆。"""

    def __init__(self, count: int):
        self.count = count
        super().__init__(
            f"检测到 {count} 条记忆的合并来源已失效，可修复断裂引用后重新导出"
        )


class BrokenSupersessionReferencesError(ValueError):
    """备份前发现 superseded_by 指向不存在的后继记忆。"""

    def __init__(self, count: int):
        self.count = count
        super().__init__(f"检测到 {count} 条记忆的版本后继已失效，无法导出完整备份")


class DatabaseDisabled(RuntimeError):
    """数据库总闸关闭时拒绝创建或返回连接池。"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if not shared.DATABASE_ENABLED:
        raise DatabaseDisabled("DATABASE_ENABLED=false，数据库连接已停用")
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL 未设置！")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, statement_cache_size=0)
        print("✅ 数据库连接池已创建")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ 数据库连接池已关闭")


# ============================================================
# 表结构初始化
# ============================================================

async def init_tables():
    global HAS_PGVECTOR
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT,
                model           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                metadata        TEXT
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              SERIAL PRIMARY KEY,
                content         TEXT NOT NULL,
                importance      INTEGER DEFAULT 5,
                source_session  TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                last_accessed   TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_fts
            ON memories
            USING gin(to_tsvector('simple', content));
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations (session_id, created_at);
        """)

        # 工具调用支持：加 metadata 字段（已有表自动迁移）
        await conn.execute("""
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS metadata TEXT;
        """)

        # content 允许 NULL（工具调用时 assistant 的 content 可能为空）
        await conn.execute("""
            ALTER TABLE conversations ALTER COLUMN content DROP NOT NULL;
        """)

        # 原始对话召回索引。NULL 同时作为可恢复 backfill 的持久账本。
        await conn.execute("""
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_content_tsv
            ON conversations USING GIN (content_tsv);
        """)

        # 网关配置表（存储运行时可变配置）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS gateway_config (
                key     TEXT PRIMARY KEY,
                value   TEXT DEFAULT ''
            );
        """)

        # 分区缓存状态表（存储每个session的轮转状态）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS session_cache_state (
                session_id      TEXT PRIMARY KEY,
                summary         TEXT DEFAULT '',
                a_start_round   INTEGER DEFAULT 0,
                seen_fragment_ids TEXT[] DEFAULT '{}',
                seen_fragment_times JSONB DEFAULT '{}'::jsonb,
                seen_memory_times JSONB DEFAULT '{}'::jsonb,
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS seen_fragment_ids TEXT[] DEFAULT '{}';
        """)
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS seen_fragment_times JSONB DEFAULT '{}'::jsonb;
        """)
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS seen_memory_times JSONB DEFAULT '{}'::jsonb;
        """)
        await conn.execute("""
            UPDATE session_cache_state AS scs
            SET seen_fragment_times = (
                SELECT COALESCE(
                    jsonb_object_agg(fragment_id, to_jsonb(scs.updated_at)),
                    '{}'::jsonb
                )
                FROM unnest(COALESCE(scs.seen_fragment_ids, '{}'::text[])) AS fragment_id
            )
            WHERE COALESCE(scs.seen_fragment_times, '{}'::jsonb) = '{}'::jsonb
              AND cardinality(COALESCE(scs.seen_fragment_ids, '{}'::text[])) > 0;
        """)

        # ---- 三层记忆架构字段（layer / title / is_active / merged_from / event_date）----
        # layer: 1=原始碎片, 2=事件记忆, 3=核心记忆
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS layer INTEGER DEFAULT 1;
        """)

        # title: 记忆标题（语义锚点，用于搜索加权）
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS title TEXT DEFAULT NULL;
        """)

        # is_active: 是否参与搜索（碎片合并后变为 false）
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;
        """)

        # merged_from: 合并来源的碎片ID列表
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS merged_from INTEGER[] DEFAULT NULL;
        """)

        # event_date: 事件日期（用于按天整理）
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS event_date DATE DEFAULT NULL;
        """)

        # external_id: 调用方提供的稳定幂等键
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS external_id TEXT;
        """)
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_external_id
            ON memories (external_id)
            WHERE external_id IS NOT NULL;
        """)

        # superseded_by: 自动冲突接管后的后继记忆
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS superseded_by INTEGER;
        """)
        await conn.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_memories_superseded_by'
                      AND conrelid = 'memories'::regclass
                ) THEN
                    ALTER TABLE memories
                    ADD CONSTRAINT fk_memories_superseded_by
                    FOREIGN KEY (superseded_by) REFERENCES memories(id);
                END IF;
            END $$;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_superseded_by
            ON memories (superseded_by)
            WHERE superseded_by IS NOT NULL;
        """)

        # 三层记忆索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories (layer);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_active ON memories (is_active);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_event_date ON memories (event_date);
        """)

        # 尝试启用pgvector扩展（向量搜索）
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            HAS_PGVECTOR = True
            print("✅ pgvector扩展已启用")

            # 对话表向量列
            await conn.execute(f"""
                ALTER TABLE conversations
                ADD COLUMN IF NOT EXISTS embedding vector({shared.EMBEDDING_DIM});
            """)

            # 记忆表向量列
            await conn.execute(f"""
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS embedding vector({shared.EMBEDDING_DIM});
            """)
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding
                    ON memories USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10);
                """)
            except Exception:
                pass  # ivfflat需要一定行数才能建索引，初期跳过
        except Exception as e:
            HAS_PGVECTOR = False
            print(f"⚠️ pgvector不可用（{e}），向量搜索将使用Python端计算")

            # 回退：用TEXT列存JSON格式的向量
            await conn.execute("""
                ALTER TABLE conversations ADD COLUMN IF NOT EXISTS embedding_json TEXT;
            """)
            await conn.execute("""
                ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_json TEXT;
            """)

    print("✅ 数据库表结构已就绪")


# ============================================================
# 网关配置
# ============================================================

async def get_gateway_config(key: str, default: str = "") -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT value FROM gateway_config WHERE key = $1", key)
        return row['value'] if row else default


async def set_gateway_config(key: str, value: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO gateway_config (key, value) VALUES ($1, $2)
            ON CONFLICT (key) DO UPDATE SET value = $2
        """, key, value)


async def get_all_gateway_config() -> dict:
    """获取所有配置项"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT key, value FROM gateway_config")
        return {r['key']: r['value'] for r in rows}
