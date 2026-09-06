"""Conversation, partition state, token usage, and conversation exports."""

import json
from datetime import datetime, timedelta, timezone as dt_timezone

import shared
from db import core as db_core
from db import search as db_search

# ============================================================
# 对话记录操作
# ============================================================

async def save_message(session_id: str, role: str, content: str, model: str = "", metadata: str = None):
    tsv_text = (
        db_search.jieba_tokenize_for_tsv(content or "")
        if shared.CONVERSATION_RECALL_ENABLED
        else None
    )
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO conversations (
                   session_id, role, content, model, metadata, content_tsv
               ) VALUES (
                   $1, $2, $3, $4, $5,
                   array_to_tsvector(string_to_array($6, ' '))
               )
               RETURNING id""",
            session_id, role, content, model, metadata, tsv_text,
        )
    message_id = row["id"] if row else None
    if (
        message_id is not None
        and shared.CONVERSATION_RECALL_ENABLED
        and shared.EMBEDDING_API_KEY
        and content
        and content.strip()
    ):
        db_search.kick_embedding_backfill()
    return message_id


async def get_last_user_content(session_id: str) -> str:
    """获取指定session最后一条user消息的content"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT content FROM conversations
            WHERE session_id = $1 AND role = 'user'
            ORDER BY created_at DESC
            LIMIT 1
        """, session_id)
        return row['content'] if row else ""


async def update_last_assistant_message(session_id: str, new_content: str, model: str = ""):
    """覆盖指定session最后一条assistant消息的content（用于re-roll去重）"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM conversations
            WHERE session_id = $1 AND role = 'assistant'
            ORDER BY created_at DESC
            LIMIT 1
        """, session_id)
        if row:
            embedding_column = "embedding" if db_core.HAS_PGVECTOR else "embedding_json"
            tsv_text = (
                db_search.jieba_tokenize_for_tsv(new_content or "")
                if shared.CONVERSATION_RECALL_ENABLED
                else None
            )
            await conn.execute(
                f"""UPDATE conversations
                   SET content = $1,
                       model = $2,
                       content_tsv = array_to_tsvector(string_to_array($3, ' ')),
                       {embedding_column} = NULL
                   WHERE id = $4""",
                new_content, model, tsv_text, row['id']
            )
            if (
                shared.CONVERSATION_RECALL_ENABLED
                and shared.EMBEDDING_API_KEY
                and new_content
                and new_content.strip()
            ):
                db_search.kick_embedding_backfill()
            return True
        return False


async def search_conversations(query: str, limit: int = 20, offset: int = 0):
    """搜索对话内容，返回匹配的session列表"""
    keywords = db_search.extract_search_keywords(query)
    if not keywords:
        return [], 0

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        where_parts = []
        params = []
        for i, kw in enumerate(keywords):
            where_parts.append(f"c.content ILIKE '%' || ${i+1} || '%'")
            params.append(kw)
        where_clause = " OR ".join(where_parts)

        count_sql = f"""
            SELECT COUNT(DISTINCT c.session_id) as total
            FROM conversations c
            WHERE {where_clause}
        """
        total_row = await conn.fetchrow(count_sql, *params)
        total = total_row['total'] if total_row else 0

        if total == 0:
            return [], 0

        limit_idx = len(params) + 1
        offset_idx = len(params) + 2
        params.extend([limit, offset])

        sql = f"""
            WITH matched_sessions AS (
                SELECT DISTINCT c.session_id
                FROM conversations c
                WHERE {where_clause}
            ),
            session_info AS (
                SELECT
                    ms.session_id,
                    MIN(c.created_at) as first_time,
                    MAX(c.created_at) as last_time,
                    COUNT(*) as message_count
                FROM matched_sessions ms
                JOIN conversations c ON c.session_id = ms.session_id
                GROUP BY ms.session_id
            )
            SELECT
                si.session_id,
                si.first_time,
                si.last_time,
                si.message_count
            FROM session_info si
            ORDER BY si.last_time DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """
        rows = await conn.fetch(sql, *params)

        results = []
        for r in rows:
            results.append({
                'session_id': r['session_id'],
                'first_time': r['first_time'].isoformat() if r['first_time'] else None,
                'last_time': r['last_time'].isoformat() if r['last_time'] else None,
                'message_count': r['message_count'],
            })

        return results, total


async def update_message_content(message_id: int, new_content: str):
    """更新单条对话消息的内容"""
    tsv_text = (
        db_search.jieba_tokenize_for_tsv(new_content or "")
        if shared.CONVERSATION_RECALL_ENABLED
        else None
    )
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        embedding_column = "embedding" if db_core.HAS_PGVECTOR else "embedding_json"
        result = await conn.execute(
            f"""UPDATE conversations
               SET content = $1,
                   content_tsv = array_to_tsvector(string_to_array($2, ' ')),
                   {embedding_column} = NULL
               WHERE id = $3""",
            new_content, tsv_text, message_id,
        )
    updated = int(result.split()[-1]) if result else 0
    if (
        updated
        and shared.CONVERSATION_RECALL_ENABLED
        and shared.EMBEDDING_API_KEY
        and new_content
        and new_content.strip()
    ):
        db_search.kick_embedding_backfill()
    return updated


async def delete_single_message(message_id: int):
    """删除单条对话消息（硬删除）"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM conversations WHERE id = $1",
            message_id,
        )
        return int(result.split()[-1]) if result else 0


# ============================================================
# 对话历史读取（分区缓存用）
# ============================================================

async def get_conversation_messages(session_id: str, limit: int = 100):
    """按时间正序读取session的消息"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT role, content, metadata, created_at
            FROM conversations
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2
        """, session_id, limit)
        return [dict(r) for r in rows]


# ============================================================
# 分区缓存状态管理
# ============================================================

def _active_seen_fragment_ids(seen_fragment_times, ttl_hours: float, now=None) -> list:
    """Return fragment IDs whose individual seen timestamps are still inside TTL."""
    if ttl_hours <= 0:
        return []
    if isinstance(seen_fragment_times, str):
        try:
            seen_fragment_times = json.loads(seen_fragment_times)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(seen_fragment_times, dict):
        return []

    now = now or datetime.now(dt_timezone.utc)
    cutoff = now - timedelta(hours=ttl_hours)
    active = []
    for fragment_id, raw_seen_at in seen_fragment_times.items():
        try:
            if isinstance(raw_seen_at, datetime):
                seen_at = raw_seen_at
            else:
                seen_at = datetime.fromisoformat(str(raw_seen_at).replace("Z", "+00:00"))
            if seen_at.tzinfo is None:
                seen_at = seen_at.replace(tzinfo=dt_timezone.utc)
            if seen_at >= cutoff:
                active.append(str(fragment_id))
        except (TypeError, ValueError):
            continue
    return sorted(set(active))


def _active_seen_memory_ids(seen_memory_times, ttl_hours: float, now=None) -> list:
    """Return integer memory IDs whose individual seen timestamps are inside TTL."""
    active = []
    for memory_id in _active_seen_fragment_ids(seen_memory_times, ttl_hours, now):
        try:
            active.append(int(memory_id))
        except (TypeError, ValueError):
            continue
    return sorted(set(active))


async def get_active_seen_memory_ids(session_id: str, ttl_hours: float = 6) -> list:
    """Read only the memory seen ledger for one partition session."""
    ttl_hours = float(ttl_hours)
    if not session_id or ttl_hours <= 0:
        return []
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        seen_memory_times = await conn.fetchval(
            """SELECT seen_memory_times
               FROM session_cache_state WHERE session_id = $1""",
            session_id,
        )
    return _active_seen_memory_ids(seen_memory_times or {}, ttl_hours)


async def get_session_cache_state(session_id: str, seen_ttl_hours: float = None) -> dict:
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT summary, a_start_round, seen_fragment_ids,
                      seen_fragment_times, updated_at
               FROM session_cache_state WHERE session_id = $1""",
            session_id
        )
        if row:
            raw_summary = row['summary'] or ''
            summary_parts = []
            if raw_summary:
                try:
                    parsed = json.loads(raw_summary)
                    if isinstance(parsed, list):
                        summary_parts = parsed
                    else:
                        summary_parts = [raw_summary]
                except (json.JSONDecodeError, ValueError):
                    summary_parts = [raw_summary]
            raw_seen_times = row.get('seen_fragment_times') or {}
            if isinstance(raw_seen_times, str):
                try:
                    raw_seen_times = json.loads(raw_seen_times)
                except (json.JSONDecodeError, ValueError):
                    raw_seen_times = {}
            if not raw_seen_times and row['seen_fragment_ids']:
                legacy_seen_at = row['updated_at'] or datetime.now(dt_timezone.utc)
                raw_seen_times = {
                    str(fragment_id): legacy_seen_at.isoformat()
                    for fragment_id in row['seen_fragment_ids']
                }
            seen_fragment_ids = (
                _active_seen_fragment_ids(raw_seen_times, seen_ttl_hours)
                if seen_ttl_hours is not None
                else sorted(str(fragment_id) for fragment_id in raw_seen_times)
            )
            return {
                'summary_parts': summary_parts,
                'a_start_round': row['a_start_round'] or 0,
                'seen_fragment_ids': seen_fragment_ids,
                'updated_at': row['updated_at'],
            }
        return {
            'summary_parts': [],
            'a_start_round': 0,
            'seen_fragment_ids': [],
            'updated_at': None,
        }


async def save_session_cache_state(session_id: str, summary_parts: list, a_start_round: int):
    summary_json = json.dumps(summary_parts, ensure_ascii=False)
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO session_cache_state (session_id, summary, a_start_round, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (session_id)
            DO UPDATE SET summary = $2, a_start_round = $3, updated_at = NOW()
        """, session_id, summary_json, a_start_round)


async def _merge_seen_ledger(session_id: str, column: str, fresh_seen: str, ttl_hours: float):
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"""INSERT INTO session_cache_state (
                   session_id, {column}, updated_at
               ) VALUES ($1, $2::jsonb, NOW())
               ON CONFLICT (session_id) DO UPDATE
               SET {column} = (
                       SELECT COALESCE(
                           jsonb_object_agg(active.key, active.value),
                            '{{}}'::jsonb
                       )
                       FROM jsonb_each(
                           COALESCE(
                               session_cache_state.{column},
                               '{{}}'::jsonb
                           )
                       ) AS active(key, value)
                        WHERE (active.value #>> '{{}}')::timestamptz >=
                             NOW() - ($3::double precision * INTERVAL '1 hour')
                   ) || EXCLUDED.{column},
                   updated_at = NOW()""",
            session_id, fresh_seen, ttl_hours,
        )


async def mark_fragments_seen(session_id: str, fragment_ids: list, ttl_hours: float = 6):
    """成功请求结束后原子合并已注入 fragment_id，不覆盖分区摘要状态。"""
    ids = sorted({str(value) for value in fragment_ids if value})
    ttl_hours = float(ttl_hours)
    if not session_id or not ids or ttl_hours <= 0:
        return 0
    seen_at = datetime.now(dt_timezone.utc).isoformat()
    fresh_seen = json.dumps(
        {fragment_id: seen_at for fragment_id in ids},
        ensure_ascii=False,
    )
    await _merge_seen_ledger(
        session_id,
        "seen_fragment_times",
        fresh_seen,
        ttl_hours,
    )
    return len(ids)


async def mark_memories_seen(session_id: str, memory_ids: list, ttl_hours: float = 6):
    """成功请求结束后原子合并已注入 memory id，不覆盖分区与对话召回状态。"""
    ids = sorted({
        int(value)
        for value in memory_ids
        if isinstance(value, int) and not isinstance(value, bool)
    })
    ttl_hours = float(ttl_hours)
    if not session_id or not ids or ttl_hours <= 0:
        return 0
    seen_at = datetime.now(dt_timezone.utc).isoformat()
    fresh_seen = json.dumps(
        {str(memory_id): seen_at for memory_id in ids},
        ensure_ascii=False,
    )
    await _merge_seen_ledger(
        session_id,
        "seen_memory_times",
        fresh_seen,
        ttl_hours,
    )
    return len(ids)


# ============================================================
# Token 使用记录
# ============================================================

async def ensure_token_usage_table():
    """确保token_usage表存在（在init_tables里调用）"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT,
                model           TEXT,
                prompt_tokens   INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens    INTEGER DEFAULT 0,
                usage_type      TEXT DEFAULT 'chat',
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_created ON token_usage (created_at DESC);
        """)


async def save_token_usage(session_id: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, usage_type: str = "chat"):
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO token_usage (session_id, model, prompt_tokens, completion_tokens, total_tokens, usage_type)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, session_id, model, prompt_tokens, completion_tokens, total_tokens, usage_type)


# ============================================================
# 对话记录管理
# ============================================================

async def get_conversations_paginated(page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        total_row = await conn.fetchrow(
            "SELECT COUNT(DISTINCT session_id) as total FROM conversations"
        )
        total = total_row['total'] if total_row else 0

        rows = await conn.fetch("""
            WITH session_info AS (
                SELECT session_id, MIN(created_at) as first_time, MAX(created_at) as last_time, COUNT(*) as message_count
                FROM conversations GROUP BY session_id ORDER BY last_time DESC LIMIT $1 OFFSET $2
            )
            SELECT si.*,
                   COALESCE(tu.total_all, 0) as total_tokens
            FROM session_info si
            LEFT JOIN (
                SELECT session_id, SUM(total_tokens) as total_all FROM token_usage WHERE usage_type = 'chat' GROUP BY session_id
            ) tu ON si.session_id = tu.session_id
            ORDER BY si.last_time DESC
        """, per_page, offset)

        results = []
        for r in rows:
            preview_row = await conn.fetchrow(
                "SELECT content FROM conversations WHERE session_id = $1 AND role = 'user' ORDER BY created_at LIMIT 1",
                r['session_id']
            )
            preview = preview_row['content'][:80] if preview_row else ''
            title = (preview[:30] + '...' if len(preview) > 30 else preview) or r['session_id']
            results.append({
                'session_id': r['session_id'],
                'title': title,
                'first_time': r['first_time'].isoformat() if r['first_time'] else None,
                'last_time': r['last_time'].isoformat() if r['last_time'] else None,
                'message_count': r['message_count'],
                'preview': preview,
                'total_tokens': r['total_tokens'],
            })
        return results, total


async def delete_conversation(session_id: str):
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = $1", session_id)


async def batch_delete_conversations(session_ids: list):
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE session_id = ANY($1)", session_ids)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = ANY($1)", session_ids)


async def merge_sessions_to_target(source_ids: list, target_id: str) -> dict:
    if not source_ids:
        return {'merged_sessions': 0, 'merged_messages': 0, 'merged_token_records': 0}
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        msg_count = await conn.fetchval("SELECT COUNT(*) FROM conversations WHERE session_id = ANY($1)", source_ids)
        await conn.execute("UPDATE conversations SET session_id = $1 WHERE session_id = ANY($2)", target_id, source_ids)
        token_count = await conn.fetchval("SELECT COUNT(*) FROM token_usage WHERE session_id = ANY($1)", source_ids)
        await conn.execute("UPDATE token_usage SET session_id = $1 WHERE session_id = ANY($2)", target_id, source_ids)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = ANY($1)", source_ids)
        return {'merged_sessions': len(source_ids), 'merged_messages': msg_count or 0, 'merged_token_records': token_count or 0}


async def list_all_session_cache_states() -> list:
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT scs.session_id, scs.summary, scs.a_start_round, scs.updated_at,
                   COALESCE(c.message_count, 0) as message_count,
                   COALESCE(tu.chat_tokens, 0) as chat_tokens
            FROM session_cache_state scs
            LEFT JOIN (SELECT session_id, COUNT(*) as message_count FROM conversations GROUP BY session_id) c ON scs.session_id = c.session_id
            LEFT JOIN (SELECT session_id, SUM(total_tokens) as chat_tokens FROM token_usage WHERE usage_type = 'chat' GROUP BY session_id) tu ON scs.session_id = tu.session_id
            ORDER BY scs.updated_at DESC
        """)
        results = []
        for r in rows:
            raw_summary = r['summary'] or ''
            try:
                parsed = json.loads(raw_summary)
                if isinstance(parsed, list):
                    summary_parts = parsed
                else:
                    summary_parts = [raw_summary] if raw_summary else []
            except (json.JSONDecodeError, ValueError):
                summary_parts = [raw_summary] if raw_summary else []
            results.append({
                'session_id': r['session_id'],
                'summary': '\n\n'.join(summary_parts),
                'summary_length': sum(len(p) for p in summary_parts),
                'summary_count': len(summary_parts),
                'a_start_round': r['a_start_round'],
                'updated_at': r['updated_at'].isoformat() if r['updated_at'] else None,
                'message_count': r['message_count'],
                'chat_tokens': r['chat_tokens'],
            })
        return results


async def delete_session_cache_state(session_id: str):
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = $1", session_id)


async def rename_session_id(old_id: str, new_id: str) -> bool:
    """重命名对话线ID（事务内同时修改三个表）"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 检查新ID是否已存在
            exists = await conn.fetchval(
                "SELECT 1 FROM session_cache_state WHERE session_id = $1", new_id
            )
            if exists:
                return False
            # session_cache_state
            await conn.execute(
                "UPDATE session_cache_state SET session_id = $1 WHERE session_id = $2",
                new_id, old_id
            )
            # conversations
            await conn.execute(
                "UPDATE conversations SET session_id = $1 WHERE session_id = $2",
                new_id, old_id
            )
            # token_usage
            await conn.execute(
                "UPDATE token_usage SET session_id = $1 WHERE session_id = $2",
                new_id, old_id
            )
            return True


def db_row_to_message(row: dict) -> dict:
    """
    把DB记录还原成API消息格式。

    普通消息: {"role": "user", "content": "你好"}
    工具调用: {"role": "assistant", "content": null, "tool_calls": [...]}
    工具结果: {"role": "tool", "content": "结果", "tool_call_id": "call_xxx"}
    思维链:   {"role": "assistant", "content": "回答", "reasoning_content": "思维链"}
    """
    import json as _json
    msg = {"role": row["role"], "content": row.get("content") or ""}

    meta_str = row.get("metadata")
    if meta_str:
        try:
            meta = _json.loads(meta_str)
            # assistant 带 tool_calls
            if "tool_calls" in meta:
                msg["tool_calls"] = meta["tool_calls"]
                if not row.get("content"):
                    msg["content"] = None
            # assistant 带 reasoning_content（deepseek thinking mode）
            if "reasoning_content" in meta:
                msg["reasoning_content"] = meta["reasoning_content"]
            # tool 消息带 tool_call_id
            if "tool_call_id" in meta:
                msg["tool_call_id"] = meta["tool_call_id"]
            # 其他可能的字段（name 等）
            if "name" in meta:
                msg["name"] = meta["name"]
        except Exception:
            pass

    return msg


async def export_all_conversations():
    """导出所有对话记录（用于备份）"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT session_id, role, content, model, created_at
            FROM conversations
            ORDER BY session_id, created_at
        """)
        return [
            {
                'session_id': r['session_id'],
                'role': r['role'],
                'content': r['content'],
                'model': r['model'] or '',
                'created_at': r['created_at'].isoformat() if r['created_at'] else None,
            }
            for r in rows
        ]


async def import_conversations(records: list):
    """
    导入对话记录（自动去重）

    records: [{ session_id, role, content, model?, created_at? }, ...]
    按 session_id + role + created_at 三元组去重，已存在的跳过。
    返回 (导入数量, 跳过数量)
    """
    if not records:
        return 0, 0

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        imported = 0
        skipped = 0
        for r in records:
            session_id = r.get('session_id')
            role = r.get('role')
            content = r.get('content')

            if not all([session_id, role, content]):
                continue

            model = r.get('model', '')
            created_at = r.get('created_at')
            tsv_text = (
                db_search.jieba_tokenize_for_tsv(content)
                if shared.CONVERSATION_RECALL_ENABLED
                else None
            )

            # 解析时间
            if created_at and isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    created_at = None

            # 去重检查
            if created_at:
                existing = await conn.fetchrow("""
                    SELECT id FROM conversations
                    WHERE session_id = $1 AND role = $2 AND created_at = $3
                    LIMIT 1
                """, session_id, role, created_at)

                if existing:
                    skipped += 1
                    continue

                await conn.execute("""
                    INSERT INTO conversations (
                        session_id, role, content, model, created_at, content_tsv
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        array_to_tsvector(string_to_array($6, ' '))
                    )
                """, session_id, role, content, model, created_at, tsv_text)
            else:
                await conn.execute("""
                    INSERT INTO conversations (
                        session_id, role, content, model, content_tsv
                    ) VALUES (
                        $1, $2, $3, $4,
                        array_to_tsvector(string_to_array($5, ' '))
                    )
                """, session_id, role, content, model, tsv_text)

            imported += 1

        if skipped:
            print(f"📥 导入对话: {imported} 条新增, {skipped} 条已存在跳过")
        else:
            print(f"📥 导入对话: {imported} 条新增")

        if (
            imported
            and shared.CONVERSATION_RECALL_ENABLED
            and shared.EMBEDDING_API_KEY
        ):
            db_search.kick_embedding_backfill()
        return imported, skipped
