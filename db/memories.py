"""Memory persistence, search, backup, and lifecycle operations."""

import json
import logging
from datetime import date, datetime, timedelta, timezone as dt_timezone

import shared
from db import core as db_core
from db import search as db_search
from db.core import BrokenMergeReferencesError, BrokenSupersessionReferencesError

logger = logging.getLogger(__name__)

_EXTRACTION_RELEVANT_LIMIT = 10
_EXTRACTION_RECENT_LIMIT = 10

# ============================================================
# 记忆操作
# ============================================================

async def save_memory(content: str, importance: int = 5, source_session: str = "",
                      created_at: datetime = None, title: str = None,
                      layer: int = 1, event_date=None, external_id: str = None):
    """created_at 传入时保留原时间（备份恢复用），否则落库默认 NOW()"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO memories "
            "(content, importance, source_session, created_at, title, layer, "
            "event_date, external_id, is_active) "
            "VALUES ($1, $2, $3, COALESCE($4, NOW()), $5, $6, $7, $8, TRUE) "
            "ON CONFLICT (external_id) WHERE external_id IS NOT NULL "
            "DO NOTHING RETURNING id",
            content, importance, source_session, created_at, title, layer,
            event_date, external_id,
        )

        # MEMORY_VECTOR_ENABLED 时自动计算 embedding
        if shared.MEMORY_VECTOR_ENABLED and row:
            try:
                embedding = await db_search.compute_embedding(content)
                if embedding:
                    await db_search.save_memory_embedding(conn, row['id'], embedding)
            except Exception as e:
                print(f"⚠️ 记忆 {row['id']} embedding自动计算失败: {e}")
        return row["id"] if row else None


async def save_extracted_memory(
    content: str,
    importance: int,
    source_session: str,
    supersede_id=None,
    candidate_ids=None,
):
    """Save one extracted fact and atomically retire an allowed active predecessor."""
    allowed_ids = {
        memory_id for memory_id in (candidate_ids or [])
        if isinstance(memory_id, int) and not isinstance(memory_id, bool)
    }
    requested_id = (
        supersede_id
        if isinstance(supersede_id, int) and not isinstance(supersede_id, bool)
        else None
    )

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO memories
                    (content, importance, source_session, layer, is_active)
                VALUES ($1, $2, $3, 1, TRUE)
                RETURNING id
                """,
                content,
                importance,
                source_session,
            )
            new_id = int(row["id"])
            retired_id = None
            if requested_id in allowed_ids:
                predecessor = await conn.fetchrow(
                    """
                    SELECT id, is_active, superseded_by
                    FROM memories
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    requested_id,
                )
                if (
                    predecessor
                    and predecessor["is_active"] is True
                    and predecessor["superseded_by"] is None
                ):
                    result = await conn.execute(
                        """
                        UPDATE memories
                        SET is_active = FALSE, superseded_by = $2
                        WHERE id = $1
                          AND is_active = TRUE
                          AND superseded_by IS NULL
                        """,
                        requested_id,
                        new_id,
                    )
                    if result == "UPDATE 1":
                        retired_id = requested_id

    if shared.MEMORY_VECTOR_ENABLED:
        try:
            embedding = await db_search.compute_embedding(content)
            if embedding:
                async with pool.acquire() as conn:
                    await db_search.save_memory_embedding(conn, new_id, embedding)
        except Exception as exc:
            print(f"⚠️ 记忆 {new_id} embedding自动计算失败: {exc}")

    return {
        "id": new_id,
        "action": "supersede" if retired_id is not None else "new",
        "superseded_id": retired_id,
    }


async def get_memory_by_external_id(external_id: str):
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, content, importance, title, layer, event_date,
                   source_session, external_id, is_active, created_at
            FROM memories
            WHERE external_id = $1
            """,
            external_id,
        )
    return dict(row) if row else None


def _normalize_excluded_ids(exclude_ids) -> list:
    return sorted({
        int(value)
        for value in (exclude_ids or [])
        if isinstance(value, int) and not isinstance(value, bool)
    })


async def search_memories(query: str, limit: int = 10, exclude_ids=None):
    """
    搜索相关记忆

    MEMORY_VECTOR_ENABLED=true 时走混合搜索（关键词 + 向量）
    否则走纯关键词搜索
    """
    excluded_ids = _normalize_excluded_ids(exclude_ids)
    if shared.MEMORY_VECTOR_ENABLED:
        return await search_memories_hybrid(query, limit, exclude_ids=excluded_ids)

    # ---- 纯关键词搜索 ----
    keywords = db_search.extract_search_keywords(query)

    if not keywords:
        return []

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        # 每个关键词命中得1分
        case_parts = []
        params = []
        for i, kw in enumerate(keywords):
            case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
            params.append(kw)

        hit_count_expr = " + ".join(case_parts)
        max_hits = len(keywords)

        # 至少命中一个关键词（只搜索活跃记忆）
        where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
        where_clause = f"is_active = TRUE AND ({' OR '.join(where_parts)})"

        if excluded_ids:
            exclude_idx = len(params) + 1
            params.append(excluded_ids)
            where_clause += f" AND NOT (id = ANY(${exclude_idx}::int[]))"

        limit_idx = len(params) + 1
        params.append(limit)

        # 时效天数：事件记忆按本地日历日差（AT TIME ZONE 'UTC' 拿到无时区的 UTC 挂钟，
        # 加偏移后取日期，不依赖数据库会话时区）；普通碎片按 created_at 精确时长
        recency_days_expr = (
            "CASE WHEN event_date IS NOT NULL "
            f"THEN GREATEST(0, ((NOW() AT TIME ZONE 'UTC' + INTERVAL '{shared.TIMEZONE_HOURS} hours')::date - event_date))::float "
            "ELSE GREATEST(0, EXTRACT(EPOCH FROM (NOW() - created_at))) / 86400.0 END"
        )
        sql = f"""
            SELECT
                id, content, importance, created_at, event_date,
                ({hit_count_expr}) AS hit_count,
                ({recency_days_expr}) AS effective_days,
                (
                    {shared.WEIGHT_KEYWORD} * ({hit_count_expr})::float / {max_hits}.0 +
                    {shared.WEIGHT_IMPORTANCE} * importance::float / 10.0 +
                    {shared.WEIGHT_RECENCY} * (1.0 / (1.0 + ({recency_days_expr})))
                ) AS score
            FROM memories
            WHERE {where_clause}
            ORDER BY score DESC, importance DESC, effective_days ASC
            LIMIT ${limit_idx}
        """

        results = await conn.fetch(sql, *params)

        # 过滤低分记忆
        if shared.MIN_SCORE_THRESHOLD > 0:
            before_count = len(results)
            results = [r for r in results if r['score'] >= shared.MIN_SCORE_THRESHOLD]
            filtered = before_count - len(results)
        else:
            filtered = 0

        if results:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]}{'...' if len(keywords)>8 else ''} → 命中 {len(results)} 条" + (f"（过滤 {filtered} 条低分）" if filtered else ""))
            for r in results[:3]:
                print(f"   📌 [score={r['score']:.3f}] (hits={r['hit_count']}, imp={r['importance']}) {r['content'][:60]}...")

            ids = [r["id"] for r in results]
            await conn.execute(
                "UPDATE memories SET last_accessed = NOW() WHERE id = ANY($1::int[])",
                ids,
            )
        else:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]} → 无结果" + (f"（{filtered} 条被分数阈值过滤）" if filtered else ""))

        return results


def _effective_days_ago(event_date, created_at, now_utc):
    """时效天数：事件记忆按本地日历日差（event_date 是本地日期，不做 UTC 换算），
    普通碎片按 created_at 精确时长"""
    if event_date:
        local_today = (now_utc + timedelta(hours=shared.TIMEZONE_HOURS)).date()
        return max(0.0, float((local_today - event_date).days))
    return max(0.0, (now_utc - created_at).total_seconds() / 86400.0)


async def search_memories_hybrid(
    query: str,
    limit: int = 10,
    return_mode: bool = False,
    exclude_ids=None,
):
    """
    记忆混合搜索：关键词 + 向量，归一化后四维加权

    权重：MEMORY_HW_KEYWORD + MEMORY_HW_SEMANTIC + MEMORY_HW_IMPORTANCE + MEMORY_HW_RECENCY
    """
    excluded_ids = _normalize_excluded_ids(exclude_ids)
    keywords = db_search.extract_search_keywords(query)
    query_embedding = await db_search.get_query_embedding(query) if shared.EMBEDDING_API_KEY else []
    search_mode = "hybrid" if query_embedding else "keyword"

    if not keywords and not query_embedding:
        return ([], search_mode) if return_mode else []

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        candidates = {}  # id -> {content, importance, created_at, kw_score, similarity}

        # ---- 关键词路 ----
        if keywords:
            case_parts = []
            params = []
            for i, kw in enumerate(keywords):
                case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
                params.append(kw)

            hit_count_expr = " + ".join(case_parts)
            max_hits = len(keywords)
            where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
            where_clause = f"is_active = TRUE AND ({' OR '.join(where_parts)})"

            if excluded_ids:
                exclude_idx = len(params) + 1
                params.append(excluded_ids)
                where_clause += f" AND NOT (id = ANY(${exclude_idx}::int[]))"

            limit_idx = len(params) + 1
            params.append(limit * 3)

            kw_sql = f"""
                SELECT id, content, importance, created_at, event_date,
                       ({hit_count_expr}) AS hit_count,
                       ({hit_count_expr})::float / {max_hits}.0 AS kw_score
                FROM memories
                WHERE {where_clause}
                ORDER BY kw_score DESC
                LIMIT ${limit_idx}
            """
            kw_rows = await conn.fetch(kw_sql, *params)

            for r in kw_rows:
                candidates[r['id']] = {
                    'content': r['content'],
                    'importance': r['importance'],
                    'created_at': r['created_at'],
                    'event_date': r['event_date'],
                    'hit_count': r['hit_count'],
                    'kw_score': float(r['kw_score']),
                    'similarity': 0.0,
                }

        # ---- 向量路 ----
        if query_embedding:
            if db_core.HAS_PGVECTOR:
                vec_str = '[' + ','.join(str(f) for f in query_embedding) + ']'
                sem_rows = await conn.fetch("""
                    SELECT id, content, importance, created_at, event_date,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM memories
                    WHERE embedding IS NOT NULL AND is_active = TRUE
                      AND NOT (id = ANY($2::int[]))
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """, vec_str, excluded_ids, limit * 3)
            else:
                # Python端计算cosine
                all_mem = await conn.fetch("""
                    SELECT id, content, importance, created_at, event_date, embedding_json
                    FROM memories
                    WHERE embedding_json IS NOT NULL AND is_active = TRUE
                      AND NOT (id = ANY($1::int[]))
                """, excluded_ids)

                scored = []
                for row in all_mem:
                    try:
                        emb = json.loads(row['embedding_json'])
                        sim = db_search._cosine_sim(query_embedding, emb)
                        scored.append({**dict(row), 'similarity': sim})
                    except Exception:
                        continue
                scored.sort(key=lambda x: -x['similarity'])
                sem_rows = scored[:limit * 3]

            for r in sem_rows:
                sim = float(r['similarity'])
                if sim < shared.MEMORY_SEMANTIC_THRESHOLD:
                    continue
                mid = r['id']
                if mid in candidates:
                    candidates[mid]['similarity'] = sim
                else:
                    candidates[mid] = {
                        'content': r['content'],
                        'importance': r['importance'],
                        'created_at': r['created_at'],
                        'event_date': r['event_date'],
                        'hit_count': 0,
                        'kw_score': 0.0,
                        'similarity': sim,
                    }

            # debug：向量路统计
            sem_total = len(sem_rows)
            sem_passed = sum(1 for r in sem_rows if float(r['similarity']) >= shared.MEMORY_SEMANTIC_THRESHOLD)
            sem_max = max((float(r['similarity']) for r in sem_rows), default=0)
            if sem_total > 0 and sem_passed == 0:
                print(f"   🔢 向量路: {sem_total}条候选全被阈值过滤（最高sim={sem_max:.3f}, 阈值={shared.MEMORY_SEMANTIC_THRESHOLD}）")
            elif sem_total > 0:
                print(f"   🔢 向量路: {sem_passed}/{sem_total}条通过阈值（最高sim={sem_max:.3f}）")

        if not candidates:
            print(f"🔍 混合搜索 '{query}' → 两路均无结果")
            return ([], search_mode) if return_mode else []

        # ---- 归一化 + 加权 ----
        kw_norm = db_search._min_max_normalize({mid: v['kw_score'] for mid, v in candidates.items()})
        sem_norm = db_search._min_max_normalize({mid: v['similarity'] for mid, v in candidates.items()})

        now = datetime.now(dt_timezone.utc)
        final = []
        for mid, info in candidates.items():
            kw = kw_norm.get(mid, 0.0)
            sem = sem_norm.get(mid, 0.0)
            imp = info['importance'] / 10.0
            days = _effective_days_ago(info.get('event_date'), info['created_at'], now)
            rec = 1.0 / (1.0 + days)

            score = (shared.MEMORY_HW_KEYWORD * kw +
                     shared.MEMORY_HW_SEMANTIC * sem +
                     shared.MEMORY_HW_IMPORTANCE * imp +
                     shared.MEMORY_HW_RECENCY * rec)

            final.append({
                'id': mid,
                'content': info['content'],
                'importance': info['importance'],
                'created_at': info['created_at'],
                'event_date': info.get('event_date'),
                'hit_count': info['hit_count'],
                'similarity': info['similarity'],
                'score': score,
            })

        final.sort(key=lambda x: (-x['score'], -x['importance']))

        # 过滤低分
        if shared.MIN_SCORE_THRESHOLD > 0:
            before_count = len(final)
            final = [r for r in final if r['score'] >= shared.MIN_SCORE_THRESHOLD]
            filtered = before_count - len(final)
        else:
            filtered = 0

        results = final[:limit]

        if results:
            mode_tag = "混合" if query_embedding else "关键词"
            kw_tag = f"关键词 {keywords[:6]}" if keywords else "无关键词"
            print(f"🔍 {mode_tag}搜索 '{query}' → {kw_tag} → 命中 {len(results)} 条" + (f"（过滤 {filtered} 条低分）" if filtered else ""))
            for r in results[:3]:
                print(f"   📌 [score={r['score']:.3f}] (kw={r['hit_count']}, sim={r['similarity']:.2f}, imp={r['importance']}) {r['content'][:60]}...")

            ids = [r["id"] for r in results]
            await conn.execute(
                "UPDATE memories SET last_accessed = NOW() WHERE id = ANY($1::int[])",
                ids,
            )
        else:
            print(f"🔍 混合搜索 '{query}' → 无结果" + (f"（{filtered} 条被过滤）" if filtered else ""))

        output = [dict(r) for r in results]
        return (output, search_mode) if return_mode else output


async def search_memories_with_mode(query: str, limit: int = 10):
    """搜索记忆，并报告本次实际使用了混合搜索还是关键词搜索。"""
    if shared.MEMORY_VECTOR_ENABLED:
        return await search_memories_hybrid(query, limit, return_mode=True)
    return await search_memories(query, limit), "keyword"


async def get_extraction_candidates(
    query: str,
    relevant_limit: int = _EXTRACTION_RELEVANT_LIMIT,
    recent_limit: int = _EXTRACTION_RECENT_LIMIT,
):
    """Return read-only dedup candidates: relevant top K union latest M active."""
    def candidate_row(row):
        return {
            key: row[key]
            for key in ("id", "content", "importance", "created_at", "event_date")
        }

    keywords = db_search.extract_search_keywords(query)
    query_embedding = (
        await db_search.get_query_embedding(query)
        if relevant_limit > 0 and shared.MEMORY_VECTOR_ENABLED and shared.EMBEDDING_API_KEY
        else []
    )

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        relevant = {}
        candidate_pool_size = relevant_limit * 3

        if keywords and relevant_limit > 0:
            hit_parts = [
                f"CASE WHEN content ILIKE '%' || ${index + 1} || '%' THEN 1 ELSE 0 END"
                for index in range(len(keywords))
            ]
            hit_count = " + ".join(hit_parts)
            where_parts = [
                f"content ILIKE '%' || ${index + 1} || '%'"
                for index in range(len(keywords))
            ]
            keyword_rows = await conn.fetch(
                f"""
                SELECT id, content, importance, created_at, event_date,
                       ({hit_count})::float / {len(keywords)}.0 AS keyword_score
                FROM memories
                WHERE is_active = TRUE AND ({' OR '.join(where_parts)})
                ORDER BY keyword_score DESC, id ASC
                LIMIT ${len(keywords) + 1}
                """,
                *keywords,
                candidate_pool_size,
            )
            for row in keyword_rows:
                relevant[row["id"]] = {
                    "row": candidate_row(row),
                    "keyword_score": float(row["keyword_score"]),
                    "similarity": None,
                }

        if query_embedding and relevant_limit > 0:
            semantic_rows = []
            try:
                if db_core.HAS_PGVECTOR:
                    vector_text = "[" + ",".join(str(value) for value in query_embedding) + "]"
                    semantic_rows = await conn.fetch(
                        """
                        SELECT id, content, importance, created_at, event_date,
                               1 - (embedding <=> $1::vector) AS similarity
                        FROM memories
                        WHERE embedding IS NOT NULL AND is_active = TRUE
                        ORDER BY embedding <=> $1::vector, id ASC
                        LIMIT $2
                        """,
                        vector_text,
                        candidate_pool_size,
                    )
                else:
                    embedded_rows = await conn.fetch(
                        """
                        SELECT id, content, importance, created_at, event_date,
                               embedding_json
                        FROM memories
                        WHERE embedding_json IS NOT NULL AND is_active = TRUE
                        """
                    )
                    for row in embedded_rows:
                        try:
                            semantic_rows.append({
                                **dict(row),
                                "similarity": db_search._cosine_sim(
                                    query_embedding,
                                    json.loads(row["embedding_json"]),
                                ),
                            })
                        except Exception:
                            continue
                    semantic_rows.sort(key=lambda row: (-row["similarity"], row["id"]))
                    semantic_rows = semantic_rows[:candidate_pool_size]
            except Exception as exc:
                print(f"⚠️ 提取候选向量检索失败，退回关键词和最新记忆: {exc}")

            for row in semantic_rows:
                item = relevant.setdefault(row["id"], {
                    "row": candidate_row(row),
                    "keyword_score": None,
                    "similarity": None,
                })
                item["similarity"] = float(row["similarity"])

        def relevance_key(item):
            keyword_score = item["keyword_score"]
            similarity = item["similarity"]
            best_score = max(
                score for score in (keyword_score, similarity) if score is not None
            )
            return (
                -best_score,
                -(keyword_score if keyword_score is not None else float("-inf")),
                -(similarity if similarity is not None else float("-inf")),
                item["row"]["id"],
            )

        relevant_rows = [
            item["row"]
            for item in sorted(relevant.values(), key=relevance_key)[:relevant_limit]
        ]
        recent_rows = []
        if recent_limit > 0:
            recent_rows = await conn.fetch(
                """
                SELECT id, content, importance, created_at, event_date
                FROM memories
                WHERE is_active = TRUE
                ORDER BY created_at DESC, id DESC
                LIMIT $1
                """,
                recent_limit,
            )

    candidates = []
    seen_ids = set()
    for row in [*relevant_rows, *recent_rows]:
        memory = candidate_row(row)
        if memory["id"] in seen_ids:
            continue
        seen_ids.add(memory["id"])
        candidates.append(memory)
    return candidates


async def get_pending_memory_embedding_count():
    """查询还没有embedding的记忆数量"""
    embedding_column = "embedding" if db_core.HAS_PGVECTOR else "embedding_json"
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            f"SELECT COUNT(*) FROM memories WHERE {embedding_column} IS NULL AND content IS NOT NULL"
        )


async def backfill_memory_embeddings(batch_size: int = 20):
    """给已有记忆补算embedding（没有embedding的记忆）"""
    if not shared.EMBEDDING_API_KEY:
        print("⚠️ EMBEDDING_API_KEY 未设置，无法补算embedding")
        return 0

    embedding_column = "embedding" if db_core.HAS_PGVECTOR else "embedding_json"
    pool = await db_core.get_pool()
    total_updated = 0

    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT id, content FROM memories
            WHERE {embedding_column} IS NULL AND content IS NOT NULL
            ORDER BY id
            LIMIT $1
        """, batch_size)

    if not rows:
        print("✅ 所有记忆已有embedding，无需补算")
        return 0

    print(f"🔄 开始补算记忆embedding... 本批 {len(rows)} 条")

    async with pool.acquire() as conn:
        for row in rows:
            try:
                embedding = await db_search.compute_embedding(row['content'] or '')
                if embedding:
                    await db_search.save_memory_embedding(conn, row['id'], embedding)
                    total_updated += 1
            except Exception as e:
                print(f"⚠️ 记忆 {row['id']} embedding计算失败: {e}")

    # 检查剩余
    async with pool.acquire() as conn:
        remaining = await conn.fetchval(
            f"SELECT COUNT(*) FROM memories WHERE {embedding_column} IS NULL AND content IS NOT NULL"
        )

    print(f"✅ 本批补算完成：{total_updated}/{len(rows)} 条成功" + (f"，剩余 {remaining} 条待处理" if remaining > 0 else ""))
    return total_updated


async def get_recent_memories(limit: int = 20):
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """SELECT id, content, importance, created_at
               FROM memories
               WHERE is_active = TRUE
               ORDER BY created_at DESC
               LIMIT $1""",
            limit,
        )


async def get_all_memories_count():
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"]


async def get_all_memories():
    """导出所有记忆（用于备份，含归档记录与三层结构字段）

    embedding/embedding_json 和 last_accessed 是可重算的派生数据与访问状态，不进备份。
    """
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance, source_session, created_at,
                   layer, title, is_active, merged_from, event_date, superseded_by
            FROM memories ORDER BY id
        """)
        memories = [dict(r) for r in rows]

    memory_ids = {memory["id"] for memory in memories}
    broken_references = []
    for memory in memories:
        missing = sorted(set(memory.get("merged_from") or []) - memory_ids)
        if missing:
            broken_references.append((memory["id"], missing))
    if broken_references:
        logger.warning(
            "Backup blocked by broken merged_from references: %s",
            broken_references,
        )
        raise BrokenMergeReferencesError(len(broken_references))

    broken_supersession_references = [
        memory["id"]
        for memory in memories
        if memory.get("superseded_by") is not None
        and memory["superseded_by"] not in memory_ids
    ]
    if broken_supersession_references:
        logger.warning(
            "Backup blocked by broken superseded_by references: %s",
            broken_supersession_references,
        )
        raise BrokenSupersessionReferencesError(
            len(broken_supersession_references)
        )

    return memories


async def repair_broken_merge_references():
    """清除已经无法完整撤回的 merged_from 关系，保留父记忆本身。"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch("""
                WITH broken AS (
                    SELECT parent.id
                    FROM memories AS parent
                    WHERE parent.merged_from IS NOT NULL
                      AND EXISTS (
                          SELECT 1
                          FROM unnest(parent.merged_from) AS refs(source_id)
                          WHERE NOT EXISTS (
                              SELECT 1
                              FROM memories AS source
                              WHERE source.id = refs.source_id
                          )
                      )
                )
                UPDATE memories AS parent
                SET merged_from = NULL
                FROM broken
                WHERE parent.id = broken.id
                RETURNING parent.id
            """)
            return len(rows)


def _parse_backup_datetime(value):
    """解析备份里的时间字符串；解析不了返回 None（落库走默认 NOW()）"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt_timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=dt_timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(str(value)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt_timezone.utc)
        except ValueError:
            return None


def _parse_backup_date(value):
    """解析备份里的日期字符串（event_date 用），解析不了返回 None"""
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


async def import_memories_v2(memories: list, schema_version: int = 2):
    """恢复版本化备份：单事务建映射，再回填合并与版本关系。

    - 同内容且库中唯一 → 跳过并映射到已有行（同一份备份重复导入幂等）
    - 同内容但库中多行 → 不猜挂哪条，计入 conflicts 回执，引用它的合并链降级
    - merged_from 引用了备份中不存在的 backup_id → 备份损坏，抛错整批回滚
    - embedding 不随导入计算，恢复后由 backfill 重算
    """
    # ---- 事务外纯格式校验：先收集全部 backup_id，再验证引用封闭性 ----
    backup_ids = set()
    for mem in memories:
        if not isinstance(mem, dict):
            raise ValueError("记忆条目必须是 JSON 对象")
        bid = mem.get("backup_id")
        if isinstance(bid, bool) or not isinstance(bid, int):
            raise ValueError(f"backup_id 缺失或非法: {bid!r}")
        if bid in backup_ids:
            raise ValueError(f"backup_id 重复: {bid}")
        backup_ids.add(bid)
        if not isinstance(mem.get("content"), str) or not mem["content"].strip():
            raise ValueError(f"记忆 {bid} 缺少 content")
        if mem.get("layer", 1) not in (1, 2, 3):
            raise ValueError(f"记忆 {bid} 层级非法: {mem.get('layer')!r}")
    for mem in memories:
        for ref in (mem.get("merged_from") or []):
            if isinstance(ref, bool) or not isinstance(ref, int):
                raise ValueError(f"记忆 {mem['backup_id']} 的 merged_from 含非法引用: {ref!r}")
            if ref not in backup_ids:
                raise ValueError(
                    f"记忆 {mem['backup_id']} 的 merged_from 引用了备份中不存在的 {ref}，备份不完整"
                )
        successor = mem.get("superseded_by") if schema_version >= 3 else None
        if successor is not None:
            if isinstance(successor, bool) or not isinstance(successor, int):
                raise ValueError(
                    f"记忆 {mem['backup_id']} 的 superseded_by 引用非法: {successor!r}"
                )
            if successor not in backup_ids:
                raise ValueError(
                    f"记忆 {mem['backup_id']} 的 superseded_by 引用了备份中不存在的 {successor}，备份不完整"
                )

    pool = await db_core.get_pool()
    imported = 0
    skipped = 0
    conflicts = []
    degraded = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            # ---- 第一遍：插入并建立 旧 backup_id → 新库 id 映射 ----
            id_map = {}
            for mem in memories:
                bid = mem["backup_id"]
                content = mem["content"]
                rows = await conn.fetch(
                    "SELECT id FROM memories WHERE content = $1", content
                )
                if len(rows) == 1:
                    id_map[bid] = int(rows[0]["id"])
                    skipped += 1
                    continue
                if len(rows) > 1:
                    conflicts.append({
                        "backup_id": bid,
                        "matched_ids": sorted(int(r["id"]) for r in rows),
                    })
                    skipped += 1
                    continue
                row = await conn.fetchrow("""
                    INSERT INTO memories (content, importance, source_session, created_at,
                                          layer, title, is_active, event_date)
                    VALUES ($1, $2, $3, COALESCE($4, NOW()), $5, $6, $7, $8)
                    RETURNING id
                """,
                    content,
                    mem.get("importance", 5),
                    mem.get("source_session") or "json-import",
                    _parse_backup_datetime(mem.get("created_at")),
                    mem.get("layer", 1),
                    mem.get("title") or "",
                    bool(mem.get("is_active", True)),
                    _parse_backup_date(mem.get("event_date")),
                )
                id_map[bid] = int(row["id"])
                imported += 1

            # ---- 第二遍：用映射回填 merged_from ----
            for mem in memories:
                refs = mem.get("merged_from") or []
                if not refs:
                    continue
                bid = mem["backup_id"]
                new_id = id_map.get(bid)
                if new_id is None:
                    # 父条本身因内容冲突被跳过，没有落库行可回填
                    continue
                unresolved = [ref for ref in refs if ref not in id_map]
                if unresolved:
                    # 来源条目因冲突未建立映射：不猜关系，保持 NULL 并回执降级
                    degraded.append({"backup_id": bid, "unresolved": unresolved})
                    continue
                await conn.execute(
                    "UPDATE memories SET merged_from = $1 WHERE id = $2",
                    [id_map[ref] for ref in refs], new_id,
                )

            if schema_version >= 3:
                for mem in memories:
                    successor = mem.get("superseded_by")
                    if successor is None:
                        continue
                    old_id = id_map.get(mem["backup_id"])
                    successor_id = id_map.get(successor)
                    if old_id is None or successor_id is None:
                        raise ValueError(
                            f"记忆 {mem['backup_id']} 的版本链因内容冲突无法完整恢复"
                        )
                    await conn.execute(
                        """UPDATE memories
                           SET superseded_by = $1, is_active = FALSE
                           WHERE id = $2""",
                        successor_id,
                        old_id,
                    )

                restored_ids = list(id_map.values())
                broken_count = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM memories AS old
                    LEFT JOIN memories AS successor
                      ON successor.id = old.superseded_by
                    WHERE old.id = ANY($1::int[])
                      AND old.superseded_by IS NOT NULL
                      AND successor.id IS NULL
                    """,
                    restored_ids,
                )
                if broken_count:
                    raise ValueError("版本链恢复后校验失败")

    total = await get_all_memories_count()
    result = {
        "status": "done",
        "schema_version": schema_version,
        "imported": imported,
        "skipped": skipped,
        "conflicts": conflicts,
        "degraded": degraded,
        "total": total,
    }
    if shared.MEMORY_VECTOR_ENABLED:
        try:
            result["pending_embeddings"] = await get_pending_memory_embedding_count()
        except Exception:
            pass
    return result


async def get_all_memories_detail(limit: int = None, layer: int = None, active_only: bool = None):
    """获取所有记忆（含 id，用于管理页面）

    Args:
        limit: 可选，限制返回数量
        layer: 可选，筛选指定层级（1=原始碎片, 2=事件记忆, 3=核心记忆）
        active_only: 可选，是否只返回 is_active=true 的记忆
    """
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        conditions = []
        params = []
        param_idx = 1

        if layer is not None:
            conditions.append(f"layer = ${param_idx}")
            params.append(layer)
            param_idx += 1

        if active_only is not None:
            conditions.append(f"is_active = ${param_idx}")
            params.append(active_only)
            param_idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        if limit is not None:
            limit_clause = f"LIMIT ${param_idx}"
            params.append(limit)
        else:
            limit_clause = ""

        rows = await conn.fetch(f"""
            SELECT id, content, importance, source_session, created_at,
                   layer, title, is_active, merged_from, event_date, superseded_by
            FROM memories
            {where_clause}
            ORDER BY id
            {limit_clause}
        """, *params)
        return [dict(r) for r in rows]


async def delete_archived_memory(memory_id: int):
    """永久删除一条未被合并关系引用的已归档记忆。"""
    return await delete_archived_memories_batch([memory_id])


async def delete_archived_memories_batch(memory_ids: list):
    """批量永久删除未被合并关系引用的已归档记忆。"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """WITH targets AS (
               SELECT memory.id,
                          memory.superseded_by IS NOT NULL
                          OR EXISTS (
                              SELECT 1
                              FROM memories AS parent
                              WHERE memory.id = ANY(
                                  COALESCE(parent.merged_from, '{}'::int[])
                              )
                          )
                          OR EXISTS (
                              SELECT 1
                              FROM memories AS predecessor
                              WHERE predecessor.superseded_by = memory.id
                          ) AS protected
                   FROM memories AS memory
                   WHERE memory.id = ANY($1::int[])
                     AND memory.is_active = FALSE
               ), deleted AS (
                   DELETE FROM memories AS memory
                   USING targets
                   WHERE memory.id = targets.id AND NOT targets.protected
                   RETURNING memory.id
               )
               SELECT (SELECT COUNT(*) FROM deleted)::int AS deleted,
                      (SELECT COUNT(*) FROM targets WHERE protected)::int AS protected""",
            memory_ids,
        )
        return {
            "deleted": result["deleted"] if result else 0,
            "protected": result["protected"] if result else 0,
        }


async def soft_delete_memories_batch(memory_ids: list):
    """批量软删除记忆，返回实际转为不活跃的数量。"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """UPDATE memories
               SET is_active = FALSE
               WHERE id = ANY($1::int[]) AND is_active = TRUE""",
            memory_ids,
        )
        return int(result.split()[-1]) if result else 0


async def restore_archived_memories_batch(memory_ids: list):
    """批量恢复普通归档记忆；版本前驱必须走显式撤销。"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """UPDATE memories
               SET is_active = TRUE
               WHERE id = ANY($1::int[])
                 AND is_active = FALSE
                 AND superseded_by IS NULL""",
            memory_ids,
        )
        return int(result.split()[-1]) if result else 0


async def restore_archived_memory(memory_id: int):
    """恢复一条普通归档记忆；版本前驱必须走显式撤销。"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id, is_active, superseded_by
                FROM memories
                WHERE id = $1
                FOR UPDATE
                """,
                memory_id,
            )
            if not row:
                return {"status": "not_found"}
            if row["superseded_by"] is not None:
                return {"status": "superseded"}
            if row["is_active"] is False:
                await conn.execute(
                    "UPDATE memories SET is_active = TRUE WHERE id = $1",
                    memory_id,
                )
            return {"status": "ok"}


async def undo_memory_supersession(memory_id: int):
    """Restore one superseded predecessor and keep its successor untouched."""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id, superseded_by
                FROM memories
                WHERE id = $1
                FOR UPDATE
                """,
                memory_id,
            )
            if not row:
                return {"status": "not_found"}
            if row["superseded_by"] is None:
                return {"status": "not_superseded"}
            await conn.execute(
                """
                UPDATE memories
                SET is_active = TRUE, superseded_by = NULL
                WHERE id = $1
                """,
                memory_id,
            )
            return {"status": "ok"}


# ============================================================
# 三层记忆架构（碎片/事件/核心）
# ============================================================

async def get_fragments_by_date(event_date):
    """获取指定日期的原始碎片（用于每日整理）"""
    # 把本地日期转成UTC时间范围，避免DATE()用UTC截断导致日期偏移
    local_tz = dt_timezone(timedelta(hours=shared.TIMEZONE_HOURS))
    start_utc = datetime(event_date.year, event_date.month, event_date.day, tzinfo=local_tz).astimezone(dt_timezone.utc)
    end_utc = start_utc + timedelta(days=1)

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance, created_at
            FROM memories
            WHERE layer = 1 AND is_active = TRUE
            AND created_at >= $1 AND created_at < $2
            ORDER BY created_at
        """, start_utc, end_utc)
        return [dict(r) for r in rows]


async def deactivate_memories(memory_ids: list):
    """将记忆标记为不活跃（合并后的碎片）"""
    if not memory_ids:
        return
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE memories SET is_active = FALSE
            WHERE id = ANY($1::int[])
        """, memory_ids)


async def create_consolidated_events(events: list, expected_fragment_ids: list):
    """原子地创建整理事件并归档被完整覆盖的来源碎片。

    模型结果必须完整且唯一地覆盖 expected_fragment_ids。事务开始后再次锁定并
    验证所有来源仍是活跃碎片，避免并发整理或手动操作造成部分提交。
    """
    expected_ids = [int(memory_id) for memory_id in expected_fragment_ids]
    expected_set = set(expected_ids)
    if not expected_ids:
        return []
    if len(expected_ids) != len(expected_set):
        raise ValueError("expected_fragment_ids 存在重复")

    merged_ids = []
    seen_ids = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("整理事件必须是JSON对象")
        if not isinstance(event.get("content"), str) or not event["content"].strip():
            raise ValueError("整理事件缺少 content")
        event_ids = event.get("merged_ids", [])
        if not event_ids:
            raise ValueError("整理事件缺少 merged_ids")
        for memory_id in event_ids:
            if isinstance(memory_id, bool) or not isinstance(memory_id, int):
                raise ValueError(f"整理事件包含非法碎片ID: {memory_id}")
            if memory_id not in expected_set:
                raise ValueError(f"整理事件引用了范围外碎片: {memory_id}")
            if memory_id in seen_ids:
                raise ValueError(f"碎片被多个事件重复引用: {memory_id}")
            seen_ids.add(memory_id)
            merged_ids.append(memory_id)

    missing_ids = expected_set - seen_ids
    if missing_ids:
        raise ValueError(f"整理事件未覆盖全部碎片: {sorted(missing_ids)}")

    pool = await db_core.get_pool()
    created = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch("""
                SELECT id
                FROM memories
                WHERE id = ANY($1::int[])
                  AND layer = 1
                  AND is_active = TRUE
                FOR UPDATE
            """, expected_ids)
            active_ids = {int(row["id"]) for row in rows}
            if active_ids != expected_set:
                unavailable = sorted(expected_set - active_ids)
                raise RuntimeError(f"部分来源碎片已被其他操作修改: {unavailable}")

            for event in events:
                row = await conn.fetchrow("""
                    INSERT INTO memories (
                        content, importance, layer, title,
                        is_active, merged_from, event_date
                    )
                    VALUES ($1, $2, 2, $3, TRUE, $4, $5)
                    RETURNING id
                """,
                    event.get("content", ""),
                    event.get("importance", 5),
                    event.get("title", ""),
                    event["merged_ids"],
                    event.get("event_date"),
                )
                if not row:
                    raise RuntimeError("创建事件记忆失败")
                created.append({
                    "id": int(row["id"]),
                    "content": event.get("content", ""),
                })

            result = await conn.execute("""
                UPDATE memories
                SET is_active = FALSE
                WHERE id = ANY($1::int[])
                  AND layer = 1
                  AND is_active = TRUE
            """, merged_ids)
            updated = int(result.split()[-1]) if result else 0
            if updated != len(expected_ids):
                raise RuntimeError(
                    f"归档来源碎片数量不符: expected={len(expected_ids)}, updated={updated}"
                )

    # embedding 失败不影响事件与来源碎片的原子提交，和旧逻辑保持一致。
    if shared.MEMORY_VECTOR_ENABLED and created:
        async with pool.acquire() as conn:
            for event in created:
                try:
                    embedding = await db_search.compute_embedding(event["content"])
                    if embedding:
                        await db_search.save_memory_embedding(conn, event["id"], embedding)
                except Exception as exc:
                    print(f"⚠️ 事件记忆embedding计算失败（id={event['id']}）: {exc}")

    return [event["id"] for event in created]


async def promote_to_core(memory_id: int, title: str = None):
    """将记忆升级为核心记忆"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        if title:
            await conn.execute("""
                UPDATE memories SET layer = 3, title = $2
                WHERE id = $1
            """, memory_id, title)
        else:
            await conn.execute("""
                UPDATE memories SET layer = 3
                WHERE id = $1
            """, memory_id)


async def merge_memories(memory_ids: list, new_title: str, new_content: str,
                         importance: int, layer: int = 2):
    """合并多条记忆为一条新记忆"""
    if not memory_ids:
        return None

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        # 获取原记忆的日期（取最早的；来源含事件记忆时优先其真实发生日，而非整理日）
        rows = await conn.fetch("""
            SELECT MIN(COALESCE(event_date, DATE(created_at))) as event_date
            FROM memories WHERE id = ANY($1::int[])
        """, memory_ids)
        event_date = rows[0]['event_date'] if rows else None

        # 创建新记忆
        row = await conn.fetchrow("""
            INSERT INTO memories (content, importance, layer, title, is_active, merged_from, event_date)
            VALUES ($1, $2, $3, $4, TRUE, $5, $6)
            RETURNING id
        """, new_content, importance, layer, new_title, memory_ids, event_date)

        new_id = row['id'] if row else None

        # 向量搜索：计算并保存 embedding
        if shared.MEMORY_VECTOR_ENABLED and new_id:
            try:
                embedding = await db_search.compute_embedding(new_content)
                if embedding:
                    await db_search.save_memory_embedding(conn, new_id, embedding)
            except Exception as e:
                print(f"⚠️ 合并记忆embedding计算失败（id={new_id}）: {e}")

        # 将原记忆标记为不活跃
        if new_id:
            await deactivate_memories(memory_ids)

        return new_id


async def check_duplicate_memory(new_content: str, threshold: float = 0.7) -> dict:
    """检查新记忆是否与现有记忆重复

    三层去重策略：
    1. 精确匹配：内容完全相同
    2. 包含关系：新内容包含旧内容，或旧内容包含新内容
    3. 关键词重叠度：Jaccard 相似度 > threshold

    Returns:
        {
            "is_duplicate": bool,
            "reason": str,  # "exact" / "containment" / "similarity"
            "matched_id": int or None,
            "similarity": float or None
        }
    """
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        # 获取所有活跃记忆
        rows = await conn.fetch("""
            SELECT id, content FROM memories
            WHERE is_active = TRUE
        """)

        new_content_lower = new_content.strip().lower()
        new_keywords = set(db_search.extract_search_keywords(new_content))

        for row in rows:
            old_content = row['content']
            old_content_lower = old_content.strip().lower()

            # 第一层：精确匹配
            if new_content_lower == old_content_lower:
                return {
                    "is_duplicate": True,
                    "reason": "exact",
                    "matched_id": row['id'],
                    "similarity": 1.0
                }

            # 第二层：包含关系
            if new_content_lower in old_content_lower:
                return {
                    "is_duplicate": True,
                    "reason": "containment",
                    "matched_id": row['id'],
                    "similarity": len(new_content) / len(old_content)
                }
            if old_content_lower in new_content_lower:
                return {
                    "is_duplicate": True,
                    "reason": "containment_update",
                    "matched_id": row['id'],
                    "similarity": len(old_content) / len(new_content)
                }

            # 第三层：关键词重叠度（Jaccard 相似度）
            old_keywords = set(db_search.extract_search_keywords(old_content))
            if new_keywords and old_keywords:
                intersection = new_keywords & old_keywords
                union = new_keywords | old_keywords
                similarity = len(intersection) / len(union) if union else 0

                if similarity > threshold:
                    return {
                        "is_duplicate": True,
                        "reason": "similarity",
                        "matched_id": row['id'],
                        "similarity": similarity
                    }

        return {
            "is_duplicate": False,
            "reason": None,
            "matched_id": None,
            "similarity": None
        }


async def update_memory_with_layer(memory_id: int, content: str = None,
                                    importance: int = None, title: str = None,
                                    layer: int = None, is_active: bool = None):
    """更新记忆（支持三层架构新字段）"""
    updates = []
    params = []
    param_idx = 2  # $1 给 memory_id

    if content is not None:
        updates.append(f"content = ${param_idx}")
        params.append(content)
        param_idx += 1

    if importance is not None:
        updates.append(f"importance = ${param_idx}")
        params.append(importance)
        param_idx += 1

    if title is not None:
        updates.append(f"title = ${param_idx}")
        params.append(title)
        param_idx += 1

    if layer is not None:
        updates.append(f"layer = ${param_idx}")
        params.append(layer)
        param_idx += 1

    if is_active is not None:
        updates.append(f"is_active = ${param_idx}")
        params.append(is_active)
        param_idx += 1

    if not updates:
        return

    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = $1",
            memory_id, *params
        )


async def get_layer_statistics():
    """获取各层记忆的统计数据"""
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                layer,
                COUNT(*) as count,
                COUNT(*) FILTER (WHERE is_active = TRUE) as active_count
            FROM memories
            GROUP BY layer
            ORDER BY layer
        """)

        stats = {
            "layer_1": {"total": 0, "active": 0},  # 原始碎片
            "layer_2": {"total": 0, "active": 0},  # 事件记忆
            "layer_3": {"total": 0, "active": 0},  # 核心记忆
        }

        for row in rows:
            layer = row['layer'] or 1  # 默认为层级1
            key = f"layer_{layer}"
            if key in stats:
                stats[key] = {
                    "total": row['count'],
                    "active": row['active_count']
                }

        return stats


async def cleanup_old_fragments(days: int = 30):
    """清理指定天数前的归档碎片

    只清理满足以下条件的记忆：
    - layer = 1（原始碎片）
    - is_active = FALSE（已归档）
    - created_at 在 days 天之前
    - 不属于自动取代版本链

    Returns:
        {"deleted": 删除数量, "revert_disabled": 结束撤回能力的父记忆数量}
    """
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        cutoff_date = datetime.now() - timedelta(days=days)

        async with conn.transaction():
            rows = await conn.fetch("""
                SELECT memory.id
                FROM memories AS memory
                WHERE memory.layer = 1
                  AND memory.is_active = FALSE
                  AND memory.created_at < $1
                  AND memory.superseded_by IS NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM memories AS predecessor
                      WHERE predecessor.superseded_by = memory.id
                  )
                FOR UPDATE
            """, cutoff_date)
            fragment_ids = [int(row["id"]) for row in rows]
            if not fragment_ids:
                return {"deleted": 0, "revert_disabled": 0}

            result = await conn.execute("""
                UPDATE memories
                SET merged_from = NULL
                WHERE merged_from && $1::int[]
            """, fragment_ids)
            revert_disabled = int(result.split()[-1]) if result else 0

            result = await conn.execute("""
                DELETE FROM memories
                WHERE id = ANY($1::int[])
            """, fragment_ids)
            deleted = int(result.split()[-1]) if result else 0
            return {
                "deleted": deleted,
                "revert_disabled": revert_disabled,
            }


async def revert_merge(memory_id: int):
    """撤回合并操作

    恢复原始碎片（is_active = TRUE），删除合并后的事件记忆

    Args:
        memory_id: 要撤回的事件记忆ID

    Returns:
        {"status": "ok", "restored": 恢复的碎片数量}
        或 {"error": "错误信息"}
    """
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow("""
                SELECT memory.id, memory.layer, memory.merged_from,
                       memory.superseded_by IS NOT NULL OR EXISTS (
                           SELECT 1
                           FROM memories AS predecessor
                           WHERE predecessor.superseded_by = memory.id
                       ) AS protected_by_supersession
                FROM memories AS memory
                WHERE memory.id = $1
                FOR UPDATE OF memory
            """, memory_id)

            if not row:
                return {"error": "记忆不存在"}

            if row['layer'] != 2:
                return {"error": "只能撤回事件记忆的合并"}

            if row.get('protected_by_supersession', False):
                return {"error": "版本链中的记忆不能撤回合并"}

            merged_from = row['merged_from']
            if not merged_from or len(merged_from) == 0:
                return {"error": "没有完整的合并来源，无法撤回"}

            source_rows = await conn.fetch("""
                SELECT id
                FROM memories
                WHERE id = ANY($1::int[])
                FOR UPDATE
            """, merged_from)
            source_ids = {int(source["id"]) for source in source_rows}
            expected_ids = set(merged_from)
            if source_ids != expected_ids:
                missing = sorted(expected_ids - source_ids)
                return {
                    "error": f"合并来源不完整，缺少 {len(missing)} 条，未执行撤回"
                }

            result = await conn.execute("""
                UPDATE memories SET is_active = TRUE
                WHERE id = ANY($1::int[])
            """, merged_from)
            restored = int(result.split()[-1]) if result else 0
            if restored != len(expected_ids):
                raise RuntimeError(
                    f"恢复来源数量不符: expected={len(expected_ids)}, restored={restored}"
                )

            await conn.execute("""
                DELETE FROM memories WHERE id = $1
            """, memory_id)

            return {"status": "ok", "restored": restored}
