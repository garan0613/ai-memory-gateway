"""Memory CRUD, import, lifecycle, and backfill routes."""

import asyncio
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import shared
from db import core as db_core
from db import search as db_search
from db import conversations as db_conversations
from db import memories as db_memories
from memory_extractor import score_memories

logger = logging.getLogger(__name__)

bootstrap_router = APIRouter()
router = APIRouter()
maintenance_router = APIRouter()

# ============================================================
# 记忆管理接口
# ============================================================


@bootstrap_router.get("/import/seed-memories")
async def import_seed_memories():
    """一次性导入预置记忆（从 seed_memories.py）"""
    try:
        from seed_memories import run_seed_import
        result = await run_seed_import()
        return result
    except ImportError:
        return {"error": "未找到 seed_memories.py，请参考 seed_memories_example.py 创建"}
    except Exception:
        return shared._api_failure("导入预置记忆失败")


@bootstrap_router.get("/export/memories")
async def export_memories():
    """
    导出所有记忆为 JSON（用于备份或迁移）
    浏览器访问这个地址就会返回所有记忆数据
    """
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}

    try:
        memories = await db_memories.get_all_memories()
        # 库内 id 在备份里叫 backup_id：只是恢复时重建 merged_from 关系的映射键，
        # 不承诺导入后保持同一 id；日期转成字符串
        for mem in memories:
            mem["backup_id"] = mem.pop("id")
            if mem.get("created_at"):
                mem["created_at"] = str(mem["created_at"])
            if mem.get("event_date"):
                mem["event_date"] = str(mem["event_date"])

        return {
            "schema_version": 3,
            "total": len(memories),
            "exported_at": str(datetime.now()),
            "memories": memories,
        }
    except db_memories.BrokenMergeReferencesError as e:
        return {
            "error": f"检测到 {e.count} 条记忆的合并来源已失效，可修复断裂引用后重新导出",
            "code": "broken_merge_references",
            "count": e.count,
        }
    except db_memories.BrokenSupersessionReferencesError as e:
        return {
            "error": str(e),
            "code": "broken_supersession_references",
            "count": e.count,
        }
    except Exception:
        return shared._api_failure("导出记忆失败")


@bootstrap_router.post("/api/memories/repair-broken-references")
async def api_repair_broken_merge_references():
    """显式清除失效的合并来源关系，使完整备份可以重新导出。"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    repaired = await db_memories.repair_broken_merge_references()
    return {"status": "ok", "repaired": repaired}


# ============================================================
# 管理 API
# ============================================================

@router.post("/api/memories")
async def api_create_memory(request: Request):
    """由 MCP、脚本或迁移工具显式写入一条记忆。"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
    if not isinstance(data, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象"})

    content = data.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return JSONResponse(status_code=400, content={"error": "content 必须是非空字符串"})
    content = content.strip()

    importance = data.get("importance", 5)
    if (
        isinstance(importance, bool)
        or not isinstance(importance, int)
        or not 1 <= importance <= 10
    ):
        return JSONResponse(status_code=400, content={"error": "importance 必须是 1 到 10 的整数"})

    layer = data.get("layer", 1)
    if isinstance(layer, bool) or not isinstance(layer, int) or layer not in (1, 2, 3):
        return JSONResponse(status_code=400, content={"error": "layer 必须是 1、2 或 3"})

    title = data.get("title")
    if title is not None and not isinstance(title, str):
        return JSONResponse(status_code=400, content={"error": "title 必须是字符串或 null"})
    title = (title.strip() or None) if isinstance(title, str) else None

    source_session = data.get("source_session", "")
    if not isinstance(source_session, str):
        return JSONResponse(status_code=400, content={"error": "source_session 必须是字符串"})
    source_session = source_session.strip()

    external_id = data.get("external_id")
    if external_id is not None and (
        not isinstance(external_id, str) or not external_id.strip()
    ):
        return JSONResponse(status_code=400, content={"error": "external_id 必须是非空字符串"})
    external_id = external_id.strip() if isinstance(external_id, str) else None

    event_date = data.get("event_date")
    if event_date is not None:
        if not isinstance(event_date, str):
            return JSONResponse(status_code=400, content={"error": "event_date 必须是 YYYY-MM-DD 字符串或 null"})
        try:
            event_date = date.fromisoformat(event_date)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "event_date 必须是有效的 YYYY-MM-DD 日期"})

    try:
        memory_id = await db_memories.save_memory(
            content=content,
            importance=importance,
            source_session=source_session,
            title=title,
            layer=layer,
            event_date=event_date,
            external_id=external_id,
        )
        if memory_id is None:
            existing = await db_memories.get_memory_by_external_id(external_id)
            if existing is None:
                return JSONResponse(status_code=500, content={"error": "创建记忆失败"})
            for key in ("event_date", "created_at"):
                if existing.get(key):
                    existing[key] = existing[key].isoformat()
            return {
                "status": "ok",
                "inserted": False,
                "id": existing["id"],
                "memory": existing,
            }

        return {
            "status": "ok",
            "inserted": True,
            "id": memory_id,
            "memory": {
                "id": memory_id,
                "content": content,
                "importance": importance,
                "title": title,
                "layer": layer,
                "event_date": event_date.isoformat() if event_date else None,
                "source_session": source_session,
                "external_id": external_id,
                "is_active": True,
            },
        }
    except Exception:
        logger.exception("Explicit memory creation failed")
        return JSONResponse(status_code=500, content={"error": "创建记忆失败"})


@router.get("/api/memories")
async def api_get_memories(layer: int = None, active_only: bool = None):
    """获取所有记忆（管理页面用）

    Query params:
        layer: 筛选层级（1=碎片, 2=事件, 3=核心）
        active_only: 是否只返回活跃记忆
    """
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    memories = await db_memories.get_all_memories_detail(layer=layer, active_only=active_only)
    tz_offset = timezone(timedelta(hours=shared.TIMEZONE_HOURS))
    for m in memories:
        if m.get("created_at"):
            dt = m["created_at"]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            m["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
    # 获取层级统计
    try:
        layer_stats = await db_memories.get_layer_statistics()
    except Exception:
        layer_stats = None

    result = {"memories": memories}
    if layer_stats:
        result["layer_stats"] = layer_stats
    return result


async def _run_memory_search(q: str, limit: int):
    """语义搜索记忆（Dashboard用，走后端 search_memories）"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    if not q.strip():
        return {"error": "搜索关键词不能为空", "results": []}
    try:
        results, mode = await db_memories.search_memories_with_mode(q.strip(), limit)
        tz_offset = timezone(timedelta(hours=shared.TIMEZONE_HOURS))
        out = []
        for r in results:
            item = dict(r)
            if item.get("created_at"):
                dt = item["created_at"]
                if hasattr(dt, 'tzinfo'):
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    item["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
            out.append(item)
        response = {"results": out, "total": len(out), "mode": mode}
        if mode == "keyword":
            response["warning"] = (
                "向量模型本次未生效，已按关键词搜索；"
                "请检查向量搜索开关和 Embedding 配置"
            )
        return response
    except Exception:
        return shared._api_failure("搜索失败", results=[])


@router.get("/api/memories/search")
async def api_search_memories(q: str = "", limit: int = 20):
    return await _run_memory_search(q, limit)


@router.post("/api/memories/search")
async def api_search_memories_post(request: Request):
    """POST 变体避免查询原文进入 URL 与访问日志。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象"})
    q = body.get("q", "")
    limit = body.get("limit", 20)
    if not isinstance(q, str) or not isinstance(limit, int) or isinstance(limit, bool):
        return JSONResponse(status_code=400, content={"error": "q 必须是字符串，limit 必须是整数"})
    return await _run_memory_search(q, limit)


@router.put("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, request: Request):
    """更新单条记忆（支持 content / importance / title / layer）"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    await db_memories.update_memory_with_layer(
        memory_id,
        content=data.get("content"),
        importance=data.get("importance"),
        title=data.get("title"),
        layer=data.get("layer"),
    )
    return {"status": "ok", "id": memory_id}


@router.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int, soft: bool = True):
    """删除单条记忆

    Query params:
        soft: true=可恢复软删除，false=永久删除已归档记忆
    """
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    if soft:
        await db_memories.update_memory_with_layer(memory_id, is_active=False)
    else:
        result = await db_memories.delete_archived_memory(memory_id)
        if result["protected"]:
            return JSONResponse(
                status_code=409,
                content={"error": "该记忆仍属于合并或版本链，不能永久删除"},
            )
        if not result["deleted"]:
            return JSONResponse(
                status_code=400,
                content={"error": "只能永久删除已归档记忆"},
            )
    return {"status": "ok", "id": memory_id}


@router.post("/api/memories/batch-update")
async def api_batch_update(request: Request):
    """批量更新记忆"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    updates = data.get("updates", [])
    if not updates:
        return {"error": "没有要更新的记忆"}
    for item in updates:
        await db_memories.update_memory_with_layer(
            item["id"],
            content=item.get("content"),
            importance=item.get("importance"),
            title=item.get("title"),
            layer=item.get("layer"),
        )
    return {"status": "ok", "updated": len(updates)}


@router.post("/api/memories/batch-delete")
async def api_batch_delete(request: Request):
    """批量软删除活跃记忆，或永久删除已归档记忆。"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    ids = data.get("ids", [])
    if not ids:
        return {"error": "未选择记忆"}
    soft = data.get("soft", True)
    if not isinstance(soft, bool):
        return JSONResponse(
            status_code=400,
            content={"error": "soft 必须是布尔值"},
        )
    if soft:
        deleted = await db_memories.soft_delete_memories_batch(ids)
        protected = 0
    else:
        result = await db_memories.delete_archived_memories_batch(ids)
        deleted = result["deleted"]
        protected = result["protected"]
    return {"status": "ok", "deleted": deleted, "protected": protected}


@router.post("/api/memories/batch-restore")
async def api_batch_restore(request: Request):
    """批量恢复已归档记忆。"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    ids = data.get("ids", [])
    if not ids:
        return {"error": "未选择记忆"}
    restored = await db_memories.restore_archived_memories_batch(ids)
    return {"status": "ok", "restored": restored}


# ============================================================
# 三层记忆架构：整理 / 合并 / 升级 / 统计
# ============================================================

CONSOLIDATION_PROMPT = """
你是记忆整理助手。请将以下对话碎片整理成完整的事件记录。

要求：
1. 按主题/事件分组，相关的碎片合并到一起
2. 每个事件一条记录，不要太细碎也不要太笼统
3. 每条记录包含：标题（10字内）+ 完整描述
4. 合并重复内容，保留重要细节
5. 保留原文中的主观感受、情绪表达和个人化用语，不要改写为客观陈述或第三方总结
6. content字段中不要使用双引号，用单引号或书名号代替
7. 每个输入碎片ID必须且只能出现在一个事件的merged_ids中，不得遗漏或重复；无法与其他内容合并的碎片也要单独生成一条事件

碎片记忆：
{fragments}

请用 JSON 格式输出：
[
  {{
    "title": "事件标题（10字内）",
    "content": "完整的事件描述",
    "importance": 5,
    "merged_ids": [1, 2, 3]
  }}
]

只输出 JSON，不要其他内容。确保 JSON 语法正确。
"""

# 整理状态（异步执行，防重入）
_consolidate_status = {
    "running": False,
    "started_at": None,
    "result": None,
    "error": None,
}


class ConsolidationError(Exception):
    """记忆整理失败，调用方可以安全地保留全部原始碎片。"""


class ConsolidationTruncatedError(ConsolidationError):
    """模型输出达到上限，不能把残缺 JSON 当成有效结果。"""


class ConsolidationCoverageError(ConsolidationError):
    """模型返回的 merged_ids 没有完整且唯一地覆盖当前批次。"""


def _parse_json_array(content):
    """解析完整 JSON 数组，允许代码围栏和前后说明，不接受半截数组。"""
    text = (content or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline >= 0 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    candidates = [text]
    first_array = text.find("[")
    if first_array > 0:
        candidates.append(text[first_array:])

    last_error = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value, _ = json.JSONDecoder(strict=False).raw_decode(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(value, list):
            return value
        last_error = ValueError("AI 返回的 JSON 顶层不是数组")

    detail = str(last_error) if last_error else "响应为空或未包含 JSON 数组"
    raise ConsolidationError(f"JSON解析失败: {detail}")


def _completion_metadata(data, max_tokens):
    """读取停止原因和 usage；部分兼容上游只返回其中一项。"""
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None:
        # 推理模型的思考 token 会计入 completion_tokens，可能超过 max_tokens
        # 而正文完整。上游明确报 length 才算截断，报 stop 的完整回复直接放行
        truncated = finish_reason == "length"
    else:
        truncated = isinstance(completion_tokens, int) and completion_tokens >= max_tokens
    return {
        "content": (choice.get("message") or {}).get("content") or "",
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "truncated": truncated,
    }


async def _post_consolidation_completion(client, prompt, model, max_tokens, label):
    """调用整理模型；仅对 429 做有界重试。"""
    last_error = None
    for attempt in range(3):
        response = await client.post(
            shared.API_BASE_URL,
            headers={
                "Authorization": f"Bearer {shared.get_memory_api_key()}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
        )

        if response.status_code == 429:
            wait_time = (attempt + 1) * 10
            print(f"⚠️ {label} API 429限流，{wait_time}秒后重试（第{attempt + 1}次）")
            last_error = f"429 Too Many Requests（重试{attempt + 1}次）"
            await asyncio.sleep(wait_time)
            continue

        if response.status_code != 200:
            raise ConsolidationError(
                f"{label} API调用失败: HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ConsolidationError(f"{label} API返回的响应不是JSON: {exc}") from exc

        metadata = _completion_metadata(data, max_tokens)
        usage_text = (
            f"{metadata['completion_tokens']}/{max_tokens}"
            if metadata["completion_tokens"] is not None
            else f"未知/{max_tokens}"
        )
        reasoning_text = (
            f"，其中推理 {metadata['reasoning_tokens']}"
            if metadata["reasoning_tokens"] is not None
            else ""
        )
        print(
            f"🧩 {label}模型返回 {len(metadata['content'])} 字符，"
            f"finish_reason={metadata['finish_reason']}，"
            f"completion_tokens={usage_text}{reasoning_text}",
            flush=True,
        )
        return metadata

    raise ConsolidationError(f"{label} API调用失败: {last_error}")


async def _request_consolidation_events(client, fragments, model, max_tokens):
    fragments_text = "\n".join([
        f"[ID={fragment['id']}] "
        f"({fragment['created_at'].strftime('%m-%d') if hasattr(fragment['created_at'], 'strftime') else str(fragment['created_at'])[:10]}) "
        f"{fragment['content']}"
        for fragment in fragments
    ])
    prompt = CONSOLIDATION_PROMPT.format(fragments=fragments_text)
    metadata = await _post_consolidation_completion(
        client, prompt, model, max_tokens, "整理"
    )

    if metadata["truncated"]:
        raise ConsolidationTruncatedError(
            f"整理输出达到上限（finish_reason={metadata['finish_reason']}，"
            f"completion_tokens={metadata['completion_tokens']}/{max_tokens}）"
        )

    try:
        return _parse_json_array(metadata["content"])
    except ConsolidationError as original_error:
        repair_prompt = (
            "请修复以下JSON的语法错误，只输出修复后的完整JSON数组，不要删减任何事件，"
            "不要添加其他内容：\n"
            f"{metadata['content']}"
        )
        repaired = await _post_consolidation_completion(
            client, repair_prompt, model, max_tokens, "JSON修复"
        )
        if repaired["truncated"]:
            raise ConsolidationTruncatedError(
                f"JSON修复输出达到上限（finish_reason={repaired['finish_reason']}，"
                f"completion_tokens={repaired['completion_tokens']}/{max_tokens}）"
            ) from original_error
        try:
            return _parse_json_array(repaired["content"])
        except ConsolidationError as repair_error:
            raise ConsolidationError(
                f"JSON解析失败（AI修复也失败）: {repair_error}"
            ) from original_error


def _normalize_consolidation_events(events, fragments, event_date):
    """校验模型结果，并把 merged_ids 规范为唯一、完整的整数集合。"""
    expected_ids = [int(fragment["id"]) for fragment in fragments]
    expected_set = set(expected_ids)
    if len(expected_ids) != len(expected_set):
        raise ConsolidationCoverageError("输入碎片ID存在重复")

    normalized = []
    seen_ids = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ConsolidationCoverageError(f"第 {index + 1} 个事件不是JSON对象")

        raw_ids = event.get("merged_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ConsolidationCoverageError(f"第 {index + 1} 个事件缺少 merged_ids")

        merged_ids = []
        for raw_id in raw_ids:
            if isinstance(raw_id, bool):
                raise ConsolidationCoverageError(f"非法碎片ID: {raw_id}")
            if isinstance(raw_id, int):
                memory_id = raw_id
            elif isinstance(raw_id, str) and raw_id.strip().isdigit():
                memory_id = int(raw_id)
            else:
                raise ConsolidationCoverageError(f"非法碎片ID: {raw_id}")
            if memory_id not in expected_set:
                raise ConsolidationCoverageError(f"模型返回了批次外碎片ID: {memory_id}")
            if memory_id in seen_ids:
                raise ConsolidationCoverageError(f"碎片ID被重复合并: {memory_id}")
            seen_ids.add(memory_id)
            merged_ids.append(memory_id)

        raw_content = event.get("content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ConsolidationCoverageError(f"第 {index + 1} 个事件缺少 content")
        content = raw_content.strip()

        try:
            importance = int(event.get("importance", 5))
        except (TypeError, ValueError):
            importance = 5

        normalized.append({
            "title": str(event.get("title", "")).strip(),
            "content": content,
            "importance": max(1, min(10, importance)),
            "merged_ids": merged_ids,
            "event_date": event_date,
        })

    missing_ids = expected_set - seen_ids
    if missing_ids:
        raise ConsolidationCoverageError(
            f"模型遗漏碎片ID: {sorted(missing_ids)}"
        )
    return normalized


async def _consolidate_fragment_batch(client, fragments, event_date, model, max_tokens):
    """整理一个批次；输出过长或覆盖不全时递归二分，不写数据库。"""
    try:
        raw_events = await _request_consolidation_events(
            client, fragments, model, max_tokens
        )
        events = _normalize_consolidation_events(
            raw_events, fragments, event_date
        )
        return {
            "events": events,
            "batches": 1,
            "split_retries": 0,
        }
    except (ConsolidationTruncatedError, ConsolidationCoverageError) as exc:
        if len(fragments) <= 1:
            raise ConsolidationError(
                f"单条碎片仍无法安全整理（ID={fragments[0]['id']}）: {exc}"
            ) from exc

        midpoint = len(fragments) // 2
        print(
            f"⚠️ 整理批次需要拆分（{len(fragments)} 条）: {exc}",
            flush=True,
        )
        left = await _consolidate_fragment_batch(
            client, fragments[:midpoint], event_date, model, max_tokens
        )
        right = await _consolidate_fragment_batch(
            client, fragments[midpoint:], event_date, model, max_tokens
        )
        return {
            "events": left["events"] + right["events"],
            "batches": left["batches"] + right["batches"],
            "split_retries": left["split_retries"] + right["split_retries"] + 1,
        }


async def consolidate_memories_for_date_range(start_date, end_date):
    """按本地日期整理碎片；全部批次成功后再用单个事务写库。"""
    # 跟提取、评分共用同一份模型配置，面板热更新也一起跟。
    # 别 fallback 到 DEFAULT_MODEL，那是主聊天模型，拿它跑后台批处理会按主力模型计价
    import memory_extractor as _me_mod
    consolidation_model = _me_mod.MEMORY_MODEL
    consolidation_max_tokens = int(os.getenv("MEMORY_MAX_TOKENS", "4000"))

    try:
        all_fragments = []
        all_events = []
        batches = 0
        split_retries = 0
        days_processed = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            current_date = start_date
            while current_date <= end_date:
                fragments = await db_memories.get_fragments_by_date(current_date)
                if fragments:
                    result = await _consolidate_fragment_batch(
                        client,
                        fragments,
                        current_date,
                        consolidation_model,
                        consolidation_max_tokens,
                    )
                    all_fragments.extend(fragments)
                    all_events.extend(result["events"])
                    batches += result["batches"]
                    split_retries += result["split_retries"]
                    days_processed += 1
                current_date += timedelta(days=1)

        if not all_fragments:
            return {
                "status": "no_fragments",
                "start_date": str(start_date),
                "end_date": str(end_date),
            }

        expected_fragment_ids = [int(fragment["id"]) for fragment in all_fragments]
        created_ids = await db_memories.create_consolidated_events(
            all_events, expected_fragment_ids
        )

        return {
            "status": "ok",
            "start_date": str(start_date),
            "end_date": str(end_date),
            "days_processed": days_processed,
            "batches_processed": batches,
            "split_retries": split_retries,
            "fragments_processed": len(all_fragments),
            "events_created": len(created_ids),
        }
    except Exception:
        logger.exception("Memory consolidation failed")
        return {"status": "error", "error": "整理失败，原始碎片未归档"}


@router.post("/api/memories/consolidate")
async def api_manual_consolidate(request: Request):
    """手动触发整理（异步，立即返回）

    Body:
        start_date: 开始日期（YYYY-MM-DD 格式）
        end_date: 结束日期（YYYY-MM-DD 格式）
        或
        date: 单个日期（兼容旧版）
    """
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    if _consolidate_status.get("running"):
        return {"status": "already_running", "started_at": _consolidate_status.get("started_at")}

    data = await request.json()

    # 解析日期参数
    if "date" in data and "start_date" not in data:
        start_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
        end_date = start_date
    else:
        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")

        if not start_date_str or not end_date_str:
            return {"error": "请提供开始和结束日期"}

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        if start_date > end_date:
            return {"error": "开始日期不能晚于结束日期"}

    async def _run():
        _consolidate_status.update({"running": True, "started_at": f"{start_date}~{end_date}", "result": None, "error": None})
        try:
            result = await consolidate_memories_for_date_range(start_date, end_date)
            _consolidate_status["result"] = result
            print(f"[manual/consolidate] 整理 {start_date}~{end_date}: {result}")
        except Exception:
            logger.exception("Manual memory consolidation failed for %s~%s", start_date, end_date)
            _consolidate_status["error"] = "整理失败，原始碎片未归档"
        finally:
            _consolidate_status["running"] = False

    asyncio.create_task(_run())
    return {"status": "started", "start_date": str(start_date), "end_date": str(end_date)}


@router.get("/api/memories/consolidate/status")
async def api_consolidate_status():
    """查询整理任务状态"""
    return _consolidate_status


@router.post("/api/memories/{memory_id}/promote")
async def api_promote_to_core(memory_id: int, request: Request):
    """将记忆升级为核心记忆"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    data = await request.json()
    title = data.get("title")

    await db_memories.promote_to_core(memory_id, title=title)
    return {"status": "ok", "memory_id": memory_id, "layer": 3}


@router.post("/api/memories/merge")
async def api_merge_memories(request: Request):
    """手动合并多条记忆"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    data = await request.json()
    memory_ids = data.get("ids", [])
    new_title = data.get("title", "")
    new_content = data.get("content", "")
    importance = data.get("importance", 5)
    layer = data.get("layer", 2)

    if not memory_ids or not new_content:
        return {"error": "请提供记忆ID列表和合并后内容"}

    new_id = await db_memories.merge_memories(memory_ids, new_title, new_content, importance, layer)
    return {"status": "ok", "new_id": new_id, "merged": len(memory_ids)}


@router.post("/api/memories/check-duplicate")
async def api_check_duplicate(request: Request):
    """检查记忆是否重复"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    data = await request.json()
    content = data.get("content", "")
    threshold = data.get("threshold", 0.7)

    if not content:
        return {"error": "请提供记忆内容"}

    result = await db_memories.check_duplicate_memory(content, threshold)
    return result


@router.post("/api/memories/cleanup-fragments")
async def api_cleanup_fragments(request: Request):
    """清理指定天数前的归档碎片

    Body:
        days: 清理多少天前的归档碎片（默认30天）
    """
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    data = await request.json()
    days = data.get("days", 30)

    try:
        result = await db_memories.cleanup_old_fragments(days)
        return {
            "status": "ok",
            "deleted": result["deleted"],
            "revert_disabled": result["revert_disabled"],
            "days": days,
        }
    except Exception:
        return shared._api_failure("清理归档碎片失败")


@router.post("/api/memories/{memory_id}/revert-merge")
async def api_revert_merge(memory_id: int):
    """撤回合并操作：恢复原始碎片，删除合并后的事件记忆"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    try:
        result = await db_memories.revert_merge(memory_id)
        return result
    except Exception:
        return shared._api_failure("撤回合并失败")


@router.post("/api/memories/{memory_id}/restore")
async def api_restore_memory(memory_id: int):
    """恢复已归档的记忆（将 is_active 设为 TRUE）"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    try:
        result = await db_memories.restore_archived_memory(memory_id)
        if result["status"] == "not_found":
            return JSONResponse(status_code=404, content={"error": "记忆不存在"})
        if result["status"] == "superseded":
            return JSONResponse(
                status_code=409,
                content={"error": "该记忆属于版本链，请使用撤销取代"},
            )
        return {"status": "ok", "id": memory_id}
    except Exception:
        return shared._api_failure("恢复记忆失败")


@router.post("/api/memories/{memory_id}/undo-supersede")
async def api_undo_memory_supersession(memory_id: int):
    """撤销自动取代：恢复旧记忆，保留后继记忆。"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    try:
        result = await db_memories.undo_memory_supersession(memory_id)
        if result["status"] == "not_found":
            return JSONResponse(status_code=404, content={"error": "记忆不存在"})
        if result["status"] == "not_superseded":
            return JSONResponse(
                status_code=409,
                content={"error": "该记忆没有可撤销的取代关系"},
            )
        return result
    except Exception:
        return shared._api_failure("撤销取代失败")


@router.get("/api/memories/layer-stats")
async def api_layer_statistics():
    """获取各层记忆统计数据"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    try:
        stats = await db_memories.get_layer_statistics()
        return stats
    except Exception:
        return shared._api_failure("加载记忆统计失败")


@router.post("/import/text")
async def import_text_memories(request: Request):
    """从纯文本导入记忆（每行一条），可选自动评分"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}

    try:
        data = await request.json()
        lines = data.get("lines", [])
        skip_scoring = data.get("skip_scoring", False)

        if not lines:
            return {"error": "没有找到记忆条目"}

        if skip_scoring:
            scored = [{"content": t, "importance": 5} for t in lines]
        else:
            scored = await score_memories(lines)

        imported = 0
        skipped = 0

        for mem in scored:
            content = mem.get("content", "")
            if not content:
                continue

            pool = await db_core.get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE content = $1", content
                )

            if existing > 0:
                skipped += 1
                continue

            await db_memories.save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session="text-import",
            )
            imported += 1

        total = await db_memories.get_all_memories_count()
        return {
            "status": "done",
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }
    except Exception:
        return shared._api_failure("导入记忆失败")


@router.post("/import/memories")
async def import_memories(request: Request):
    """从 JSON 导入记忆（用于迁移或恢复备份）"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}

    try:
        data = await request.json()
        memories = data.get("memories", [])

        if not memories:
            return {"error": "没有找到记忆数据，请确认 JSON 格式正确"}

        # v2/v3 版本化备份：v3 额外恢复自动取代版本链
        schema_version = data.get("schema_version")
        if schema_version in (2, 3):
            try:
                return await db_memories.import_memories_v2(
                    memories,
                    schema_version=schema_version,
                )
            except ValueError as e:
                return {"error": f"备份校验失败：{e}"}

        # v1 旧格式（无 schema_version）：按碎片导入，尽量保留原 created_at
        imported = 0
        skipped = 0

        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue

            pool = await db_core.get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE content = $1", content
                )

            if existing > 0:
                skipped += 1
                continue

            await db_memories.save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session=mem.get("source_session", "json-import"),
                created_at=db_memories._parse_backup_datetime(mem.get("created_at")),
            )
            imported += 1

        total = await db_memories.get_all_memories_count()
        return {
            "status": "done",
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }
    except Exception:
        return shared._api_failure("导入记忆失败")


# ============================================================
# 记忆向量补算（带进度追踪）
# ============================================================

_backfill_mem_status = {
    "running": False,
    "total": 0,
    "done": 0,
    "error": None,
    "finished_at": None,
}

@maintenance_router.post("/api/admin/backfill-memory-embeddings")
async def api_backfill_memory_embeddings():
    """给已有记忆补算embedding（后台异步执行，前端轮询进度）"""
    if not shared.MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    if _backfill_mem_status["running"]:
        return {"error": "补算任务正在运行中，请等待完成"}

    try:
        total = await db_memories.get_pending_memory_embedding_count()
    except Exception:
        return shared._api_failure("查询待处理数量失败")

    if total == 0:
        return {"status": "done", "message": "所有记忆已有embedding，无需补算", "total": 0, "done": 0}

    _backfill_mem_status["running"] = True
    _backfill_mem_status["total"] = total
    _backfill_mem_status["done"] = 0
    _backfill_mem_status["error"] = None
    _backfill_mem_status["finished_at"] = None

    async def run_backfill():
        try:
            while _backfill_mem_status["running"]:
                updated = await db_memories.backfill_memory_embeddings(batch_size=20)
                _backfill_mem_status["done"] += updated

                if updated == 0:
                    break

                await asyncio.sleep(1)

            _backfill_mem_status["finished_at"] = datetime.now(timezone.utc).isoformat()
            print(f"✅ 记忆embedding补算完成：{_backfill_mem_status['done']}/{_backfill_mem_status['total']}")
        except Exception:
            logger.exception("Memory embedding backfill failed")
            _backfill_mem_status["error"] = "记忆向量补算失败"
        finally:
            _backfill_mem_status["running"] = False

    asyncio.create_task(run_backfill())
    return {"status": "started", "total": total}

@maintenance_router.get("/api/admin/backfill-memory-embeddings/status")
async def api_backfill_memory_embeddings_status():
    """查询记忆embedding补算进度"""
    return {
        "running": _backfill_mem_status["running"],
        "total": _backfill_mem_status["total"],
        "done": _backfill_mem_status["done"],
        "error": _backfill_mem_status["error"],
        "finished_at": _backfill_mem_status["finished_at"],
    }
