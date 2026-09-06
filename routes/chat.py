"""Core OpenAI-compatible chat routes."""

import asyncio
import json
import logging
import uuid

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

import shared
import partition_engine
import memory_pipeline
from db import core as db_core
from db import search as db_search
from db import conversations as db_conversations
from db import memories as db_memories

router = APIRouter()
logger = logging.getLogger(__name__)


async def _mark_pending_seen(session_id, pending_fragment_ids, pending_memory_ids):
    if pending_fragment_ids:
        try:
            await db_conversations.mark_fragments_seen(
                session_id,
                pending_fragment_ids,
                shared.CONVERSATION_SEEN_TTL_HOURS,
            )
        except Exception as e:
            print(f"⚠️ 对话召回 seen 写入失败，保留待下次重试: {e}")

    if pending_memory_ids:
        try:
            await db_conversations.mark_memories_seen(
                session_id,
                pending_memory_ids,
                shared.MEMORY_SEEN_TTL_HOURS,
            )
        except Exception as e:
            print(f"⚠️ 记忆注入 seen 写入失败，保留待下次重试: {e}")

# ============================================================
# API 接口
# ============================================================

@router.get("/")
async def health_check():
    """健康检查"""
    resolved_system_prompt = await shared.get_system_prompt()
    memory_count = 0
    if shared.MEMORY_ENABLED:
        try:
            memory_count = await db_memories.get_all_memories_count()
        except:
            pass

    return {
        "status": "running",
        "gateway": "Pawwake v4.1.1",
        "system_prompt_loaded": len(resolved_system_prompt) > 0,
        "system_prompt_length": len(resolved_system_prompt),
        "database_enabled": shared.DATABASE_ENABLED,
        "conversation_persistence_enabled": shared.conversation_persistence_enabled(),
        "memory_enabled": shared.MEMORY_ENABLED,
        "cache_partition_enabled": shared.CACHE_PARTITION_ENABLED,
        "conversation_recall_enabled": shared.CONVERSATION_RECALL_ENABLED,
        "memory_count": memory_count,
        "memory_extract_interval": shared.MEMORY_EXTRACT_INTERVAL,
    }


@router.get("/v1/models")
async def list_models():
    """模型列表（让客户端不报错）"""
    return {
        "object": "list",
        "data": [
            {
                "id": shared.DEFAULT_MODEL,
                "object": "model",
                "created": 1700000000,
                "owned_by": "pawwake",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """核心转发接口"""
    if not shared.API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "API_KEY 未设置，请在环境变量中配置"},
        )

    try:
        return await _chat_completions_inner(request)
    except Exception:
        logger.exception("Chat completion gateway failed")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Gateway internal error", "type": "gateway_error"}},
        )


async def _chat_completions_inner(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    pending_fragment_ids = []
    pending_memory_ids = []

    # ---------- 检测是否应跳过对话存储 ----------
    # 优先尊重客户端显式声明；无法加 header 的客户端则识别其标题生成模板。
    explicit_skip = request.headers.get("X-Skip-Conversation-Log", "").lower() == "true"
    auxiliary_title_request = partition_engine._is_title_generation_request(messages)
    skip_conversation_log = explicit_skip or auxiliary_title_request
    if auxiliary_title_request:
        print("⏭️  检测到标题生成请求：跳过分区缓存、记忆注入、对话存储和会话 Token 统计")

    # ---------- 提取用户最新消息 ----------
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            break

    # ---------- 构建 system prompt ----------
    # 先保存原始对话消息（不含 system prompt），用于记忆提取
    original_messages = [msg for msg in messages if msg.get("role") != "system"]
    extraction_context_messages = original_messages
    extraction_round_count = None
    resolved_system_prompt = "" if skip_conversation_log else await shared.get_system_prompt()

    # ---------- 检测工具调用消息 ----------
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    if tool_messages:
        print(f"🔧 检测到 {len(tool_messages)} 条工具结果消息")

    # ---------- 生成 session ID ----------
    session_id = str(uuid.uuid4())[:8]

    # ---------- 分区缓存模式 ----------
    if shared.CACHE_PARTITION_ENABLED and not skip_conversation_log:
        active_sid = shared.get_active_session_id()
        if active_sid:
            session_id = active_sid

        # 从DB读取历史
        try:
            db_history = await db_conversations.get_conversation_messages(session_id, limit=10000)
            db_msgs = []
            for m in (db_history or []):
                msg = db_conversations.db_row_to_message(m)
                msg['created_at'] = m.get('created_at')  # 保留时间戳供分区时间窗口判断
                db_msgs.append(msg)
        except Exception as e:
            print(f"❌ 分区缓存不可用：读取对话历史失败: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "分区缓存需要数据库，当前无法读取对话历史。请检查数据库连接或暂时关闭 CACHE_PARTITION_ENABLED。",
                        "type": "partition_database_unavailable",
                    }
                },
            )

        # 提取客户端新消息（非system），可能是user、tool、或带tool_calls的assistant
        client_new_msgs = [m for m in messages if m.get("role") != "system"]
        # 分区模式下，assistant消息来自上一轮response（DB里已存），过滤掉避免重复
        client_new_msgs = [m for m in client_new_msgs if m.get("role") != "assistant"]
        # 分区模式下DB已有完整历史，客户端发来的旧user是冗余的。
        # 但有些客户端把图片和文字拆成多条连续的user发送（图在前文字在后），
        # 只留最后一条会把图那条当冗余丢掉（图片不入库，DB里也找不回来）。
        # 所以按原始消息顺序保留"末尾连续的user块"：历史冗余user总是被assistant隔开，不会混入。
        tail_user_ids = set()
        for m in reversed([m for m in messages if m.get("role") != "system"]):
            if m.get("role") == "user":
                tail_user_ids.add(id(m))
            else:
                break
        user_msgs = [m for m in client_new_msgs if m.get("role") == "user"]
        if len(user_msgs) > len(tail_user_ids):
            client_new_msgs = [
                m for m in client_new_msgs
                if m.get("role") != "user" or id(m) in tail_user_ids
            ]
            print(f"🔧 去重: 过滤{len(user_msgs)-len(tail_user_ids)}条冗余user，保留末尾连续{len(tail_user_ids)}条")
        # 工具结果轮次处理：基于DB状态 + 当前轮次tool_call_id精确判断
        client_tools = [m for m in client_new_msgs if m.get("role") == "tool"]
        if client_tools:
            # 判断DB是否处于"等待tool结果"状态（最后一条是assistant(tool_calls)）
            db_last = db_msgs[-1] if db_msgs else None
            db_expecting_tool = (db_last and db_last.get("role") == "assistant" and db_last.get("tool_calls"))

            if not db_expecting_tool:
                # DB不在等待tool结果 → 客户端的所有tool都是历史残留（含手动删除后的幽灵）
                stale_ids = [m.get('tool_call_id', '?') for m in client_tools]
                print(f"🔧 去重: DB未在等待tool结果，丢弃{len(client_tools)}条客户端tool (ids: {stale_ids})")
                client_new_msgs = [m for m in client_new_msgs if m.get("role") != "tool"]
            else:
                # DB在等待tool → 只保留匹配当前轮次assistant(tool_calls)的tool
                expected_tool_ids = {tc.get("id") for tc in db_last.get("tool_calls", []) if tc.get("id")}
                new_tools = [m for m in client_tools if m.get("tool_call_id") in expected_tool_ids]
                stale_tools = [m for m in client_tools if m.get("tool_call_id") not in expected_tool_ids]

                if stale_tools:
                    print(f"🔧 去重: 丢弃{len(stale_tools)}条非当前轮次tool (ids: {[m.get('tool_call_id','?') for m in stale_tools]})")
                if new_tools:
                    print(f"🔧 保留{len(new_tools)}条当前轮次tool (ids: {[m.get('tool_call_id','?') for m in new_tools]})")

                # 重建 client_new_msgs（user此时已只剩末尾连续块，全部保回，别把拆条发送的图丢了）
                tail_users = [m for m in client_new_msgs if m.get("role") == "user"]
                client_new_msgs = new_tools[:] + tail_users

                if new_tools:
                    # Race condition 防护：DB的assistant(tool_calls)已确认存在（db_expecting_tool=True），
                    # 但仍需检查是否被其他并发请求意外清除
                    new_tool_ids = {m.get("tool_call_id") for m in new_tools if m.get("tool_call_id")}
                    db_has_matching_ast = False
                    for m in db_msgs:
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            ast_tc_ids = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
                            if new_tool_ids & ast_tc_ids:
                                db_has_matching_ast = True
                                break
                    if not db_has_matching_ast and new_tool_ids:
                        for m in messages:
                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                ast_tc_ids = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
                                if new_tool_ids & ast_tc_ids:
                                    client_new_msgs.insert(0, m)
                                    print(f"⚠️ Race防护: 从客户端补充assistant(tool_calls)")
                                    break
        all_msgs = db_msgs + client_new_msgs
        extraction_context_messages = all_msgs
        extraction_round_count = len(partition_engine.group_by_rounds(all_msgs))

        # 同步更新tool_messages，避免process_memories_background存重复的旧tool
        tool_messages = [m for m in client_new_msgs if m.get("role") == "tool"]

        print(f"📦 分区模式: DB历史{len(db_msgs)}条 + 客户端消息{len(client_new_msgs)}条")

        partition_prompt = resolved_system_prompt
        if shared.memory_injection_enabled():
            partition_prompt = (resolved_system_prompt or "") + partition_engine.MEMORY_USAGE_GUIDE
        # 保留客户端自带的 system（工具说明等），拼接到网关 prompt 之后，
        # 与非分区路径的行为对齐（前端 system 稳定时不影响 BP1 缓存命中）
        client_system_text = partition_engine._extract_client_system_text(messages)
        if client_system_text:
            partition_prompt = ((partition_prompt or "") + "\n\n" + client_system_text).strip()
        try:
            conversation_recall_text, pending_fragment_ids = (
                await memory_pipeline.build_conversation_recall_text(user_message, session_id)
            )

            async def build_session_memory_text(message: str):
                return await memory_pipeline.build_memory_text(
                    message,
                    session_id,
                    pending_memory_ids,
                )

            messages = await partition_engine.build_partitioned_messages(
                session_id,
                all_msgs,
                partition_prompt,
                user_message,
                conversation_recall_text,
                build_session_memory_text,
            )
        except Exception as e:
            print(f"❌ 分区缓存不可用：读取轮转状态失败: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "分区缓存需要数据库，当前无法读取轮转状态。请检查数据库连接或暂时关闭 CACHE_PARTITION_ENABLED。",
                        "type": "partition_database_unavailable",
                    }
                },
            )
        body["messages"] = messages

    else:
        # ---------- 原有逻辑：system prompt + 记忆注入 ----------
        if not skip_conversation_log and (resolved_system_prompt or (shared.MEMORY_ENABLED and user_message)):
            if shared.MEMORY_ENABLED and user_message:
                enhanced_prompt = await memory_pipeline.build_system_prompt_with_memories(
                    user_message,
                    resolved_system_prompt,
                )
            else:
                enhanced_prompt = resolved_system_prompt

            if enhanced_prompt:
                has_system = any(msg.get("role") == "system" for msg in messages)
                if has_system:
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system":
                            messages[i]["content"] = enhanced_prompt + "\n\n" + msg["content"]
                            break
                else:
                    messages.insert(0, {"role": "system", "content": enhanced_prompt})

        body["messages"] = messages

    # ---------- 模型处理 ----------
    model = body.get("model", shared.DEFAULT_MODEL)
    if not model:
        model = shared.DEFAULT_MODEL
    body["model"] = model

    # ---------- cache_control 兼容性处理 ----------
    if shared.CACHE_PARTITION_ENABLED and not partition_engine._is_anthropic_model(model):
        partition_engine._strip_cache_control(body.get("messages", []))

    # ---------- 转发请求 ----------
    headers = {
        "Authorization": f"Bearer {shared.API_KEY}",
        "Content-Type": "application/json",
    }
    # OpenRouter 需要的额外头
    if "openrouter" in shared.API_BASE_URL:
        headers["HTTP-Referer"] = shared.EXTRA_REFERER
        headers["X-Title"] = shared.EXTRA_TITLE

    is_stream = body.get("stream", False)

    # 强制流式传输（解决部分客户端不发stream=true的问题）
    if shared.FORCE_STREAM and not is_stream:
        is_stream = True
        body["stream"] = True
        print(f"⚡ 强制开启流式传输（FORCE_STREAM=true）")

    # 注入推理参数（解决客户端走网关时不带reasoning参数的问题）
    if shared.REASONING_EFFORT and not skip_conversation_log:
        # 统一用 reasoning_effort（Claude/OpenAI/Google Gemini OpenAI兼容端点都支持）
        # 先删除客户端可能已带的值，确保用我们配置的
        body.pop("reasoning_effort", None)
        body.pop("google", None)
        body["reasoning_effort"] = shared.REASONING_EFFORT
        print(f"🧠 注入推理参数: reasoning_effort={shared.REASONING_EFFORT}")

    print(f"📡 请求: model={model}, stream={is_stream}, memory={'on' if shared.MEMORY_ENABLED else 'off'}", flush=True)

    # 调试：打印请求体中的推理相关字段
    debug_keys = {k: v for k, v in body.items() if k in ('reasoning_effort', 'google', 'reasoning')}
    if debug_keys:
        print(f"📡 推理字段: {debug_keys}", flush=True)

    if is_stream:
        return StreamingResponse(
            stream_and_capture(
                headers,
                body,
                session_id,
                user_message,
                model,
                extraction_context_messages,
                skip_conversation_log,
                tool_messages,
                pending_fragment_ids,
                extraction_round_count,
                pending_memory_ids,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(shared.API_BASE_URL, headers=headers, json=body)

            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg = ""
                assistant_tool_calls = None
                assistant_reasoning = None
                try:
                    msg_obj = resp_data["choices"][0]["message"]
                    assistant_msg = msg_obj.get("content") or ""
                    if msg_obj.get("tool_calls"):
                        assistant_tool_calls = msg_obj["tool_calls"]
                        print(f"🔧 Response 包含 {len(assistant_tool_calls)} 个工具调用")
                    if msg_obj.get("reasoning_content"):
                        assistant_reasoning = msg_obj["reasoning_content"]
                        print(f"🧠 Response 包含 reasoning_content ({len(assistant_reasoning)}字符)")
                except (KeyError, IndexError):
                    pass

                await _mark_pending_seen(
                    session_id,
                    pending_fragment_ids,
                    pending_memory_ids,
                )

                if shared.conversation_persistence_enabled() and (user_message or tool_messages):
                    asyncio.create_task(
                        memory_pipeline.process_memories_background(session_id, user_message, assistant_msg, model,
                                                    context_messages=extraction_context_messages,
                                                    context_round_count=extraction_round_count,
                                                    skip_conversation_log=skip_conversation_log,
                                                    tool_messages=tool_messages, assistant_tool_calls=assistant_tool_calls,
                                                    assistant_reasoning=assistant_reasoning)
                    )

                return JSONResponse(status_code=200, content=resp_data)
            else:
                try:
                    error_content = response.json()
                except Exception:
                    error_content = {"error": {"message": response.text[:500], "type": "upstream_error"}}
                return JSONResponse(status_code=response.status_code, content=error_content)


async def stream_and_capture(
    headers: dict,
    body: dict,
    session_id: str,
    user_message: str,
    model: str,
    extraction_context_messages: list = None,
    skip_conversation_log: bool = False,
    tool_messages: list = None,
    pending_fragment_ids: list = None,
    extraction_round_count: int = None,
    pending_memory_ids: list = None,
):
    """流式响应 + 捕获完整回复（原始字节透传，确保SSE格式和thinking数据完整）"""
    full_response = []
    full_reasoning = []
    stream_usage = {}
    line_buffer = ""
    accumulated_tool_calls = {}  # index -> OpenAI-compatible tool call
    stream_succeeded = False

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", shared.API_BASE_URL, headers=headers, json=body) as response:
            # 打印上游响应头（排查thinking问题用）
            upstream_ct = response.headers.get("content-type", "")
            print(f"📨 上游响应: status={response.status_code}, content-type={upstream_ct}", flush=True)

            # 上游非200时，提前打印messages结构方便debug
            if response.status_code != 200:
                msg_summary = [{"role": m.get("role"), "tool_calls": bool(m.get("tool_calls")), "tool_call_id": m.get("tool_call_id", ""), "content_type": type(m.get("content")).__name__} for m in body.get("messages", [])]
                print(f"❌ 发送的messages结构({len(msg_summary)}条): {msg_summary}", flush=True)

            error_body_parts = []
            is_error = response.status_code != 200

            async for chunk in response.aiter_bytes():
                # 原始字节直接透传给客户端
                yield chunk

                if is_error:
                    error_body_parts.append(chunk)
                    continue

                # 旁路解析：从字节流中提取assistant回复内容，用于后续记忆提取
                text = chunk.decode("utf-8", errors="ignore")
                line_buffer += text
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])

                            if "usage" in data:
                                stream_usage = data["usage"]

                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response.append(content)

                            # 收集reasoning_content（deepseek thinking mode）
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                full_reasoning.append(reasoning)

                            # 累积tool_calls
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {
                                            "index": idx,
                                            "id": tc.get("id", ""),
                                            "type": tc.get("type", "function"),
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    if tc.get("id"):
                                        accumulated_tool_calls[idx]["id"] = tc["id"]
                                    for key, value in tc.items():
                                        if key not in {"index", "id", "type", "function"}:
                                            accumulated_tool_calls[idx][key] = value
                                    if "function" in tc:
                                        fn = tc["function"]
                                        if fn.get("name"):
                                            accumulated_tool_calls[idx]["function"]["name"] = fn["name"]
                                        if "arguments" in fn:
                                            accumulated_tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                                        for key, value in fn.items():
                                            if key not in {"name", "arguments"}:
                                                accumulated_tool_calls[idx]["function"][key] = value
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
            stream_succeeded = response.status_code == 200

    assistant_msg = "".join(full_response)
    assistant_reasoning = "".join(full_reasoning) if full_reasoning else None
    assistant_tool_calls = list(accumulated_tool_calls.values()) if accumulated_tool_calls else None

    if assistant_reasoning:
        print(f"🧠 Stream response 包含 reasoning_content ({len(assistant_reasoning)}字符)")

    # 打印上游错误内容
    if error_body_parts:
        error_text = b"".join(error_body_parts).decode("utf-8", errors="ignore")[:500]
        print(f"❌ 上游错误内容: {error_text}", flush=True)

    if assistant_tool_calls:
        print(f"🔧 Stream response 包含 {len(assistant_tool_calls)} 个工具调用")

    if stream_succeeded:
        await _mark_pending_seen(
            session_id,
            pending_fragment_ids,
            pending_memory_ids,
        )

    if stream_usage:
        pt = stream_usage.get("prompt_tokens", 0)
        ct = stream_usage.get("completion_tokens", 0)
        tt = stream_usage.get("total_tokens", 0)
        if shared.DATABASE_ENABLED and tt > 0 and not skip_conversation_log:
            asyncio.create_task(db_conversations.save_token_usage(session_id, model, pt, ct, tt))
            print(f"📊 Stream Token: {pt} + {ct} = {tt}")

    if shared.conversation_persistence_enabled() and (user_message or tool_messages):
        asyncio.create_task(
            memory_pipeline.process_memories_background(session_id, user_message, assistant_msg, model,
                                        context_messages=extraction_context_messages,
                                        context_round_count=extraction_round_count,
                                        skip_conversation_log=skip_conversation_log,
                                        tool_messages=tool_messages, assistant_tool_calls=assistant_tool_calls,
                                        assistant_reasoning=assistant_reasoning)
        )
