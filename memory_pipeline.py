"""Memory injection, conversation recall, and background extraction."""

import json
from datetime import datetime, timedelta, timezone

import shared
import partition_engine
from db import core as db_core
from db import search as db_search
from db import conversations as db_conversations
from db import memories as db_memories
from memory_extractor import extract_memories

# ============================================================
# 记忆注入
# ============================================================

def _memory_date_prefix(mem: dict) -> str:
    if mem.get("event_date"):
        return f"[{str(mem['event_date'])[:10]}] "
    if mem.get("created_at"):
        try:
            utc_str = str(mem["created_at"])[:19]
            utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            local_dt = utc_dt + timedelta(hours=shared.TIMEZONE_HOURS)
            return f"[{local_dt.strftime('%Y-%m-%d')}] "
        except:
            return f"[{str(mem['created_at'])[:10]}] "
    return ""


async def build_system_prompt_with_memories(user_message: str, base_prompt: str) -> str:
    """
    构建带记忆的 system prompt
    1. 用用户消息搜索相关记忆
    2. 格式化成文本拼接到人设后面
    """
    if not shared.memory_injection_enabled():
        return base_prompt

    try:
        memories = await db_memories.search_memories(user_message, limit=shared.MAX_MEMORIES_INJECT)

        if not memories:
            return base_prompt

        # 格式化记忆文本（带日期，帮助模型判断新旧）
        # event_date 优先：整理产物的 created_at 是整理当天，真实发生日存在 event_date 里。
        # event_date 本身就是本地日期，不做时区换算
        memory_lines = []
        for mem in memories:
            memory_lines.append(f"- {_memory_date_prefix(mem)}{mem['content']}")
        memory_text = "\n".join(memory_lines)

        enhanced_prompt = f"""{base_prompt}

【从过往对话中检索到的相关记忆】
{memory_text}

# 记忆应用
- 像朋友般自然运用这些记忆，不刻意展示
- 仅在相关话题出现时引用，避免主动提及
- 对重要信息（如健康、日期、约定）保持一致性
- 新信息与记忆冲突时，以新信息为准
- 模糊记忆可表达不确定性："记得你似乎说过..."

# 交流方式
- 自然引用："记得你说过..."或"上次我们聊到..."
- 避免机械式表达如"根据我的记忆..."或"检索到的信息显示..."
- 共同经历可温情回忆："上次那个事挺好玩的"

记忆是丰富对话的工具，而非对话焦点。"""

        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return enhanced_prompt

    except Exception as e:
        print(f"⚠️  记忆检索失败: {e}，使用纯人设")
        return base_prompt


async def build_memory_text(
    user_message: str,
    session_id: str = None,
    injected_ids: list = None,
) -> str:
    """搜索记忆并格式化为注入文本（分区缓存模式用）。"""
    if not shared.memory_injection_enabled():
        return ""
    try:
        excluded_ids = []
        if session_id and shared.MEMORY_SEEN_TTL_HOURS > 0:
            excluded_ids = await db_conversations.get_active_seen_memory_ids(
                session_id,
                shared.MEMORY_SEEN_TTL_HOURS,
            )
        memories = await db_memories.search_memories(
            user_message,
            limit=shared.MAX_MEMORIES_INJECT,
            exclude_ids=excluded_ids,
        )
        if not memories:
            return ""

        memory_lines = []
        for mem in memories:
            memory_lines.append(f"- {_memory_date_prefix(mem)}{mem['content']}")

        if injected_ids is not None:
            injected_ids.extend(
                mem["id"]
                for mem in memories
                if isinstance(mem.get("id"), int) and not isinstance(mem.get("id"), bool)
            )

        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return (
            "<retrieved_memories>\n"
            "以下是网关从过往对话中自动检索的相关记忆，供参考，非用户本次输入：\n"
            + "\n".join(memory_lines)
            + "\n</retrieved_memories>"
        )
    except Exception as e:
        print(f"⚠️ 记忆检索失败: {e}")
        return ""


def _format_recall_timestamp(raw_ts) -> str:
    """召回片段时间前缀：按 TIMEZONE_HOURS（环境变量，启动时生效）转本地时间到分钟。

    naive 时间戳按 UTC 处理；任何解析失败退回截取日期的旧行为。
    """
    try:
        if isinstance(raw_ts, datetime):
            ts = raw_ts
        else:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        local = ts.astimezone(timezone(timedelta(hours=shared.TIMEZONE_HOURS)))
        return local.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(raw_ts)[:10]


async def build_conversation_recall_text(user_message: str, session_id: str):
    """为分区模式检索未注入过的对话片段，并返回待确认的 fragment_id。"""
    if (
        not shared.CONVERSATION_RECALL_ENABLED
        or shared.MAX_CONVERSATIONS_INJECT <= 0
        or not user_message.strip()
    ):
        return "", []
    try:
        state = await db_conversations.get_session_cache_state(
            session_id,
            shared.CONVERSATION_SEEN_TTL_HOURS,
        )
        results, _ = await db_search.search_chat_fragments(
            user_message,
            max_sessions=shared.MAX_CONVERSATIONS_INJECT,
            max_matches_per_session=1,
            context=1,
            mode="hybrid",
            exclude_session_ids=[session_id],
            exclude_fragment_ids=state.get("seen_fragment_ids", []),
        )
        if not results:
            return "", []

        blocks = []
        fragment_ids = []
        role_labels = {"user": "用户", "assistant": "AI", "tool": "工具"}
        for result in results:
            for fragment, fragment_id in zip(
                result.get("fragments", []),
                result.get("fragment_ids", []),
            ):
                lines = []
                for message in fragment:
                    date_prefix = ""
                    if message.get("created_at"):
                        date_prefix = f"[{_format_recall_timestamp(message['created_at'])}] "
                    role = role_labels.get(message.get("role"), message.get("role", "消息"))
                    lines.append(f"{date_prefix}{role}: {message.get('content', '')}")
                if lines and fragment_id:
                    blocks.append("\n".join(lines))
                    fragment_ids.append(fragment_id)

        if not blocks:
            return "", []
        return (
            "<retrieved_conversations>\n"
            "以下是网关检索的相关历史对话片段，供参考，非用户本次输入：\n"
            + "\n\n".join(blocks)
            + "\n</retrieved_conversations>",
            fragment_ids,
        )
    except Exception as e:
        print(f"⚠️ 对话召回失败: {e}")
        return "", []


# ============================================================
# 后台记忆处理
# ============================================================

def _extraction_query_text(messages: list) -> str:
    """Flatten the exact user/assistant extraction window into search text."""
    parts = []
    for message in messages:
        if message.get("role") not in {"user", "assistant"}:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
            )
        else:
            text = ""
        if text.strip():
            parts.append(text.strip())
    return "\n".join(parts)

async def process_memories_background(
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    model: str,
    context_messages: list = None,
    context_round_count: int = None,
    skip_conversation_log: bool = False,
    tool_messages: list = None,
    assistant_tool_calls: list = None,
    assistant_reasoning: str = None,
):
    """
    后台异步：存储对话 + 提取记忆（不阻塞主流程）

    记忆提取受 MEMORY_EXTRACT_INTERVAL 控制：
    - 0: 禁用自动提取
    - 1: 每轮提取（默认）
    - N: 每 N 轮提取一次
    对话记录始终保存，不受间隔影响（除非 skip_conversation_log=True）。

    context_messages: 分区模式使用 DB 历史与本轮消息组成的权威上下文；
                      非分区模式使用客户端发来的非 system 消息。
    context_round_count: 分区模式当前 session 的逻辑轮数；非分区模式为 None。
    skip_conversation_log: 跳过对话存储（标题生成等辅助请求时使用）
    tool_messages: 客户端发来的工具结果消息列表
    assistant_tool_calls: response中assistant的工具调用列表（如果有）
    assistant_reasoning: response中assistant的reasoning_content（deepseek thinking mode）
    """

    try:
        # Debug: 打印存储分支判断依据
        print(f"💾 process_memories_background: user_msg={bool(user_msg)}, tool_messages={len(tool_messages) if tool_messages else 0}, "
              f"assistant_tool_calls={len(assistant_tool_calls) if assistant_tool_calls else 0}, skip={skip_conversation_log}")
        if tool_messages:
            print(f"💾 tool详情: {[{'role': m.get('role'), 'tool_call_id': m.get('tool_call_id', '?')} for m in tool_messages]}")

        assistant_meta = None
        if not skip_conversation_log:
            ast_meta_dict = {}
            if assistant_tool_calls:
                ast_meta_dict["tool_calls"] = assistant_tool_calls
            if assistant_reasoning:
                ast_meta_dict["reasoning_content"] = assistant_reasoning
            assistant_meta = json.dumps(ast_meta_dict) if ast_meta_dict else None

        # 1. 存储对话记录（除非明确跳过）
        if skip_conversation_log:
            print(f"⏭️  跳过对话存储（辅助请求）")
        elif tool_messages:
            # 工具结果轮次：存tool消息 + assistant回复（user消息在之前的轮次已存过）
            for tm in tool_messages:
                meta_dict = {}
                if tm.get("tool_call_id"):
                    meta_dict["tool_call_id"] = tm["tool_call_id"]
                if tm.get("name"):
                    meta_dict["name"] = tm["name"]
                meta = json.dumps(meta_dict) if meta_dict else None
                await db_conversations.save_message(session_id, "tool", tm.get("content", ""), model, metadata=meta)

            if assistant_msg or assistant_tool_calls:
                await db_conversations.save_message(session_id, "assistant", assistant_msg or "", model, metadata=assistant_meta)
                print(f"🔧 存储: {len(tool_messages)}条tool + 1条assistant" + (" (含tool_calls)" if assistant_tool_calls else "") + (" (含reasoning)" if assistant_reasoning else ""))
        else:
            # 普通对话或首次工具调用
            if assistant_tool_calls:
                # 首次工具调用：assistant回复包含tool_calls，存user + assistant(tool_calls)
                await db_conversations.save_message(session_id, "user", user_msg, model)
                await db_conversations.save_message(session_id, "assistant", assistant_msg or "", model, metadata=assistant_meta)
                print(f"🔧 存储: user + assistant (含{len(assistant_tool_calls)}个tool_calls)" + (" (含reasoning)" if assistant_reasoning else ""))
            else:
                # 纯文字对话：re-roll检测 + 存user + assistant
                last_user = await db_conversations.get_last_user_content(session_id)
                if last_user and last_user.strip() == user_msg.strip():
                    updated = await db_conversations.update_last_assistant_message(session_id, assistant_msg, model)
                    if updated:
                        print(f"🔄 检测到re-roll，已覆盖最后一条assistant回复")
                    else:
                        await db_conversations.save_message(session_id, "user", user_msg, model)
                        await db_conversations.save_message(session_id, "assistant", assistant_msg, model, metadata=assistant_meta)
                else:
                    await db_conversations.save_message(session_id, "user", user_msg, model)
                    await db_conversations.save_message(session_id, "assistant", assistant_msg, model, metadata=assistant_meta)

        # 2. 检查是否需要提取记忆
        if not shared.MEMORY_ENABLED:
            print(f"⏭️  记忆系统已关闭；仅保留分区缓存或对话召回所需的对话记录")
            return

        if shared.MEMORY_EXTRACT_INTERVAL == 0:
            print(f"⏭️  记忆自动提取已禁用，跳过")
            return

        # 工具调用尚未结束时只落库，等最终 assistant 回复再做一次完整提取。
        if assistant_tool_calls:
            print("⏭️  当前逻辑轮仍在等待工具结果，推迟记忆提取")
            return

        if context_round_count is None:
            shared._nonpartition_round_counter += 1
            current_round_count = shared._nonpartition_round_counter
            round_scope = "非分区累计"
        else:
            current_round_count = max(0, int(context_round_count))
            round_scope = f"session={session_id}"

        if current_round_count <= 0:
            print("⏭️  当前上下文没有可提取的逻辑轮")
            return

        if (
            shared.MEMORY_EXTRACT_INTERVAL > 1
            and current_round_count % shared.MEMORY_EXTRACT_INTERVAL != 0
        ):
            print(
                f"⏭️  {round_scope} 第 {current_round_count} 轮，跳过记忆提取"
                f"（每 {shared.MEMORY_EXTRACT_INTERVAL} 轮提取一次）"
            )
            return

        if shared.MEMORY_EXTRACT_INTERVAL > 1:
            print(f"📝 {round_scope} 第 {current_round_count} 轮，执行记忆提取")

        # 3. 按逻辑轮截取近期上下文，而非发送完整会话。
        if context_messages:
            messages_for_extraction, selected_round_count = (
                partition_engine._build_memory_extraction_messages(
                    context_messages,
                    assistant_msg,
                    shared.MEMORY_EXTRACT_INTERVAL,
                )
            )
            print(
                f"📝 截取最近 {selected_round_count} 个逻辑轮提取记忆"
                f"（{len(messages_for_extraction)} 条 user/assistant 消息）"
            )
        else:
            messages_for_extraction = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]

        # 4. 用同一提取窗口只读检索相关候选与最新活跃记忆。
        candidate_query = _extraction_query_text(messages_for_extraction)
        existing = await db_memories.get_extraction_candidates(candidate_query)
        candidate_ids = {row["id"] for row in existing}

        new_memories = await extract_memories(
            messages_for_extraction,
            existing_memories=existing,
        )

        # 过滤垃圾记忆（不靠模型自觉，硬过滤）
        META_BLACKLIST = [
            "记忆库", "记忆系统", "检索", "没有被记录", "没有被提取",
            "记忆遗漏", "尚未被记录", "写入不完整", "检索功能",
            "系统没有返回", "关键词匹配", "语义匹配", "语义检索",
            "阈值", "数据库", "seed", "导入", "部署",
            "bug", "debug", "端口", "网关",
        ]

        filtered_memories = []
        for mem in new_memories:
            if mem.get("action") == "duplicate":
                print(f"⏭️ 跳过重复记忆: {mem['content'][:60]}...")
                continue
            content = mem["content"]
            if any(kw in content for kw in META_BLACKLIST):
                print(f"🚫 过滤掉meta记忆: {content[:60]}...")
                continue
            if (
                mem.get("action") == "supersede"
                and mem.get("candidate_id") not in candidate_ids
            ):
                mem = {**mem, "action": "new", "candidate_id": None}
            filtered_memories.append(mem)

        superseded_count = 0
        for mem in filtered_memories:
            result = await db_memories.save_extracted_memory(
                content=mem["content"],
                importance=mem["importance"],
                source_session=session_id,
                supersede_id=(
                    mem.get("candidate_id")
                    if mem.get("action") == "supersede"
                    else None
                ),
                candidate_ids=candidate_ids,
            )
            if result["action"] == "supersede":
                superseded_count += 1

        if filtered_memories:
            total = await db_memories.get_all_memories_count()
            print(
                f"💾 已保存 {len(filtered_memories)} 条记忆"
                f"（取代 {superseded_count} 条，跳过/过滤 {len(new_memories) - len(filtered_memories)} 条），"
                f"总计 {total} 条"
            )

    except Exception as e:
        print(f"⚠️  后台记忆处理失败: {e}")
