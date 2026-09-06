"""
记忆提取模块 —— 用 LLM 从对话中提炼关键记忆
=============================================
每次对话结束后，把最近的对话内容发给一个便宜的模型，
让它提取出值得记住的信息，存到数据库里。

v2.3 改进：提取时注入已有记忆，让模型对比后只提取全新信息。
"""

import os
import json
import httpx
from typing import List, Dict

import shared

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 记忆模型专用 API Key（不设则回退到主 API_KEY）
# 适用于中转站按模型分组、不同模型需要不同 Key 的场景
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", "")

# 用来提取记忆的模型（便宜的就行）
# 面板上这项写着"留空用默认"，清空会写进一个空串，os.getenv 的默认值这时不生效，
# 所以默认值单独拎出来用 or 兜，热更新和重启后行为才一致
DEFAULT_MEMORY_MODEL = "anthropic/claude-haiku-4.5"
MEMORY_MODEL = os.getenv("MEMORY_MODEL") or DEFAULT_MEMORY_MODEL

# 记忆提取的输出上限，原先硬编码 1000。部分上游会把 reasoning token
# 也算进这条额度，JSON 可能在收尾前被截断，表面只报"未找到JSON数组"
MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "4000"))

def get_memory_api_key() -> str:
    return MEMORY_API_KEY or API_KEY


def _diagnose_incomplete(finish_reason, completion_tokens, reasoning_tokens) -> str:
    """JSON 收不了尾时，判断是截断还是格式不符。证据不足就返回"无法判定"，不硬猜"""
    if finish_reason == "length":
        return (
            f"输出被上限切断（上游明确报 finish_reason=length，当前 MEMORY_MAX_TOKENS={MEMORY_MAX_TOKENS}）。"
            "调高该值；模型若带推理模式，推理 token 也占这条额度"
        )

    if finish_reason == "stop":
        # stop 说明上游认为输出完整，就算推理 token 顶满 usage 也不是截断
        return "上游报正常结束，是模型没按 JSON 格式输出。检查提示词，或换一个更听话的模型"

    if isinstance(completion_tokens, int) and completion_tokens >= MEMORY_MAX_TOKENS:
        extra = f"，其中推理 {reasoning_tokens}" if reasoning_tokens is not None else ""
        return (
            f"输出很可能被切断（completion_tokens={completion_tokens}{extra}，已顶到上限 {MEMORY_MAX_TOKENS}）。"
            "先调高 MEMORY_MAX_TOKENS 再看。注意各家 usage 口径不一，这条是强证据但不是铁证"
        )

    return (
        f"原因无法判定：上游没给 finish_reason（={finish_reason}），usage 也证明不了是否触顶。"
        "先确认中转站是否返回这两个字段，再谈是截断还是格式问题"
    )


EXTRACTION_PROMPT = """你是信息提取专家，负责从对话中识别并提取值得长期记住的关键信息。

# 提取重点
- 关键信息：提取用户的重要信息和值得回忆的生活细节
- 重要事件：记忆深刻的互动，需包含人物、时间、地点（如有）

# 提取范围
- 个人：年龄、生日、职业、学历、居住地
- 偏好：明确表达的喜好或厌恶
- 健康：身体状况、过敏史、饮食禁忌
- 事件：与AI的重要互动、约定、里程碑
- 关系：家人、朋友、重要同事
- 价值观：表达的信念或长期目标
- 情感：重要的情感时刻或关系里程碑
- 生活：用户当天的活动、饮食、出行、日常经历等生活细节
- AI：AI做出的承诺、约定、重要情感表达

# 提取要求
- 事件类记忆保留双方的关键原话，用引号标注是谁说的
- 项目/技术进展只记要点（改了什么、解决了什么），不记调试过程

# 不要提取
- 日常寒暄（"你好""在吗"）
- AI的纯知识性回答（百科、翻译、代码讲解等，不涉及双方关系和承诺的内容）
- 关于记忆系统本身的讨论（"某条记忆没有被记录""记忆遗漏""没有被提取"等）
- AI的思考过程、思维链内容

# 已知信息判定【最重要】
<已知信息>
{existing_memories}
</已知信息>

- 对每条值得长期记住的信息，必须与已知信息逐条比对并给出 action
- new：已知信息中没有相同事实；补充不同属性也属于 new
- duplicate：与已知信息相同、相似或语义重复
- supersede：同一主体、同一属性发生真实替换，例如“搬到上海”取代旧住址
- 临时状态与长期事实可以并存，例如“去上海出差”不能取代旧住址
- supersede 必须填写候选列表里真实存在的 candidate_id；禁止编造 ID
- 如果对话中没有任何值得长期记住的信息，返回空数组 []

# 输出格式
请用以下 JSON 格式返回（不要包含其他内容）：
[
  {{"content": "记忆内容", "importance": 分数, "action": "new", "candidate_id": null}},
  {{"content": "记忆内容", "importance": 分数, "action": "duplicate", "candidate_id": 旧记忆ID}},
  {{"content": "记忆内容", "importance": 分数, "action": "supersede", "candidate_id": 被取代的旧记忆ID}}
]

importance 分数 1-10，10 最重要。
如果没有值得记住的新信息，返回空数组：[]
"""


async def extract_memories(messages: List[Dict[str, str]], existing_memories: List = None) -> List[Dict]:
    """
    从对话消息中提取记忆

    参数：
        messages: 对话消息列表，格式 [{"role": "user", "content": "..."}, ...]
        existing_memories: 已有候选记忆，格式为 ID + 原文；兼容旧的纯文本列表

    返回：
        记忆列表，格式 [{"content": "...", "importance": N, "action": "..."}, ...]
    """
    if not get_memory_api_key():
        print("⚠️  API_KEY 和 MEMORY_API_KEY 都未设置，跳过记忆提取")
        return []

    if not messages:
        return []

    # 把对话格式化成文本
    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"用户: {content}\n"
        elif role == "assistant":
            conversation_text += f"AI: {content}\n"

    if not conversation_text.strip():
        return []

    # 格式化已有记忆
    if existing_memories:
        memory_lines = []
        for memory in existing_memories:
            if isinstance(memory, dict):
                memory_lines.append(
                    f"- [candidate_id={memory.get('id')}] {memory.get('content', '')}"
                )
            else:
                memory_lines.append(f"- {memory}")
        memories_text = "\n".join(memory_lines)
    else:
        memories_text = "（暂无已知信息）"

    # 把已有记忆填入prompt
    prompt = EXTRACTION_PROMPT.format(existing_memories=memories_text)

    # 调用 LLM 提取记忆
    try:
        headers = {
            "Authorization": f"Bearer {get_memory_api_key()}",
            "Content-Type": "application/json",
        }
        if "openrouter" in API_BASE_URL:
            headers["HTTP-Referer"] = shared.EXTRA_REFERER
            headers["X-Title"] = shared.EXTRA_TITLE

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                API_BASE_URL,
                headers=headers,
                json={
                    "model": MEMORY_MODEL,
                    "max_tokens": MEMORY_MAX_TOKENS,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"请从以下对话中提取新的记忆：\n\n{conversation_text}"},
                    ],
                },
            )

            if response.status_code != 200:
                print(f"⚠️  记忆提取请求失败: {response.status_code}, model={MEMORY_MODEL}: {response.text[:500]}")
                return []

            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            finish_reason = choice.get("finish_reason")

            # usage 各家口径不同（推理 token 有的单列有的算进 completion），只当佐证
            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

            # 正文截断防刷屏，但长度和停止原因要给全，否则分不清是日志截断还是真截断
            usage_part = f"，completion_tokens={completion_tokens}/{MEMORY_MAX_TOKENS}" if completion_tokens is not None else "，usage 未提供"
            if reasoning_tokens is not None:
                usage_part += f"（其中推理 {reasoning_tokens}）"
            print(
                f"📝 记忆模型原始返回（{len(text)} 字符，finish_reason={finish_reason}{usage_part}）:\n{text[:500]}",
                flush=True,
            )

            # 清理可能的 markdown 格式（原始长度留给报错用，免得日志里两个数对不上）
            raw_len = len(text)
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # 强力JSON提取：如果上面清理后仍然解析失败，用正则兜底
            try:
                memories = json.loads(text)
            except json.JSONDecodeError:
                # 尝试从文本中提取第一个 [...] 结构
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        memories = json.loads(match.group())
                        print(f"📝 JSON正则兜底提取成功")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  记忆提取结果解析失败: {e}")
                        return []
                else:
                    # 不要补上收尾的 ]：断掉的可能是半个字符串，
                    # 补完只会把残缺内容伪装成一条有效记忆存进库
                    print(
                        f"⚠️  记忆提取结果中未找到完整 JSON 数组（共 {raw_len} 字符），本轮跳过\n"
                        f"    {_diagnose_incomplete(finish_reason, completion_tokens, reasoning_tokens)}"
                    )
                    return []

            if not isinstance(memories, list):
                return []

            # 模型可能先吐完一个完整数组再被切断，解析成功也未必没丢东西。
            # 只认上游明确报 length；finish_reason 缺失时才退回 token 计数兜底，
            # 报 stop 的完整回复不警告（推理 token 会把 completion_tokens 顶过上限）
            if finish_reason == "length" or (
                finish_reason is None
                and isinstance(completion_tokens, int)
                and completion_tokens >= MEMORY_MAX_TOKENS
            ):
                print(
                    f"⚠️  本次解析成功，但上游显示输出已顶到上限 {MEMORY_MAX_TOKENS}，"
                    "后面可能还有没写完的记忆。建议调高 MEMORY_MAX_TOKENS"
                )

            # 验证格式
            valid_memories = []
            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    action = str(mem.get("action", "new")).strip().lower()
                    if action not in {"new", "duplicate", "supersede"}:
                        action = "new"
                    candidate_id = mem.get("candidate_id")
                    if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
                        candidate_id = None
                    valid_memories.append({
                        "content": str(mem["content"]),
                        "importance": int(mem.get("importance", 5)),
                        "action": action,
                        "candidate_id": candidate_id,
                    })

            print(f"📝 从对话中提取了 {len(valid_memories)} 条新记忆（已对比 {len(existing_memories or [])} 条已有记忆）")
            return valid_memories

    except json.JSONDecodeError as e:
        print(f"⚠️  记忆提取结果解析失败: {e}")
        return []
    except Exception as e:
        print(f"⚠️  记忆提取出错: {e}")
        return []


SCORING_PROMPT = """你是记忆重要性评分专家。请对以下记忆条目逐条评分。

# 评分规则（1-10）
- 9-10：核心身份信息（名字、生日、职业、重要关系）
- 7-8：重要偏好、重大事件、深层情感
- 5-6：日常习惯、一般偏好
- 3-4：临时状态、偶然提及
- 1-2：琐碎信息

# 输入记忆
{memories_text}

# 输出格式
返回 JSON 数组，每条包含原文和评分：
[{{"content": "原文", "importance": 评分数字}}]

只返回 JSON，不要其他文字。"""


def _default_scores(texts: List[str]) -> List[Dict]:
    return [{"content": text, "importance": 5} for text in texts]


async def score_memories(texts: List[str]) -> List[Dict]:
    """对纯文本记忆条目批量评分"""
    if not texts:
        return []

    memories_text = "\n".join(f"- {t}" for t in texts)
    prompt = SCORING_PROMPT.format(memories_text=memories_text)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                API_BASE_URL,
                headers={
                    "Authorization": f"Bearer {get_memory_api_key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MEMORY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    # 跟提取同一个模型同一类活，跟着同一个配置走；写死会让用户调了也不生效
                    "max_tokens": MEMORY_MAX_TOKENS,
                },
            )

            if response.status_code != 200:
                print(f"⚠️  记忆评分请求失败: {response.status_code}, model={MEMORY_MODEL}: {response.text[:500]}")
                # 失败时返回默认分数
                return _default_scores(texts)

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                memories = json.loads(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        memories = json.loads(match.group())
                    except json.JSONDecodeError:
                        return _default_scores(texts)
                else:
                    return _default_scores(texts)

            if not isinstance(memories, list):
                return _default_scores(texts)

            valid = []
            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    valid.append({
                        "content": str(mem["content"]),
                        "importance": int(mem.get("importance", 5)),
                    })

            print(f"📝 为 {len(valid)} 条记忆完成自动评分")
            return valid

    except Exception as e:
        print(f"⚠️  记忆评分出错: {e}")
        return _default_scores(texts)
