"""Dashboard pages, model listing, and runtime settings."""

import os
import secrets
from urllib.parse import parse_qs

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import auth
import shared
from db import core as db_core
from db import search as db_search

router = APIRouter()
settings_router = APIRouter()
_MASKED_KEYS = {"API_KEY", "EMBEDDING_API_KEY", "MEMORY_API_KEY"}

@router.get("/dashboard/login", response_class=HTMLResponse)
async def dashboard_login_page(request: Request):
    """Dashboard 独立登录页，网关主密钥不进入浏览器。"""
    if auth.valid_dashboard_session(request.cookies.get(shared.DASHBOARD_SESSION_COOKIE, "")):
        return RedirectResponse(url="/dashboard", status_code=303)
    configured = auth.dashboard_auth_ready()
    return shared.templates.TemplateResponse(
        request,
        "login.html",
        {
            "error": None,
            "config_error": None if configured else "请先设置 DASHBOARD_PASSWORD 和至少 32 字符的 SESSION_SECRET。",
        },
        status_code=200 if configured else 503,
        headers={"Cache-Control": "no-store"},
    )


@router.post("/dashboard/login", response_class=HTMLResponse)
async def dashboard_login(request: Request):
    """校验 Dashboard 密码并签发短期 HttpOnly Cookie。"""
    if not auth.dashboard_auth_ready():
        return shared.templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": None,
                "config_error": "请先设置 DASHBOARD_PASSWORD 和至少 32 字符的 SESSION_SECRET。",
            },
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )

    body = await request.body()
    fields = parse_qs(body[:4096].decode("utf-8", errors="replace"))
    password = fields.get("password", [""])[0]
    if len(body) > 4096 or not secrets.compare_digest(
        password.encode("utf-8"),
        shared.DASHBOARD_PASSWORD.encode("utf-8"),
    ):
        return shared.templates.TemplateResponse(
            request,
            "login.html",
            {"error": "密码不正确。", "config_error": None},
            status_code=401,
            headers={"Cache-Control": "no-store"},
        )

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        shared.DASHBOARD_SESSION_COOKIE,
        auth.make_dashboard_session(),
        max_age=shared.DASHBOARD_SESSION_SECONDS,
        path="/",
        secure=True,
        httponly=True,
        samesite="strict",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/dashboard/logout")
async def dashboard_logout():
    response = RedirectResponse(url="/dashboard/login", status_code=303)
    response.delete_cookie(
        shared.DASHBOARD_SESSION_COOKIE,
        path="/",
        secure=True,
        httponly=True,
        samesite="strict",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard - 整合的记忆管理界面"""
    if not shared.DATABASE_ENABLED:
        return shared.templates.TemplateResponse(
            request,
            "database_disabled.html",
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    return shared.templates.TemplateResponse(
        request,
        "dashboard.html",
        {"memory_enabled": shared.MEMORY_ENABLED},
    )



# ============================================================
# 模型列表 API（/api/models）
# 设置面板的 combo-box 用，根据 API_BASE_URL 自动适配
# ============================================================

@settings_router.get("/api/models")
async def get_models():
    """获取可用模型列表（根据 API_BASE_URL 自动适配）"""
    is_openrouter = "openrouter.ai" in shared.API_BASE_URL
    is_google = "googleapis.com" in shared.API_BASE_URL or "generativelanguage" in shared.API_BASE_URL
    is_openai = "api.openai.com" in shared.API_BASE_URL

    try:
        if is_openrouter:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {shared.API_KEY}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    simplified = [{"id": m.get("id"), "name": m.get("name"), "context_length": m.get("context_length")} for m in models]
                    simplified.sort(key=lambda x: x.get("name", ""))
                    return {"models": simplified, "total": len(simplified), "provider": "openrouter"}

        elif is_google:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={shared.API_KEY}"
                )
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    simplified = []
                    for m in models:
                        full_name = m.get("name", "")
                        model_id = full_name.replace("models/", "") if full_name.startswith("models/") else full_name
                        display_name = m.get("displayName", model_id)
                        supported_methods = m.get("supportedGenerationMethods", [])
                        if "generateContent" in supported_methods:
                            simplified.append({"id": model_id, "name": display_name, "context_length": m.get("inputTokenLimit"), "output_limit": m.get("outputTokenLimit")})
                    def sort_key(x):
                        name = x.get("id", "")
                        if "gemini-3" in name: return "0" + name
                        elif "gemini-2.5" in name: return "1" + name
                        elif "gemini-2.0" in name: return "2" + name
                        else: return "9" + name
                    simplified.sort(key=sort_key)
                    return {"models": simplified, "total": len(simplified), "provider": "google"}
                else:
                    print(f"[get_models] Google API 返回 {response.status_code}: {response.text}")
                    return {"error": f"Google API 返回 {response.status_code}", "models": [], "provider": "google"}

        elif is_openai:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {shared.API_KEY}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    simplified = [{"id": m.get("id", ""), "name": m.get("id", "")} for m in models if m.get("id", "").startswith(("gpt-", "o1", "o3", "o4"))]
                    simplified.sort(key=lambda x: x.get("id", ""))
                    return {"models": simplified, "total": len(simplified), "provider": "openai"}
            openai_models = [
                {"id": "gpt-4.1", "name": "GPT-4.1"},
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
                {"id": "o3-mini", "name": "o3-mini"},
            ]
            return {"models": openai_models, "total": len(openai_models), "provider": "openai"}

        else:
            return {"models": [], "total": 0, "provider": "unknown", "note": "未识别的 API，请手动输入模型名"}

    except Exception:
        return shared._api_failure("加载模型列表失败", models=[])


# ============================================================
# 高级设置面板 API（/api/settings）
# Dashboard 前端设置面板用，管理所有运行时可调配置
# ============================================================

def _mask_key(key_value: str) -> str:
    """API Key 打码：只露前5位和后4位"""
    if not key_value:
        return ""
    if len(key_value) < 10:
        return "****"
    return key_value[:5] + "****" + key_value[-4:]


def _is_masked(value: str) -> bool:
    """判断值是否是打码值（用户没改过）"""
    return "****" in str(value)


@settings_router.get(
    "/api/settings",
    dependencies=[Depends(auth.require_database_enabled)],
)
async def get_settings():
    """获取高级设置（数据库优先，fallback 到环境变量/运行时默认值）"""
    try:
        db = await db_core.get_all_gateway_config()

        settings = {
            key: parser(db[key]) if db.get(key) else getattr(shared, key)
            for key, parser in shared.SETTINGS_TYPES.items()
        }
        for key in _MASKED_KEYS:
            settings[key] = _mask_key(settings[key])
        settings["MEMORY_MODEL"] = db.get("MEMORY_MODEL") or os.environ.get("MEMORY_MODEL", "")
        settings["systemPrompt"] = db.get("systemPrompt") or shared._DEFAULT_SYSTEM_PROMPT or ""

        return {"status": "ok", "settings": settings}
    except Exception:
        return shared._api_failure("加载设置失败")


@settings_router.put(
    "/api/settings",
    dependencies=[Depends(auth.require_database_enabled)],
)
async def save_settings(request: Request):
    """保存高级设置（写入数据库 + 热更新运行时变量，立即生效无需重启）"""
    try:
        data = await request.json()
        updated = []
        skipped = []

        # 只存 os.environ 的变量
        _ENV_ONLY = {"MEMORY_MODEL": str}

        for key, value in data.items():
            # --- 打码字段特殊处理 ---
            if key in _MASKED_KEYS:
                str_val = str(value).strip()
                if _is_masked(str_val):
                    skipped.append(key)
                    continue
                if not str_val:
                    await db_core.set_gateway_config(key, "")
                    if key in shared.SETTINGS_TYPES:
                        setattr(shared, key, "")
                    os.environ[key] = ""
                    updated.append(key)
                    continue
                value = str_val

            # --- systemPrompt 特殊处理 ---
            if key == "systemPrompt":
                await db_core.set_gateway_config("systemPrompt", str(value))
                shared.invalidate_system_prompt_cache()
                updated.append("systemPrompt")
                print(f"[settings] systemPrompt 已更新（{len(str(value))} 字）")
                continue

            # --- 常规字段 ---
            await db_core.set_gateway_config(key, str(value))

            if key in shared.SETTINGS_TYPES:
                typed_value = shared.SETTINGS_TYPES[key](value)
                setattr(shared, key, typed_value)
                os.environ[key] = str(value)
                updated.append(key)
                if key in _MASKED_KEYS:
                    print(f"[settings] {key} 已更新")
                else:
                    print(f"[settings] {key} = {typed_value}")

            elif key in _ENV_ONLY:
                typed_value = _ENV_ONLY[key](value)
                os.environ[key] = str(typed_value)
                updated.append(key)
                print(f"[settings] {key} = {typed_value} (env)")

            else:
                skipped.append(key)

        if updated:
            shared.sync_memory_extractor_config()
        if (
            "CONVERSATION_RECALL_ENABLED" in updated
            and shared.CONVERSATION_RECALL_ENABLED
        ):
            await db_search.rebuild_content_tsv()
            db_search.kick_embedding_backfill()

        return {
            "status": "ok",
            "updated": updated,
            "skipped": skipped,
            "message": f"已更新 {len(updated)} 项配置，立即生效"
        }
    except Exception:
        return shared._api_failure("保存设置失败")
