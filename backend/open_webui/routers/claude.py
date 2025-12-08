import asyncio
import json
import logging
from typing import Optional

import aiohttp
from aiocache import cached
from urllib.parse import quote

from fastapi import Depends, HTTPException, Request, APIRouter
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
)
from pydantic import BaseModel
from starlette.background import BackgroundTask

from open_webui.models.models import Models
from open_webui.env import (
    MODELS_CACHE_TTL,
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST,
    ENABLE_FORWARD_USER_INFO_HEADERS,
    BYPASS_MODEL_ACCESS_CONTROL,
)
from open_webui.models.users import UserModel

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS

from open_webui.utils.payload import (
    apply_model_params_to_body_openai,
    apply_system_prompt_to_body,
)

from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("CLAUDE", "INFO"))


##########################################
#
# Utility functions
#
##########################################


async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    if response:
        response.close()
    if session:
        await session.close()


async def get_headers_and_cookies(
    request: Request,
    url,
    key=None,
    config=None,
    metadata: Optional[dict] = None,
    user: UserModel = None,
):
    cookies = {}
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": config.get("api_version", "2023-06-01") if config else "2023-06-01",
        **(
            {
                "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                "X-OpenWebUI-User-Id": user.id,
                "X-OpenWebUI-User-Email": user.email,
                "X-OpenWebUI-User-Role": user.role,
                **(
                    {"X-OpenWebUI-Chat-Id": metadata.get("chat_id")}
                    if metadata and metadata.get("chat_id")
                    else {}
                ),
            }
            if ENABLE_FORWARD_USER_INFO_HEADERS
            else {}
        ),
    }

    token = None
    auth_type = config.get("auth_type") if config else None

    if auth_type == "bearer" or auth_type is None:
        # Default to bearer if not specified
        token = f"{key}"
    elif auth_type == "none":
        token = None
    elif auth_type == "session":
        cookies = request.cookies
        token = request.state.token.credentials
    elif auth_type == "system_oauth":
        cookies = request.cookies

        oauth_token = None
        try:
            if request.cookies.get("oauth_session_id", None):
                oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                    user.id,
                    request.cookies.get("oauth_session_id", None),
                )
        except Exception as e:
            log.error(f"Error getting OAuth token: {e}")

        if oauth_token:
            token = f"{oauth_token.get('access_token', '')}"

    if token:
        headers["x-api-key"] = token

    if config and config.get("headers") and isinstance(config.get("headers"), dict):
        headers = {**headers, **config.get("headers")}

    return headers, cookies


def convert_openai_to_claude_payload(payload: dict) -> dict:
    """
    Convert OpenAI-style chat completion payload to Claude's Messages API format.
    """
    claude_payload = {}

    # Handle model
    claude_payload["model"] = payload.get("model", "claude-3-5-sonnet-20241022")

    # Handle messages - separate system message from conversation
    messages = payload.get("messages", [])
    system_messages = []
    conversation_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_messages.append(content)
        elif role in ["user", "assistant"]:
            # Claude expects simple format for basic messages
            conversation_messages.append({
                "role": role,
                "content": content
            })

    # Add system prompt if present
    if system_messages:
        claude_payload["system"] = "\n\n".join(system_messages)

    claude_payload["messages"] = conversation_messages

    # Handle max_tokens - required for Claude
    if "max_tokens" in payload:
        claude_payload["max_tokens"] = payload["max_tokens"]
    elif "max_completion_tokens" in payload:
        claude_payload["max_tokens"] = payload["max_completion_tokens"]
    else:
        # Default max_tokens if not specified
        claude_payload["max_tokens"] = 4096

    # Handle optional parameters
    if "temperature" in payload:
        claude_payload["temperature"] = payload["temperature"]

    if "top_p" in payload:
        claude_payload["top_p"] = payload["top_p"]

    if "stop" in payload:
        stop_sequences = payload["stop"]
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        claude_payload["stop_sequences"] = stop_sequences

    if "stream" in payload:
        claude_payload["stream"] = payload["stream"]

    # Handle metadata
    if "user" in payload:
        claude_payload["metadata"] = {"user_id": payload["user"]}

    return claude_payload


def convert_claude_to_openai_response(claude_response: dict, model: str, stream: bool = False) -> dict:
    """
    Convert Claude API response to OpenAI-compatible format.
    """
    if stream:
        # For streaming responses, we'll handle this in the streaming function
        return claude_response

    openai_response = {
        "id": claude_response.get("id", ""),
        "object": "chat.completion",
        "created": int(claude_response.get("id", "").split("_")[1]) if "_" in claude_response.get("id", "") else 0,
        "model": model,
        "choices": [],
        "usage": {}
    }

    # Convert content blocks to OpenAI format
    content = claude_response.get("content", [])
    message_content = ""

    for block in content:
        if block.get("type") == "text":
            message_content += block.get("text", "")

    openai_response["choices"].append({
        "index": 0,
        "message": {
            "role": "assistant",
            "content": message_content
        },
        "finish_reason": claude_response.get("stop_reason", "stop")
    })

    # Convert usage stats
    usage = claude_response.get("usage", {})
    openai_response["usage"] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    }

    return openai_response


async def convert_claude_stream_to_openai(stream_content, model: str):
    """
    Convert Claude streaming response to OpenAI-compatible SSE format.
    """
    async for line in stream_content:
        if line:
            line_str = line.decode('utf-8').strip()

            if line_str.startswith("data: "):
                data_str = line_str[6:]

                if data_str == "[DONE]":
                    yield b"data: [DONE]\n\n"
                    continue

                try:
                    claude_chunk = json.loads(data_str)
                    event_type = claude_chunk.get("type")

                    if event_type == "message_start":
                        # Initial message metadata
                        message = claude_chunk.get("message", {})
                        openai_chunk = {
                            "id": message.get("id", ""),
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

                    elif event_type == "content_block_start":
                        # Content block started
                        pass

                    elif event_type == "content_block_delta":
                        # Text delta
                        delta = claude_chunk.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            openai_chunk = {
                                "id": "",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

                    elif event_type == "content_block_stop":
                        # Content block ended
                        pass

                    elif event_type == "message_delta":
                        # Message metadata delta (stop reason, usage)
                        delta = claude_chunk.get("delta", {})
                        if "stop_reason" in delta:
                            openai_chunk = {
                                "id": "",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": delta.get("stop_reason", "stop")
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

                    elif event_type == "message_stop":
                        # End of message
                        yield b"data: [DONE]\n\n"

                except json.JSONDecodeError:
                    continue
            else:
                # Pass through other SSE fields like event type
                yield (line_str + "\n").encode('utf-8')


# Claude model list - these are the current Claude models as of 2024
CLAUDE_MODELS = [
    {
        "id": "claude-3-5-sonnet-20241022",
        "name": "Claude 3.5 Sonnet (Oct 2024)",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-3-5-haiku-20241022",
        "name": "Claude 3.5 Haiku",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-3-opus-20240229",
        "name": "Claude 3 Opus",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-3-sonnet-20240229",
        "name": "Claude 3 Sonnet",
        "owned_by": "anthropic",
    },
    {
        "id": "claude-3-haiku-20240307",
        "name": "Claude 3 Haiku",
        "owned_by": "anthropic",
    },
]


##########################################
#
# API routes
#
##########################################

router = APIRouter()


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "ENABLE_CLAUDE_API": request.app.state.config.ENABLE_CLAUDE_API,
        "CLAUDE_API_BASE_URLS": request.app.state.config.CLAUDE_API_BASE_URLS,
        "CLAUDE_API_KEYS": request.app.state.config.CLAUDE_API_KEYS,
        "CLAUDE_API_CONFIGS": request.app.state.config.CLAUDE_API_CONFIGS,
    }


class ClaudeConfigForm(BaseModel):
    ENABLE_CLAUDE_API: Optional[bool] = None
    CLAUDE_API_BASE_URLS: list[str]
    CLAUDE_API_KEYS: list[str]
    CLAUDE_API_CONFIGS: dict


@router.post("/config/update")
async def update_config(
    request: Request, form_data: ClaudeConfigForm, user=Depends(get_admin_user)
):
    request.app.state.config.ENABLE_CLAUDE_API = form_data.ENABLE_CLAUDE_API
    request.app.state.config.CLAUDE_API_BASE_URLS = form_data.CLAUDE_API_BASE_URLS
    request.app.state.config.CLAUDE_API_KEYS = form_data.CLAUDE_API_KEYS

    # Check if API KEYS length is same than API URLS length
    if len(request.app.state.config.CLAUDE_API_KEYS) != len(
        request.app.state.config.CLAUDE_API_BASE_URLS
    ):
        if len(request.app.state.config.CLAUDE_API_KEYS) > len(
            request.app.state.config.CLAUDE_API_BASE_URLS
        ):
            request.app.state.config.CLAUDE_API_KEYS = (
                request.app.state.config.CLAUDE_API_KEYS[
                    : len(request.app.state.config.CLAUDE_API_BASE_URLS)
                ]
            )
        else:
            request.app.state.config.CLAUDE_API_KEYS += [""] * (
                len(request.app.state.config.CLAUDE_API_BASE_URLS)
                - len(request.app.state.config.CLAUDE_API_KEYS)
            )

    request.app.state.config.CLAUDE_API_CONFIGS = form_data.CLAUDE_API_CONFIGS

    # Remove the API configs that are not in the API URLS
    keys = list(map(str, range(len(request.app.state.config.CLAUDE_API_BASE_URLS))))
    request.app.state.config.CLAUDE_API_CONFIGS = {
        key: value
        for key, value in request.app.state.config.CLAUDE_API_CONFIGS.items()
        if key in keys
    }

    return {
        "ENABLE_CLAUDE_API": request.app.state.config.ENABLE_CLAUDE_API,
        "CLAUDE_API_BASE_URLS": request.app.state.config.CLAUDE_API_BASE_URLS,
        "CLAUDE_API_KEYS": request.app.state.config.CLAUDE_API_KEYS,
        "CLAUDE_API_CONFIGS": request.app.state.config.CLAUDE_API_CONFIGS,
    }


async def get_all_models_responses(request: Request, user: UserModel) -> list:
    if not request.app.state.config.ENABLE_CLAUDE_API:
        return []

    # Check if API KEYS length is same than API URLS length
    num_urls = len(request.app.state.config.CLAUDE_API_BASE_URLS)
    num_keys = len(request.app.state.config.CLAUDE_API_KEYS)

    if num_keys != num_urls:
        # if there are more keys than urls, remove the extra keys
        if num_keys > num_urls:
            new_keys = request.app.state.config.CLAUDE_API_KEYS[:num_urls]
            request.app.state.config.CLAUDE_API_KEYS = new_keys
        # if there are more urls than keys, add empty keys
        else:
            request.app.state.config.CLAUDE_API_KEYS += [""] * (num_urls - num_keys)

    responses = []

    for idx, url in enumerate(request.app.state.config.CLAUDE_API_BASE_URLS):
        api_config = request.app.state.config.CLAUDE_API_CONFIGS.get(
            str(idx), {}
        )

        enable = api_config.get("enable", True)
        model_ids = api_config.get("model_ids", [])

        if enable:
            if len(model_ids) == 0:
                # Return all Claude models
                model_list = {
                    "object": "list",
                    "data": [
                        {
                            **model,
                            "urlIdx": idx,
                        }
                        for model in CLAUDE_MODELS
                    ],
                }
            else:
                # Return only specified models
                model_list = {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "owned_by": "anthropic",
                            "claude": {"id": model_id},
                            "urlIdx": idx,
                        }
                        for model_id in model_ids
                    ],
                }

            responses.append(model_list)
        else:
            responses.append(None)

    return responses


async def get_filtered_models(models, user):
    # Filter models based on user access control
    filtered_models = []
    for model in models.get("data", []):
        model_info = Models.get_model_by_id(model["id"])
        if model_info:
            if user.id == model_info.user_id or has_access(
                user.id, type="read", access_control=model_info.access_control
            ):
                filtered_models.append(model)
    return filtered_models


@cached(
    ttl=MODELS_CACHE_TTL,
    key=lambda _, user: f"claude_all_models_{user.id}" if user else "claude_all_models",
)
async def get_all_models(request: Request, user: UserModel) -> dict[str, list]:
    log.info("get_all_models()")

    if not request.app.state.config.ENABLE_CLAUDE_API:
        return {"data": []}

    responses = await get_all_models_responses(request, user=user)

    def extract_data(response):
        if response and "data" in response:
            return response["data"]
        if isinstance(response, list):
            return response
        return None

    def merge_models_lists(model_lists):
        log.debug(f"merge_models_lists {model_lists}")
        merged_list = []

        for idx, models in enumerate(model_lists):
            if models is not None and "error" not in models:
                merged_list.extend(
                    [
                        {
                            **model,
                            "name": model.get("name", model["id"]),
                            "owned_by": "anthropic",
                            "claude": model,
                            "connection_type": model.get("connection_type", "external"),
                            "urlIdx": idx,
                        }
                        for model in models
                        if (model.get("id") or model.get("name"))
                    ]
                )

        return merged_list

    models = {"data": merge_models_lists(map(extract_data, responses))}
    log.debug(f"models: {models}")

    request.app.state.CLAUDE_MODELS = {model["id"]: model for model in models["data"]}
    return models


@router.get("/models")
@router.get("/models/{url_idx}")
async def get_models(
    request: Request, url_idx: Optional[int] = None, user=Depends(get_verified_user)
):
    models = {
        "data": [],
    }

    if url_idx is None:
        models = await get_all_models(request, user=user)
    else:
        api_config = request.app.state.config.CLAUDE_API_CONFIGS.get(
            str(url_idx), {}
        )

        model_ids = api_config.get("model_ids", [])

        if len(model_ids) == 0:
            # Return all Claude models
            models = {
                "data": CLAUDE_MODELS,
                "object": "list",
            }
        else:
            # Return only specified models
            models = {
                "data": [
                    {
                        "id": model_id,
                        "name": model_id,
                        "owned_by": "anthropic",
                    }
                    for model_id in model_ids
                ],
                "object": "list",
            }

    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models["data"] = await get_filtered_models(models, user)

    return models


class ConnectionVerificationForm(BaseModel):
    url: str
    key: str
    config: Optional[dict] = None


@router.post("/verify")
async def verify_connection(
    request: Request,
    form_data: ConnectionVerificationForm,
    user=Depends(get_admin_user),
):
    url = form_data.url
    key = form_data.key
    api_config = form_data.config or {}

    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
    ) as session:
        try:
            headers, cookies = await get_headers_and_cookies(
                request, url, key, api_config, user=user
            )

            # Claude doesn't have a models endpoint, so we'll test with a simple message
            test_payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1,
                "messages": [
                    {"role": "user", "content": "Hi"}
                ]
            }

            async with session.post(
                f"{url}/v1/messages",
                headers=headers,
                cookies=cookies,
                json=test_payload,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as r:
                try:
                    response_data = await r.json()
                except Exception:
                    response_data = await r.text()

                if r.status != 200:
                    if isinstance(response_data, (dict, list)):
                        return JSONResponse(
                            status_code=r.status, content=response_data
                        )
                    else:
                        return PlainTextResponse(
                            status_code=r.status, content=response_data
                        )

                # Return success with available models
                return {
                    "data": CLAUDE_MODELS,
                    "object": "list",
                }

        except aiohttp.ClientError as e:
            log.exception(f"Client error: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Open WebUI: Server Connection Error"
            )
        except Exception as e:
            log.exception(f"Unexpected error: {e}")
            raise HTTPException(
                status_code=500, detail="Open WebUI: Server Connection Error"
            )


@router.post("/chat/completions")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_filter: Optional[bool] = False,
):
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    idx = 0

    payload = {**form_data}
    metadata = payload.pop("metadata", None)

    model_id = form_data.get("model")
    model_info = Models.get_model_by_id(model_id)

    # Check model info and override the payload
    if model_info:
        if model_info.base_model_id:
            payload["model"] = model_info.base_model_id
            model_id = model_info.base_model_id

        params = model_info.params.model_dump()

        if params:
            system = params.pop("system", None)

            payload = apply_model_params_to_body_openai(params, payload)
            payload = apply_system_prompt_to_body(system, payload, metadata, user)

        # Check if user has access to the model
        if not bypass_filter and user.role == "user":
            if not (
                user.id == model_info.user_id
                or has_access(
                    user.id, type="read", access_control=model_info.access_control
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Model not found",
                )
    elif not bypass_filter:
        if user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Model not found",
            )

    await get_all_models(request, user=user)
    model = request.app.state.CLAUDE_MODELS.get(model_id)
    if model:
        idx = model["urlIdx"]
    else:
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        )

    # Get the API config for the model
    api_config = request.app.state.config.CLAUDE_API_CONFIGS.get(
        str(idx), {}
    )

    prefix_id = api_config.get("prefix_id", None)
    if prefix_id:
        payload["model"] = payload["model"].replace(f"{prefix_id}.", "")

    # Add user info to the payload if the model is a pipeline
    if "pipeline" in model and model.get("pipeline"):
        payload["user"] = {
            "name": user.name,
            "id": user.id,
            "email": user.email,
            "role": user.role,
        }

    url = request.app.state.config.CLAUDE_API_BASE_URLS[idx]
    key = request.app.state.config.CLAUDE_API_KEYS[idx]

    # Convert OpenAI format to Claude format
    original_model = payload["model"]
    is_streaming = payload.get("stream", False)
    claude_payload = convert_openai_to_claude_payload(payload)

    headers, cookies = await get_headers_and_cookies(
        request, url, key, api_config, metadata, user=user
    )

    request_url = f"{url}/v1/messages"

    r = None
    session = None
    streaming = False
    response = None

    try:
        session = aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        )

        r = await session.request(
            method="POST",
            url=request_url,
            json=claude_payload,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        # Check if response is SSE
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            streaming = True
            return StreamingResponse(
                convert_claude_stream_to_openai(r.content, original_model),
                status_code=r.status,
                headers={"Content-Type": "text/event-stream"},
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            try:
                response = await r.json()
            except Exception as e:
                log.error(e)
                response = await r.text()

            if r.status >= 400:
                if isinstance(response, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response)
                else:
                    return PlainTextResponse(status_code=r.status, content=response)

            # Convert Claude response to OpenAI format
            openai_response = convert_claude_to_openai_response(response, original_model, is_streaming)
            return openai_response
    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=r.status if r else 500,
            detail="Open WebUI: Server Connection Error",
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)
