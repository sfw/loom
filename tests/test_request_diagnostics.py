from __future__ import annotations

from loom.models.request_diagnostics import collect_request_diagnostics, infer_call_origin


def test_collect_request_diagnostics_counts_messages_and_tool_args() -> None:
    messages = [
        {"role": "system", "content": "System rules"},
        {"role": "user", "content": "Find competitors for Encor by EPCOR."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\":\"encor competitors\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "{\"success\":true,\"output\":\"result\"}",
        },
    ]
    tools = [
        {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]

    diag = collect_request_diagnostics(
        messages=messages,
        tools=tools,
        origin="test.request_diagnostics",
    )

    assert diag.origin == "test.request_diagnostics"
    assert diag.message_count == 4
    assert diag.tool_count == 1
    assert diag.assistant_tool_calls == 1
    assert diag.assistant_tool_arg_chars > 0
    assert diag.request_bytes >= diag.messages_bytes
    assert diag.request_est_tokens >= 1
    assert diag.largest_message_chars > 0

    payload = diag.to_event_payload()
    assert payload["origin"] == "test.request_diagnostics"
    assert payload["request_size_tier"] in {"normal", "large", "oversize_risk"}


def test_collect_request_diagnostics_handles_non_string_content() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": {"summary": "ok", "score": 0.9}},
    ]

    diag = collect_request_diagnostics(messages=messages)

    assert diag.message_count == 2
    assert diag.messages_chars > 0
    assert diag.messages_bytes > 0
    assert diag.request_est_tokens >= 1
    assert diag.origin


def test_infer_call_origin_returns_identifier() -> None:
    origin = infer_call_origin()
    assert isinstance(origin, str)
    assert origin.strip() != ""
