"""
tools.py — Built-in free tools + dynamic custom tool registry.
 
Built-in tools (3):
  1. get_datetime        — current date, time, timezone info
  2. calculate           — safe arithmetic expression evaluator
  3. word_counter        — word / char / sentence / paragraph stats
 
Dynamic tools:
  - Custom tools added at runtime via the UI are stored in
    st.session_state["custom_tools"] as a list of dicts:
      { "name", "description", "parameters": {OpenAI JSON schema props},
        "required": [...] }
  - dispatch_tool() handles them by returning a placeholder response
    (the LLM itself fills in the logic via its own knowledge).
"""
 
import math
import datetime
import re
import ast
import operator as op
import streamlit as st
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Get current date & time
# ─────────────────────────────────────────────────────────────────────────────
def get_datetime(timezone: str = "local") -> dict:
    """Return the current date, time, weekday, and ISO timestamp."""
    now = datetime.datetime.now()
    utc_now = datetime.datetime.utcnow()
    return {
        "local_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_datetime": utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "date": now.strftime("%B %d, %Y"),
        "time": now.strftime("%I:%M:%S %p"),
        "weekday": now.strftime("%A"),
        "iso8601": now.isoformat(),
        "unix_timestamp": int(now.timestamp()),
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Safe calculator
# ─────────────────────────────────────────────────────────────────────────────
_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
    ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
}
 
def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        fn = _ALLOWED_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return fn(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        fn = _ALLOWED_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported unary operator")
        return fn(operand)
    if isinstance(node, ast.Call):
        allowed_funcs = {
            "sqrt": math.sqrt, "abs": abs, "round": round,
            "floor": math.floor, "ceil": math.ceil,
            "log": math.log, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
        }
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in allowed_funcs:
            raise ValueError(f"Function '{func_name}' not allowed")
        args = [_safe_eval(a) for a in node.args]
        return allowed_funcs[func_name](*args)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")
 
 
def calculate(expression: str) -> dict:
    """Safely evaluate a math expression (no eval/exec). Supports +,-,*,/,**,%,sqrt,log,sin,cos,tan,etc."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return {
            "expression": expression,
            "result": result,
            "formatted": f"{result:,}" if isinstance(result, (int, float)) else str(result),
        }
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Could not evaluate: {e}"}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Word / text statistics
# ─────────────────────────────────────────────────────────────────────────────
def word_counter(text: str) -> dict:
    """Count words, characters, sentences, paragraphs, and find the top 5 frequent words."""
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
 
    freq: dict[str, int] = {}
    for w in words:
        key = w.lower()
        freq[key] = freq.get(key, 0) + 1
    top5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
 
    return {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(text.replace(" ", "")),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0,
        "top_5_words": [{"word": w, "count": c} for w, c in top5],
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Registry & dispatch
# ─────────────────────────────────────────────────────────────────────────────
TOOL_REGISTRY = {
    "get_datetime": {
        "fn": get_datetime,
        "description": "Get current date & time",
        "group": "🕐 Utilities",
        "schema": {
            "name": "get_datetime",
            "description": "Returns the current local and UTC date, time, weekday, ISO timestamp, and Unix timestamp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone hint (informational only). Default: 'local'.",
                        "default": "local",
                    }
                },
                "required": [],
            },
        },
    },
    "calculate": {
        "fn": calculate,
        "description": "Safe math calculator",
        "group": "🕐 Utilities",
        "schema": {
            "name": "calculate",
            "description": "Safely evaluate a mathematical expression. Supports +, -, *, /, **, %, sqrt(), log(), sin(), cos(), tan(), floor(), ceil(), abs(), round().",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2 ** 10 + sqrt(144)'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    "word_counter": {
        "fn": word_counter,
        "description": "Word / text statistics",
        "group": "🕐 Utilities",
        "schema": {
            "name": "word_counter",
            "description": "Count words, characters, sentences, paragraphs, and top-5 frequent words in a text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze.",
                    }
                },
                "required": ["text"],
            },
        },
    },
}
 
TOOL_GROUPS = {
    "🕐 Utilities": ["get_datetime", "calculate", "word_counter"],
}
 
 
def get_openai_tool_schemas(extra_schemas: list | None = None) -> list[dict]:
    """Return OpenAI-format tool schemas for built-in + custom tools."""
    schemas = [
        {"type": "function", "function": info["schema"]}
        for info in TOOL_REGISTRY.values()
    ]
    if extra_schemas:
        schemas.extend(extra_schemas)
    return schemas
 
 
def dispatch_tool(name: str, args: dict, custom_tools: list | None = None) -> dict:
    """Call a built-in tool by name, or return a stub for custom tools."""
    if name in TOOL_REGISTRY:
        try:
            return TOOL_REGISTRY[name]["fn"](**args)
        except Exception as e:
            return {"error": str(e)}
 
    # Custom tool — return a structured stub so the LLM can respond naturally
    return {
        "tool": name,
        "args": args,
        "note": (
            f"Custom tool '{name}' was called with the provided arguments. "
            "Please use your knowledge to respond to this request."
        ),
    }
 
 