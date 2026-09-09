"""
GPT Custom Toolbox — OpenAI SDK edition
One-page Streamlit app with 3 built-in tools + dynamic custom tool builder.
"""
 
import json
import os
import re
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOL_REGISTRY, TOOL_GROUPS, dispatch_tool, get_openai_tool_schemas
 
load_dotenv()
 
# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPT Custom Toolbox",
    page_icon="🧰",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
 
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: #0d1117 !important;
    overflow: hidden !important;
    height: 100vh !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
 
[data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
 
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.block-container {
    margin: 0 !important;
    padding: 0 !important;
    background: #0d1117 !important;
    overflow: hidden !important;
    height: 100vh !important;
    max-width: 100% !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
 
/* ── hide Streamlit chrome ── */
#MainMenu, header[data-testid="stHeader"], footer,
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
 
/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] > div { padding: 1rem 0.75rem !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }
 
/* ── sidebar components ── */
.sb-logo {
    font-size: 1rem; font-weight: 700; color: #e6edf3;
    display: flex; align-items: center; gap: 0.5rem;
    padding-bottom: 0.9rem; border-bottom: 1px solid #21262d; margin-bottom: 0.75rem;
}
.sb-badge {
    background: #1f6feb; color: #fff; font-size: 0.62rem;
    padding: 2px 8px; border-radius: 20px; font-weight: 600; letter-spacing: 0.04em;
}
.sb-section {
    color: #6e7681; font-size: 0.67rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin: 0.9rem 0 0.3rem;
}
 
/* ── topbar ── */
.topbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 999;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 1.2rem; height: 48px;
    background: #161b22; border-bottom: 1px solid #21262d;
}
.topbar-title {
    font-size: 0.95rem; font-weight: 700; color: #e6edf3;
    display: flex; align-items: center; gap: 0.45rem;
}
.topbar-pills { display: flex; gap: 0.5rem; align-items: center; }
.topbar-pill {
    background: #21262d; color: #8b949e; font-size: 0.67rem;
    padding: 3px 10px; border-radius: 20px; border: 1px solid #30363d;
    white-space: nowrap;
}
 
/* ── main content offset for topbar ── */
[data-testid="stMainBlockContainer"] {
    padding-top: 48px !important;
    height: 100vh !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
 
/* ── column panel headers ── */
.panel-header {
    padding: 0.55rem 0.2rem 0.45rem;
    border-bottom: 1px solid #21262d;
    font-size: 0.72rem; font-weight: 600; color: #6e7681;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.5rem; flex-shrink: 0;
}
 
/* ── chat messages ── */
.msg-row { display: flex; gap: 0.55rem; align-items: flex-start; margin: 0.25rem 0; }
.msg-row.user { flex-direction: row-reverse; }
.avatar {
    width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 700; margin-top: 2px;
}
.av-user { background: #1f6feb; color: #fff; }
.av-bot  { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
.bubble {
    max-width: 84%; padding: 0.5rem 0.8rem; border-radius: 12px;
    font-size: 0.83rem; line-height: 1.55; color: #e6edf3; word-break: break-word;
}
.bubble.user { background: #1f6feb18; border: 1px solid #1f6feb40; border-top-right-radius: 3px; }
.bubble.bot  { background: #161b22;   border: 1px solid #21262d;   border-top-left-radius: 3px; }
.tool-result {
    background: #0d1117; border: 1px solid #238636; border-radius: 8px;
    padding: 0.4rem 0.7rem; margin: 0.15rem 0 0.15rem 2rem;
    font-size: 0.74rem; color: #3fb950; font-family: 'Courier New', monospace;
}
.tool-result b { color: #58a6ff; }
.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 100%; gap: 0.4rem; color: #30363d;
}
.empty-state .icon { font-size: 2.2rem; }
.empty-state p { font-size: 0.8rem; margin: 0; color: #484f58; }
 
/* ── quick prompt label ── */
.quick-label {
    font-size: 0.67rem; color: #6e7681; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 0.4rem 0 0.3rem;
}
 
/* ── format hint box ── */
.hint-box {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 0.65rem 0.8rem; font-size: 0.74rem; color: #6e7681;
    line-height: 1.65; margin-top: 0.7rem;
}
.hint-box b { color: #8b949e; }
 
/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: #161b22 !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 8px !important;
    font-size: 0.83rem !important; resize: none !important;
    transition: border-color 0.15s !important;
}
.stTextArea textarea:focus {
    border-color: #1f6feb !important;
    box-shadow: 0 0 0 2px #1f6feb1a !important;
    outline: none !important;
}
.stTextInput input {
    background: #161b22 !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
    font-size: 0.82rem !important;
}
.stTextInput input:focus { border-color: #1f6feb !important; }
.stSelectbox > div > div {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 6px !important; color: #e6edf3 !important;
}
.stCheckbox label { color: #c9d1d9 !important; font-size: 0.81rem !important; }
.stButton > button {
    border-radius: 6px !important; font-size: 0.81rem !important;
    font-weight: 500 !important; transition: all 0.15s !important;
    border: 1px solid transparent !important;
}
.stButton > button[kind="primary"] {
    background: #1f6feb !important; color: #fff !important;
}
.stButton > button[kind="primary"]:hover { background: #388bfd !important; }
.stButton > button[kind="secondary"] {
    background: #21262d !important; border-color: #30363d !important; color: #c9d1d9 !important;
}
.stButton > button[kind="secondary"]:hover { border-color: #58a6ff !important; color: #58a6ff !important; }
 
/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #21262d !important;
    gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #6e7681 !important;
    font-size: 0.77rem !important; font-weight: 500 !important;
    padding: 0.45rem 0.85rem !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important; border-bottom-color: #1f6feb !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0.7rem 0 !important; }
 
/* ── expander ── */
.streamlit-expanderHeader {
    background: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 6px !important; color: #c9d1d9 !important; font-size: 0.81rem !important;
}
.streamlit-expanderContent {
    background: #0d1117 !important; border: 1px solid #21262d !important;
    border-top: none !important; border-radius: 0 0 6px 6px !important;
}
 
/* ── alerts ── */
.stAlert { border-radius: 6px !important; font-size: 0.81rem !important; }
 
/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)
 
# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if "custom_tools" not in st.session_state:
    st.session_state.custom_tools = []   # list of {name, description, parameters, required, system_hint}
if "tool_added_msg" not in st.session_state:
    st.session_state.tool_added_msg = ""
if "tool_to_remove" not in st.session_state:
    st.session_state.tool_to_remove = None
 
# ── helpers ───────────────────────────────────────────────────────────────────
def _tool_name_from_desc(desc: str) -> str:
    """Convert a human description to a snake_case tool name."""
    name = re.sub(r"[^a-zA-Z0-9 ]", "", desc.lower())
    return "_".join(name.split()[:5]) or "custom_tool"
 
 
def _build_json_schema(params_text: str) -> tuple[dict, list[str]]:
    """
    Parse a simple 'name:type:description' multiline format into an OpenAI
    JSON schema properties dict and required list.
 
    Each line: param_name : type : description
    type can be: string, number, integer, boolean
    Prefix with * to mark required, e.g.  *query:string:The search query
    """
    properties: dict = {}
    required: list[str] = []
    for line in params_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        is_required = line.startswith("*")
        line = line.lstrip("*").strip()
        parts = [p.strip() for p in line.split(":", 2)]
        if len(parts) < 2:
            continue
        pname = re.sub(r"[^a-zA-Z0-9_]", "_", parts[0])
        ptype = parts[1].lower() if parts[1].lower() in ("string", "number", "integer", "boolean") else "string"
        pdesc = parts[2] if len(parts) == 3 else ""
        properties[pname] = {"type": ptype, "description": pdesc}
        if is_required:
            required.append(pname)
    return properties, required
 
 
def _custom_tool_to_openai_schema(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": tool["parameters"],
                "required": tool["required"],
            },
        },
    }
 
 
def render_chat(messages: list):
    visible = [m for m in messages if m.get("role") in ("user", "assistant", "tool")]
    if not visible:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">💬</div>'
            '<p>No messages yet — type below to start</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "user":
            st.markdown(
                f'<div class="msg-row user">'
                f'<div class="avatar av-user">You</div>'
                f'<div class="bubble user">{content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif role == "assistant" and content:
            st.markdown(
                f'<div class="msg-row">'
                f'<div class="avatar av-bot">AI</div>'
                f'<div class="bubble bot">{content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif role == "tool":
            try:
                result_str = json.dumps(json.loads(content), indent=2)
            except Exception:
                result_str = content
            st.markdown(
                f'<div class="tool-result">'
                f'<b>⚙ {m.get("name","tool")}</b><br>'
                f'<pre style="margin:0.25rem 0 0;font-size:0.73rem;color:#3fb950;white-space:pre-wrap">{result_str}</pre>'
                f'</div>',
                unsafe_allow_html=True,
            )
 
 
# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-logo">🧰 GPT Custom Toolbox'
        '<span class="sb-badge">OpenAI</span></div>',
        unsafe_allow_html=True,
    )
 
    st.markdown('<div class="sb-section">API Key</div>', unsafe_allow_html=True)
    key_in = st.text_input("key", value=st.session_state.api_key,
                           type="password", placeholder="sk-…",
                           label_visibility="collapsed")
    if key_in:
        st.session_state.api_key = key_in
 
    st.markdown('<div class="sb-section">Model</div>', unsafe_allow_html=True)
    model = st.selectbox("model",
                         ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                         index=1, label_visibility="collapsed")
 
    st.markdown('<div class="sb-section">Temperature</div>', unsafe_allow_html=True)
    temperature = st.slider("temp", 0.0, 2.0, 0.7, 0.1, label_visibility="collapsed")
 
    # ── built-in tool toggles ──
    enabled_tools: dict[str, bool] = {}
    for group_name, tool_names in TOOL_GROUPS.items():
        st.markdown(f'<div class="sb-section">{group_name}</div>', unsafe_allow_html=True)
        for name in tool_names:
            info = TOOL_REGISTRY[name]
            enabled_tools[name] = st.checkbox(info["description"], value=True, key=f"t_{name}")
 
    # ── custom tool toggles ──
    if st.session_state.custom_tools:
        st.markdown('<div class="sb-section">Custom Tools</div>', unsafe_allow_html=True)
        for ct in st.session_state.custom_tools:
            enabled_tools[ct["name"]] = st.checkbox(
                ct["description"][:38], value=True, key=f"ct_{ct['name']}"
            )
 
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
 
# ── topbar ────────────────────────────────────────────────────────────────────
tool_count = len(TOOL_REGISTRY) + len(st.session_state.custom_tools)
st.markdown(
    f'<div class="topbar">'
    f'  <div class="topbar-title">🧰 GPT Custom Toolbox</div>'
    f'  <div class="topbar-pills">'
    f'    <span class="topbar-pill">OpenAI SDK</span>'
    f'    <span class="topbar-pill">⚙ {tool_count} tools</span>'
    f'    <span class="topbar-pill">🌡 {temperature:.1f}</span>'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)
 
# ── two-column layout ─────────────────────────────────────────────────────────
col_chat, col_builder = st.columns([3, 2], gap="small")
 
# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Chat
# ══════════════════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown('<div class="panel-header">💬 Conversation</div>', unsafe_allow_html=True)
 
    with st.container(height=420, border=False):
        render_chat(st.session_state.messages)
 
    user_input = st.text_area(
        "msg", placeholder="Type a message…  e.g. 'What time is it?' · 'Calculate 2**32' · 'Count words in…'",
        height=72, label_visibility="collapsed", key="user_input",
    )
    send_col, _ = st.columns([1, 3])
    send = send_col.button("Send ▶", type="primary", use_container_width=True)
 
    # quick prompts
    st.markdown('<div class="quick-label">Quick prompts</div>', unsafe_allow_html=True)
    qcols = st.columns(3)
    quick_prompts = [
        ("🕐", "What is today's date and time?"),
        ("🧮", "Calculate sqrt(144) + 2**8"),
        ("📝", "Count words: The quick brown fox jumps over the lazy dog"),
    ]
    quick_selected = None
    for i, (icon, qp) in enumerate(quick_prompts):
        label = icon + " " + qp[:22] + "…"
        if qcols[i].button(label, key=f"qp_{i}", use_container_width=True):
            quick_selected = qp
 
# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Tool Builder
# ══════════════════════════════════════════════════════════════════════════════
with col_builder:
    st.markdown('<div class="panel-header">🛠 Tool Builder</div>', unsafe_allow_html=True)
 
    tab_add, tab_list, tab_test = st.tabs([
        "➕ Add Tool",
        f"📋 My Tools ({len(st.session_state.custom_tools)})",
        "🧪 Test",
    ])
 
    # ── TAB 1: Add Tool ──────────────────────────────────────────────────────
    with tab_add:
        if st.session_state.tool_added_msg:
            st.success(st.session_state.tool_added_msg)
            st.session_state.tool_added_msg = ""
 
        with st.form("tool_builder_form", clear_on_submit=True):
            tool_desc = st.text_input(
                "Description *",
                placeholder="e.g. Get weather for a city",
                help="What does this tool do?",
            )
            tool_params = st.text_area(
                "Parameters",
                placeholder="*city:string:The city name\nunit:string:celsius or fahrenheit",
                height=95,
                help="One per line: *name:type:description  (prefix * = required)",
            )
            tool_system_hint = st.text_input(
                "System hint (optional)",
                placeholder="Respond with realistic mock data when called.",
                help="Instruction added to system prompt when this tool is enabled.",
            )
            submitted = st.form_submit_button("➕ Add Tool", type="primary", use_container_width=True)
 
        if submitted and tool_desc.strip():
            name = _tool_name_from_desc(tool_desc)
            existing_names = [ct["name"] for ct in st.session_state.custom_tools]
            if name in existing_names:
                name = f"{name}_{len(existing_names)}"
            properties, required = _build_json_schema(tool_params)
            st.session_state.custom_tools.append({
                "name": name,
                "description": tool_desc.strip(),
                "parameters": properties,
                "required": required,
                "system_hint": tool_system_hint.strip(),
            })
            st.session_state.tool_added_msg = f"✅ **{name}** added — switch to 'My Tools' to view."
            st.rerun()
        elif submitted:
            st.warning("Please enter a description.")
 
        st.markdown(
            '<div class="hint-box">'
            '<b>Parameter format</b><br>'
            '<code style="color:#3fb950">*name:type:description</code> — required<br>'
            '<code style="color:#8b949e">name:type:description</code> — optional<br>'
            '<b>Types:</b> string · number · integer · boolean'
            '</div>',
            unsafe_allow_html=True,
        )
 
    # ── TAB 2: My Tools ──────────────────────────────────────────────────────
    with tab_list:
        if st.session_state.tool_to_remove is not None:
            idx = st.session_state.tool_to_remove
            if 0 <= idx < len(st.session_state.custom_tools):
                st.session_state.custom_tools.pop(idx)
            st.session_state.tool_to_remove = None
            st.rerun()
 
        if not st.session_state.custom_tools:
            st.markdown(
                '<div class="empty-state" style="height:180px">'
                '<div class="icon">🔧</div>'
                '<p>No custom tools yet</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for i, ct in enumerate(st.session_state.custom_tools):
                with st.expander(f"⚙ {ct['name']}", expanded=False):
                    st.markdown(
                        f'<div style="font-size:0.79rem;color:#8b949e;margin-bottom:0.4rem">{ct["description"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.json({
                        "name": ct["name"],
                        "parameters": {
                            "type": "object",
                            "properties": ct["parameters"],
                            "required": ct["required"],
                        },
                    })
                    if ct.get("system_hint"):
                        st.caption(f"💡 {ct['system_hint']}")
                    if st.button("🗑 Remove", key=f"rm_{i}_{ct['name']}", use_container_width=True):
                        st.session_state.tool_to_remove = i
                        st.rerun()
 
    # ── TAB 3: Test ──────────────────────────────────────────────────────────
    with tab_test:
        if not st.session_state.custom_tools:
            st.markdown(
                '<div class="empty-state" style="height:180px">'
                '<div class="icon">🧪</div>'
                '<p>Add a tool first</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            tool_names = [ct["name"] for ct in st.session_state.custom_tools]
            selected_tool = st.selectbox("Tool", tool_names, key="test_tool_select",
                                         label_visibility="collapsed")
            ct_obj = next((ct for ct in st.session_state.custom_tools if ct["name"] == selected_tool), None)
            if ct_obj:
                st.markdown(
                    f'<div style="font-size:0.77rem;color:#6e7681;margin-bottom:0.5rem">{ct_obj["description"]}</div>',
                    unsafe_allow_html=True,
                )
                test_args: dict = {}
                for pname, pinfo in ct_obj["parameters"].items():
                    req = " *" if pname in ct_obj["required"] else ""
                    val = st.text_input(
                        f"{pname}{req}",
                        placeholder=pinfo.get("description", ""),
                        key=f"test_arg_{selected_tool}_{pname}",
                        help=f"Type: {pinfo.get('type','string')}",
                    )
                    if val:
                        test_args[pname] = val
                if st.button("▶ Run", type="primary", key="run_custom_tool", use_container_width=True):
                    result = dispatch_tool(selected_tool, test_args)
                    st.json(result)
 
# ── process chat ──────────────────────────────────────────────────────────────
final_input = quick_selected or (
    user_input if (send and user_input.strip()) else None
)
 
if final_input:
    if not st.session_state.api_key:
        st.warning("⚠️ Add your OpenAI API key in the sidebar.")
        st.stop()
 
    st.session_state.messages.append({"role": "user", "content": final_input})
 
    # build active schemas
    builtin_schemas = [
        s for s in get_openai_tool_schemas()
        if enabled_tools.get(s["function"]["name"], True)
    ]
    custom_schemas = [
        _custom_tool_to_openai_schema(ct)
        for ct in st.session_state.custom_tools
        if enabled_tools.get(ct["name"], True)
    ]
    active_schemas = builtin_schemas + custom_schemas
 
    # build system prompt
    tool_names_str = ", ".join(
        [info["schema"]["name"] for info in TOOL_REGISTRY.values()]
        + [ct["name"] for ct in st.session_state.custom_tools]
    )
    hints = " ".join(
        ct["system_hint"]
        for ct in st.session_state.custom_tools
        if ct.get("system_hint") and enabled_tools.get(ct["name"], True)
    )
    system_content = (
        f"You are a helpful assistant with access to tools: {tool_names_str}. "
        "Always use the appropriate tool when the user's request matches a tool's purpose. "
        "Present results clearly and concisely."
    )
    if hints:
        system_content += f" Additional instructions: {hints}"
 
    client = OpenAI(api_key=st.session_state.api_key)
 
    with st.spinner("Thinking…"):
        try:
            loop_msgs = [
                {"role": "system", "content": system_content},
                *[{k: v for k, v in m.items() if k != "tool_calls" or v}
                  for m in st.session_state.messages],
            ]
 
            for _ in range(6):
                resp = client.chat.completions.create(
                    model=model,
                    messages=loop_msgs,
                    tools=active_schemas or None,
                    tool_choice="auto" if active_schemas else None,
                    temperature=temperature,
                )
                msg = resp.choices[0].message
 
                if not msg.tool_calls:
                    st.session_state.messages.append({"role": "assistant", "content": msg.content or ""})
                    break
 
                msg_dict = {"role": "assistant", "content": msg.content or ""}
                msg_dict["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
                st.session_state.messages.append(msg_dict)
                loop_msgs.append(msg_dict)
 
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = dispatch_tool(tc.function.name, args)
                    tool_msg = {
                        "role": "tool", "tool_call_id": tc.id,
                        "name": tc.function.name, "content": json.dumps(result),
                    }
                    st.session_state.messages.append(tool_msg)
                    loop_msgs.append(tool_msg)
 
        except Exception as e:
            st.error(f"Error: {e}")
 
    st.rerun()
 