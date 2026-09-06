#!/usr/bin/env python3
"""Patch sglang's deepseekv32_detector.py (also serving DeepSeek V4 via the
DeepSeekV4Detector subclass) to tolerate the OpenAI-function-object body shape.

Root cause: the dsv4 TOOLS_TEMPLATE only demonstrates DSML XML parameter tags
(Format 1), but the detector also accepts direct-JSON invoke bodies (Format 2)
without ever showing the model what one looks like. Once tool history exists
in context, the model freelances Format-2 bodies in the OpenAI `function`
object shape it knows from pretraining:

    <DSML>invoke name="bash">{"arguments": {"command": "..."}}</DSML>invoke>

i.e. the parameter object nested one level too deep (name already carried by
the invoke tag). The detector passes the body through verbatim, so clients
receive arguments='{"arguments": {...}}' and fail schema validation
("Missing key ..."). Reproduced 4/4 on turn-2-style requests with tool
history (0/3 on fresh turn-1 requests).

Fix: schema-aware unwrap of function-shaped bodies ({"arguments": ...} or
{"name": ..., "arguments": ...}) to the inner arguments — unless the tool
genuinely declares an "arguments" property. Applied to complete bodies
(non-streaming + streaming completion) and, as stream suppression, to partial
bodies still unfolding into the wrapper (argument deltas can only extend, so a
leaked '{"arguments":' prefix could never be retracted). Everything else
(bare-params bodies, XML params, unknown shapes) keeps legacy behavior.
"""
import sys

PATH = "/sgl-workspace/sglang/python/sglang/srt/function_call/deepseekv32_detector.py"

with open(PATH) as f:
    src = f.read()

orig = src

HELPERS = '''def _dsv4_tool_param_names(tools, func_name):
    """Parameter-name set for func_name from OpenAI Tool objects (or plain
    dicts); None when the tool is unknown."""
    try:
        for t in tools or []:
            fn = getattr(getattr(t, "function", None), "name", None)
            params = getattr(getattr(t, "function", None), "parameters", None)
            if fn is None and isinstance(t, dict):
                f = t.get("function", t)
                if isinstance(f, dict):
                    fn = f.get("name")
                    params = f.get("parameters")
            if fn != func_name:
                continue
            if isinstance(params, dict):
                props = params.get("properties")
                if isinstance(props, dict):
                    return set(props.keys())
            return set()
    except Exception:
        pass
    return None


def _dsv4_unwrap_function_shape(params, tools, func_name):
    """Unwrap an OpenAI-function-object shaped params dict to the inner
    arguments — unless the tool genuinely declares that property. Fail-open:
    returns input on any doubt. Mirrors upstream vLLM's _unwrap_wrapper_args
    (wrappers "arguments"/"input", inner-keys-subset rule), plus the
    {"name", ...} two-key variant seen in the wild.
    """
    if not isinstance(params, dict):
        return params
    keys = set(params.keys())
    wrapper = None
    if len(keys) == 1 and next(iter(keys)) in ("arguments", "input"):
        wrapper = next(iter(keys))
    elif keys == {"name", "arguments"}:
        wrapper = "arguments"
    elif keys == {"name", "input"}:
        wrapper = "input"
    if wrapper is None:
        return params
    declared = _dsv4_tool_param_names(tools, func_name)
    if declared is None or wrapper in declared:
        return params
    inner = params[wrapper]
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except Exception:
            return params
    if isinstance(inner, dict) and set(inner.keys()).issubset(declared):
        return inner
    return params


def _dsv4_normalize_json_body(body, tools, func_name):
    """Normalize a complete direct-JSON invoke body; fail-open to input."""
    try:
        parsed = json.loads(body)
    except Exception:
        return body
    unwrapped = _dsv4_unwrap_function_shape(parsed, tools, func_name)
    try:
        return json.dumps(unwrapped, ensure_ascii=False)
    except Exception:
        return body


def _dsv4_suppress_wrapped_partial(stripped, tools, func_name):
    """True when a partial JSON body is (so far) only an unfolding
    function-object wrapper for a tool with no "arguments" property: the
    caller should stream nothing yet and wait for the completed block.
    Bare-param partials (any key outside name/arguments) stream as before."""
    try:
        parsed, _ = _partial_json_loads(stripped, Allow.ALL)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    if any(k not in ("name", "arguments", "input") for k in parsed):
        return False
    if not parsed:
        # Bare '{' etc: hold back one chunk so a stray '{}' prefix cannot
        # poison the arg stream; the shape declares itself on the next chunk.
        return True
    if "arguments" not in parsed and "input" not in parsed:
        return False
    declared = _dsv4_tool_param_names(tools, func_name)
    if declared is None:
        return False
    return "arguments" not in declared and "input" not in declared


def _dsv4_is_wrapped_shape(params, tools, func_name):
    """True when params is exactly a copied wrapper object (single
    "arguments"/"input" key, or name+wrapper) for a tool declaring no such
    property. Used for the XML branch (and any decoded-dict path). Unknown
    schema -> False."""
    if not isinstance(params, dict) or not params:
        return False
    keys = set(params.keys())
    wrapper = None
    if len(keys) == 1 and next(iter(keys)) in ("arguments", "input"):
        wrapper = next(iter(keys))
    elif keys in ({"name", "arguments"}, {"name", "input"}):
        wrapper = "arguments" if "arguments" in keys else "input"
    if wrapper is None:
        return False
    declared = _dsv4_tool_param_names(tools, func_name)
    if declared is None:
        return False
    return wrapper not in declared


'''

ANCHOR_CLASS = "logger = logging.getLogger(__name__)\n\n\nclass DeepSeekV32Detector(BaseFormatDetector):"
if "_dsv4_unwrap_function_shape" not in src:
    if ANCHOR_CLASS not in src:
        print("[dsv4-fnshape] ERROR: class anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(
        ANCHOR_CLASS,
        "logger = logging.getLogger(__name__)\n\n\n" + HELPERS + "class DeepSeekV32Detector(BaseFormatDetector):",
        1,
    )
else:
    # Upgrade path: replace a previous helper block wholesale (helpers run
    # from the marker to the class line).
    import re

    Pat = re.compile(
        r"def _dsv4_tool_param_names.*?^class DeepSeekV32Detector\(BaseFormatDetector\):",
        re.DOTALL | re.MULTILINE,
    )
    if Pat.search(src) is None:
        print("[dsv4-fnshape] ERROR: helper upgrade anchor not found", file=sys.stderr)
        sys.exit(1)
    src = Pat.sub(HELPERS + "class DeepSeekV32Detector(BaseFormatDetector):", src, count=1)

OLD_SIG = """    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:"""
NEW_SIG = """    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False,
        tools=None, func_name=None
    ) -> str:"""
if NEW_SIG not in src:
    if OLD_SIG not in src:
        print("[dsv4-fnshape] ERROR: _parse signature anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_SIG, NEW_SIG, 1)

OLD_JSON = """        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                # Remove incomplete invoke end call prefix in case they are captured by param
                for token in reversed(self.prefix_invoke_end_call):
                    invoke_content_stripped = invoke_content_stripped.rstrip(token)
                return invoke_content_stripped
            elif invoke_content_stripped.endswith("}"):
                return invoke_content_stripped"""
NEW_JSON = """        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                # Remove incomplete invoke end call prefix in case they are captured by param
                for token in reversed(self.prefix_invoke_end_call):
                    invoke_content_stripped = invoke_content_stripped.rstrip(token)
                if _dsv4_suppress_wrapped_partial(
                    invoke_content_stripped, tools, func_name
                ):
                    return ""
                return invoke_content_stripped
            elif invoke_content_stripped.endswith("}"):
                return _dsv4_normalize_json_body(
                    invoke_content_stripped, tools, func_name
                )"""
if NEW_JSON not in src:
    if OLD_JSON not in src:
        print("[dsv4-fnshape] ERROR: JSON branch anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_JSON, NEW_JSON, 1)

# XML branch: a param literally named "arguments" (copied wrapper shape —
# the model learns it from history renderings) must not stream decoded.
OLD_XML_RET = "        return json.dumps(parameters, ensure_ascii=False)"
NEW_XML_RET = """        if _dsv4_is_wrapped_shape(parameters, tools, func_name):
            # Copied wrapper shape: hold back partials (deltas cannot
            # retract); unwrap the inner arguments when complete.
            if allow_partial:
                return ""
            parameters = _dsv4_unwrap_function_shape(parameters, tools, func_name)

        return json.dumps(parameters, ensure_ascii=False)"""
if NEW_XML_RET not in src:
    if OLD_XML_RET not in src:
        print("[dsv4-fnshape] ERROR: XML return anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_XML_RET, NEW_XML_RET, 1)

OLD_DETECT = "                    func_args = self._parse_parameters_from_xml(invoke_content)"
NEW_DETECT = """                    func_args = self._parse_parameters_from_xml(
                        invoke_content, tools=tools, func_name=func_name
                    )"""
if NEW_DETECT not in src:
    if OLD_DETECT not in src:
        print("[dsv4-fnshape] ERROR: detect call-site anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_DETECT, NEW_DETECT, 1)

OLD_MATCH = """                    match_result = {
                        "name": func_name,
                        "parameters": json.loads(func_args),
                    }"""
NEW_MATCH = """                    match_result = {
                        "name": func_name,
                        "parameters": json.loads(func_args),
                    }
                    # Tolerate OpenAI-function-shaped bodies (XML branch);
                    # the JSON branch is already normalized inside _parse.
                    match_result["parameters"] = _dsv4_unwrap_function_shape(
                        match_result["parameters"], tools, func_name
                    )"""
if NEW_MATCH not in src:
    if OLD_MATCH not in src:
        print("[dsv4-fnshape] ERROR: match_result anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_MATCH, NEW_MATCH, 1)

OLD_STREAM = """                current_params = self._parse_parameters_from_xml(
                    invoke_content, allow_partial=not is_tool_end
                )"""
NEW_STREAM = """                current_params = self._parse_parameters_from_xml(
                    invoke_content, allow_partial=not is_tool_end,
                    tools=tools, func_name=func_name
                )"""
if NEW_STREAM not in src:
    if OLD_STREAM not in src:
        print("[dsv4-fnshape] ERROR: streaming call-site anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_STREAM, NEW_STREAM, 1)

# finish(): flush an unclosed invoke left in the buffer at stream end.
# When the model omits </invoke> (or the EOT), the name already streamed but
# the body is stuck buffered; the base finish() drops it, leaving clients
# with a name-only call (empty arguments — the "{}" plague). Re-parse the
# tail through the complete (normalized) path and emit the MISSING suffix
# only: guarded by prefix-match so already-streamed bytes are never
# duplicated, and silent otherwise (today's behavior preserved).
OLD_FINISH_ANCHOR = "    def structure_info(self) -> _GetInfoFunc:"
NEW_FINISH = '''    def finish(self, tools) -> StreamingParseResult:
        """Flush a tail invoke left unclosed at stream end (see mod note)."""
        try:
            buf = self._buffer or ""
            self._buffer = ""
            if not buf:
                return StreamingParseResult()
            idx = buf.find(self.bot_token)
            if idx == -1:
                j = buf.find("invoke")
                if j == -1:
                    return StreamingParseResult()
                idx = buf.rfind("<", 0, j)
                if idx == -1:
                    idx = j
            inner = buf[idx:]
            if inner.startswith(self.bot_token):
                inner = inner[len(self.bot_token):]
            # Drop any trailing closers already present, then synthesize a
            # complete block from the detector's own tokens so the section
            # regex (which requires the EOT) can match.
            cut = len(inner)
            for marker in (self.eot_token, self.invoke_end_token):
                if marker:
                    pos = inner.find(marker)
                    if pos != -1:
                        cut = min(cut, pos)
            inner = inner[:cut]
            if "invoke" not in inner:
                return StreamingParseResult()
            synth = (
                self.bot_token + "\\n" + inner + "\\n"
                + self.invoke_end_token + "\\n" + self.eot_token
            )
            res = self.detect_and_parse(synth, tools)
            if len(res.calls) != 1:
                return StreamingParseResult()
            if getattr(self, "current_tool_id", -1) == -1:
                return StreamingParseResult()
            c = res.calls[0]
            try:
                full = c.parameters or ""
                # Never flush an empty-args call: an unclosed invoke with no
                # parsable body is exactly the "{}" plague (a genuine zero-arg
                # call completes mid-stream via self-closing/closer tags and
                # never reaches here).
                if full in ("", "{}"):
                    return StreamingParseResult()
                sent = ""
                _streamed = getattr(self, "streamed_args_for_tool", [])
                if 0 <= self.current_tool_id < len(_streamed):
                    sent = _streamed[self.current_tool_id] or ""
                if not full or not full.startswith(sent):
                    return StreamingParseResult()
                tail = full[len(sent):]
                if not tail:
                    return StreamingParseResult()
                return StreamingParseResult(calls=[ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=tail)])
            except Exception:
                return StreamingParseResult()
        except Exception:
            return StreamingParseResult()

    def structure_info(self) -> _GetInfoFunc:'''
if NEW_FINISH not in src:
    if OLD_FINISH_ANCHOR not in src:
        print("[dsv4-fnshape] ERROR: structure_info anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_FINISH_ANCHOR, NEW_FINISH, 1)

if src == orig:
    print("[dsv4-fnshape] no changes needed (already patched)")
else:
    with open(PATH, "w") as f:
        f.write(src)
    print("[dsv4-fnshape] patched deepseekv32_detector.py")
