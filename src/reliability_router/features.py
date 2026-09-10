"""Transparent, deterministic feature extraction for routing risk."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from .models import ContextItem, FeatureVector, RouterRequest

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]") # ASCII words and single CJK characters
_TOOL_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NEGATION_RE = re.compile(r"\b(?:not|never|no|isn't|aren't|cannot|can't|won't)\b|不|没|无|并非|不是", re.I) # negative word detector, re.I ignore case


class FeatureExtractor:
    """Extract bounded [0, 1] features without an LLM call.

    The implementation intentionally favors inspectability over perfect semantic
    coverage.
    """

    HIGH_STAKES_PATTERNS = {
        "medical": re.compile(
            r"\b(?:diagnos|symptom|dosage|dose|medication|chest pain|emergency|pregnan|suicid|self-harm|therapy)\w*\b"
            r"|诊断|症状|剂量|用药|胸痛|急救|怀孕|自杀|自残|心理治疗|医疗建议",
            re.I,
        ),
        "legal": re.compile(
            r"\b(?:legal advice|lawsuit|sue|contract law|liable|liability|immigration|visa|tax law|compliance)\b"
            r"|法律建议|诉讼|起诉|合同法|责任|移民|签证|税法|合规",
            re.I,
        ),
        "financial": re.compile(
            r"\b(?:invest|stock|bond|crypto|portfolio|mortgage|loan|leverage|tax return|retirement)\w*\b"
            r"|投资|股票|债券|加密货币|资产配置|房贷|贷款|杠杆|报税|退休金",
            re.I,
        ),
        "security": re.compile(
            r"\b(?:production credential|private key|security incident|breach|exploit|malware|ransomware)\b"
            r"|生产凭证|私钥|安全事件|数据泄露|漏洞利用|恶意软件|勒索软件",
            re.I,
        ),
    }

    TRANSFORM_RE = re.compile(
        r"^\s*(?:rewrite|rephrase|paraphrase|translate|summarize|proofread|改写|翻译|总结|润色)", re.I
    )
    FRESHNESS_RE = re.compile(
        r"\b(?:today|tonight|tomorrow|yesterday|latest|newest|current|currently|right now|live|price|weather|forecast|schedule|score|standings|president|prime minister|ceo|release|version)\b"
        r"|今天|今晚|明天|昨天|最新|当前|现在|实时|价格|天气|预报|赛程|比分|排名|总统|总理|首相|CEO|发布|版本",
        re.I,
    )
    CITATION_RE = re.compile(
        r"\b(?:cite|citation|source|reference|evidence|verify|fact-check|quote exactly|link)\w*\b"
        r"|引用|引文|来源|参考资料|证据|核实|查证|事实核查|原文|链接",
        re.I,
    )
    MULTI_STEP_RE = re.compile(
        r"\b(?:analy[sz]e|compare|contrast|design|architect|plan|evaluate|debug|investigate|trade-?off|root cause|migration|strategy|experiment|benchmark|derive)\w*\b"
        r"|分析|比较|对比|设计|架构|规划|评估|调试|调查|权衡|根因|迁移|策略|实验|基准|推导",
        re.I,
    )
    NUMERICAL_RE = re.compile(
        r"\b(?:calculate|compute|estimate|probability|percentage|percent|forecast|budget|roi|p\d{2}|latency)\b"
        r"|计算|估算|概率|百分比|比例|预测|预算|回报率|延迟|\d+\s*[%×x*/+-]\s*\d+",
        re.I,
    )
    USER_PRESSURE_RE = re.compile(
        r"\b(?:just agree|agree with me|confirm (?:that )?i(?:'m| am) right|say yes|do not disagree|don't disagree|no caveat|without caveats|validate my belief)\b"
        r"|只要同意|同意我|确认我是对的|只说是|不要反驳|别质疑|不要保留意见|证明我的观点",
        re.I,
    )
    INJECTION_RE = re.compile(
        r"\b(?:ignore (?:all |the )?(?:previous|prior|above) instructions|system prompt|developer message|reveal (?:the )?(?:secret|password|key)|exfiltrate|jailbreak|hidden instruction)\b"
        r"|忽略(?:之前|以上|前面)(?:所有)?指令|系统提示词|开发者消息|泄露(?:秘密|密码|密钥)|越狱|隐藏指令",
        re.I,
    )
    AMBIGUOUS_RE = re.compile(
        r"^(?:what about (?:it|that)|do it|fix it|thoughts|help|why|this|that)[?!]?$"
        r"|^(?:这个呢|那个呢|怎么办|帮我|为什么|看法|处理一下|修一下)[？?]?$",
        re.I,
    )

    # Matched against name tokens, so "web_search"/"WebSearch"/"google search" all resolve to web.
    TOOL_ALIASES = {
        "web": {"web", "browser", "search", "internet", "google", "bing", "duckduckgo", "brave",
                "serp", "url", "fetch", "crawl", "news"},
        "weather": {"weather", "forecast", "meteo"},
        "finance": {"finance", "market", "markets", "stock", "stocks", "ticker", "quote", "price", "prices"},
        "calculator": {"calculator", "calc", "python", "compute", "code", "interpreter", "repl",
                       "math", "wolfram", "sympy"},
        "retrieval": {"retrieval", "retrieve", "rag", "search", "database", "db", "files", "file",
                      "vector", "vectordb", "index", "knowledge", "kb", "docs", "lookup", "embeddings"},
    }

    def extract(self, request: RouterRequest) -> FeatureVector:
        query = request.query.strip()
        token_estimate = max(1, math.ceil(len(query) / 4))

        high_stakes_hits = sum(bool(pattern.search(query)) for pattern in self.HIGH_STAKES_PATTERNS.values())
        high_stakes = min(1.0, 0.72 * high_stakes_hits) if high_stakes_hits else 0.0

        # if query is a rewriting request, only grab the content before colon ":"
        freshness_scope = query.split(":", 1)[0] if self.TRANSFORM_RE.search(query) else query
        freshness = 0.85 if self.FRESHNESS_RE.search(freshness_scope) else 0.0
        if re.search(r"\b20(?:2[5-9]|3\d)\b|202[5-9]年", query):
            freshness = max(freshness, 0.55)

        citation_need = 0.9 if self.CITATION_RE.search(query) else 0.0

        multi_hits = len(self.MULTI_STEP_RE.findall(query))
        multi_step = min(1.0, multi_hits * 0.35)
        if token_estimate >= 80:
            multi_step = max(multi_step, 0.55)
        elif token_estimate >= 40:
            multi_step = max(multi_step, 0.35)
        if query.count("?") + query.count("？") >= 3:
            multi_step = max(multi_step, 0.55)

        numerical = 0.72 if self.NUMERICAL_RE.search(query) else 0.0
        user_pressure = 0.9 if self.USER_PRESSURE_RE.search(query) else 0.0
        prompt_injection = self._prompt_injection_score(query, request.context)
        context_conflict = self._context_conflict_score(request.context)
        context_load = min(1.0, sum(len(item.text) for item in request.context) / 12_000)
        ambiguity = self._ambiguity_score(query, token_estimate)

        tools = self._normalize_tools(request.available_tools)
        # figure out the gap between needs and capabilities
        # same query → tool_gap=1.00 (no tools) vs 0.15 (with web).
        tool_gap = self._tool_gap(
            freshness=freshness,
            numerical=numerical,
            citation_need=citation_need,
            tools=tools,
        )
        # evidence_gap = 0.92 (bare) → 0.28 (web tool) → 0.07 (trusted context)
        evidence_gap = self._evidence_gap(
            context=request.context,
            high_stakes=high_stakes,
            freshness=freshness,
            citation_need=citation_need,
            tools=tools,
        )

        # 12 features
        return FeatureVector(
            token_estimate=token_estimate,
            ambiguity=round(ambiguity, 4),
            high_stakes=round(high_stakes, 4),
            freshness=round(freshness, 4),
            citation_need=round(citation_need, 4),
            multi_step=round(multi_step, 4),
            numerical=round(numerical, 4),
            user_pressure=round(user_pressure, 4),
            context_conflict=round(context_conflict, 4),
            tool_gap=round(tool_gap, 4),
            evidence_gap=round(evidence_gap, 4),
            context_load=round(context_load, 4),
            prompt_injection=round(prompt_injection, 4),
        )

    @staticmethod
    def _normalize_tools(available_tools: Iterable[str]) -> set[str]:
        """Split caller-supplied tool names into tokens so aliases match by name part.

        Callers name tools freely ("web_search", "WebSearch", "code_interpreter"), so an
        exact-name comparison would read every realistic name as "no tool available".
        """

        tokens: set[str] = set()
        for tool in available_tools:
            spaced = _CAMEL_BOUNDARY_RE.sub(" ", tool)
            tokens.update(_TOOL_TOKEN_RE.findall(spaced.lower()))
        return tokens

    def _ambiguity_score(self, query: str, token_estimate: int) -> float:
        if self.AMBIGUOUS_RE.fullmatch(query.strip()):
            return 0.95
        score = 0.0
        if token_estimate <= 3:
            score = 0.55
        elif token_estimate <= 6 and re.search(
            r"\b(?:it|this|that|they|he|she)\b|这个|那个|它|他们", query, re.I
        ):
            score = 0.65
        if re.search(r"\b(?:something|somehow|etc\.?|whatever)\b|之类|随便|某个", query, re.I):
            score = max(score, 0.45)
        return score

    def _prompt_injection_score(self, query: str, context: list[ContextItem]) -> float:
        score = 0.95 if self.INJECTION_RE.search(query) else 0.0
        for item in context:
            if self.INJECTION_RE.search(item.text):
                score = max(score, 0.95 if item.trust < 0.8 else 0.65)
        return score

    def _tool_gap(
        self,
        *,
        freshness: float,
        numerical: float,
        citation_need: float,
        tools: set[str],
    ) -> float:
        if freshness >= 0.7:
            required = self.TOOL_ALIASES["web"] | self.TOOL_ALIASES["weather"] | self.TOOL_ALIASES["finance"]
            return 0.15 if tools & required else 1.0
        if citation_need >= 0.7:
            required = self.TOOL_ALIASES["web"] | self.TOOL_ALIASES["retrieval"]
            return 0.2 if tools & required else 0.82
        if numerical >= 0.7:
            return 0.1 if tools & self.TOOL_ALIASES["calculator"] else 0.38
        return 0.0

    def _evidence_gap(
        self,
        *,
        context: list[ContextItem],
        high_stakes: float,
        freshness: float,
        citation_need: float,
        tools: set[str],
    ) -> float:
        need = max(high_stakes, freshness, citation_need)
        if need < 0.5:
            return 0.0
        if context:
            trusted = max((item.trust for item in context), default=0.0)
            return max(0.05, 0.5 - 0.45 * trusted)
        has_retrieval = bool(tools & (self.TOOL_ALIASES["web"] | self.TOOL_ALIASES["retrieval"]))
        return 0.28 if has_retrieval else 0.92

    def _context_conflict_score(self, context: list[ContextItem]) -> float:
        if len(context) < 2:
            return 0.0
        fingerprints: list[tuple[set[str], bool]] = []
        for item in context:
            words = {
                w.lower() for w in _WORD_RE.findall(item.text) if len(w) > 1 or "\u4e00" <= w <= "\u9fff"
            }
            stop = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "this", "that"}
            words -= stop
            fingerprints.append((words, bool(_NEGATION_RE.search(item.text))))

        best = 0.0
        for i, (left, left_neg) in enumerate(fingerprints):
            for right, right_neg in fingerprints[i + 1 :]:
                if not left or not right or left_neg == right_neg:
                    continue
                overlap = len(left & right) / max(1, len(left | right))
                if overlap >= 0.35:
                    best = max(best, min(1.0, 0.55 + overlap))
        return best


def contains_any(text: str, terms: Iterable[str]) -> bool:
    """Small utility kept public for custom feature plugins."""

    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)
