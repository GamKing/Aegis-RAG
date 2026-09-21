"""文本归一化、分词与数值匹配的底层工具集。

设计约束：
- 零第三方依赖（仅标准库），保证评测逻辑可复现、可审计；
- 面向中文语料：关键词召回采用 CJK 二元组（bigram）覆盖法；数值忠实性校验
  采用边界敏感的正则匹配，避免 "1500" 误命中 "15000" 一类的前缀串扰；
- 附带终端显示宽度工具（East Asian Wide/Fullwidth 记 2 列），保证 ASCII
  报表在含中文时仍然对齐。
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

# 全角 -> 半角（数字 / 字母 / 百分号 / 小数点），用于消除表述差异
_FULLWIDTH = (
    "０１２３４５６７８９"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    "％．"
)
_HALFWIDTH = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "%."
)
_FW_TABLE = str.maketrans(dict(zip(_FULLWIDTH, _HALFWIDTH)))

_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------

_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?%?")
_NUMBER_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<pct>%)?")
_MONTH_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_SCALE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<scale>million|billion)\b",
    re.IGNORECASE,
)
_MONTH_NUMBERS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def normalize_faithfulness_text(text: str) -> str:
    """归一化忠实性比较中的日期与英文金额单位。

    - 英文月份统一为中文数字月份（September -> 9月）；
    - million 统一换算为“亿”（X million = X/100 亿）；
    - billion 统一换算为“亿”（X billion = X*10 亿）。

    该函数只做明确的单位/日期等价变换，不把语义短语（如
    ``not assuming``）擅自推断成数字 0。
    """
    normalized = normalize_text(text)

    def month_repl(match: re.Match) -> str:
        month = _MONTH_NUMBERS[match.group("month").lower()]
        return f"{month}月"

    normalized = _MONTH_RE.sub(month_repl, normalized)

    def scale_repl(match: re.Match) -> str:
        value = float(match.group("value"))
        scale = match.group("scale").lower()
        converted = value / 100 if scale == "million" else value * 10
        # 避免 1080.0 这类无意义的小数尾部，同时保留必要精度。
        rendered = f"{converted:.10f}".rstrip("0").rstrip(".")
        return f"{rendered}亿"

    return _SCALE_RE.sub(scale_repl, normalized)


def extract_faithfulness_numbers(text: str) -> list:
    """提取经过日期/金额单位归一化后的忠实性数值。"""
    return extract_numbers(normalize_faithfulness_text(text))

# 停用词：中文高频虚词型 bigram + 英文功能词。刻意保持保守，
# 只收录几乎无区分度的词项，避免误伤政策术语。
DEFAULT_STOPWORDS = frozenset(
    {
        "我们", "你们", "他们", "可以", "以及", "如果", "但是", "对于", "根据",
        "通过", "并且", "或者", "还是", "就是", "这个", "那个", "什么", "怎么",
        "哪些", "是否", "如下",
        "the", "a", "an", "of", "to", "is", "are", "for", "and", "or",
        "in", "on", "at", "by", "with",
    }
)


def normalize_text(text: str) -> str:
    """NFC 归一化 + 全角转半角 + 去掉数字中的千分位逗号（1,500 -> 1500）。"""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_FW_TABLE)
    text = _THOUSANDS_COMMA_RE.sub("", text)
    return text


def normalize_for_match(text: str) -> str:
    """实体/子串匹配用的归一化：全角转半角、去全部空白、忽略大小写。"""
    return "".join(normalize_text(text).split()).lower()


def extract_terms(text: str, stopwords=DEFAULT_STOPWORDS) -> set:
    """抽取召回评测词项：CJK 连续段切 bigram + ASCII 字母词/数字（含 %）。

    bigram 是中文无词典分词下最稳的词项粒度：既不需要 jieba 这类外部依赖，
    又能对关键短语形成覆盖性度量。
    """
    text = normalize_text(text)
    terms: set = set()
    for m in _CJK_RUN_RE.finditer(text):
        run = m.group()
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram not in stopwords:
                terms.add(bigram)
    for m in _ASCII_TOKEN_RE.finditer(text):
        token = m.group().lower()
        if token not in stopwords:
            terms.add(token)
    return terms


def find_uncovered_ground_truth_segments(
    ground_truth: str, covered_terms: set, max_segments: int = 5
):
    """把 ground truth 中未被覆盖的 bigram 还原成人类可读的「缺失片段」。

    规则：CJK 连续段内，某 bigram 未出现在 covered_terms 中，其覆盖的两个字符
    即标记为缺失；相邻缺失字符合并成最大片段。ASCII 词项（数字/英文）单独列出。
    供 ContextRecallEvaluator 生成失败归因信息。
    """
    normalized = normalize_text(ground_truth)
    seg_with_pos: list = []  # [(片段在原文中的起始偏移, 片段), ...]
    ascii_tokens: list = []
    for m in _CJK_RUN_RE.finditer(normalized):
        run = m.group()
        base = m.start()
        missing = [False] * len(run)
        for i in range(len(run) - 1):
            if run[i : i + 2] not in covered_terms:
                missing[i] = True
                missing[i + 1] = True
        i = 0
        while i < len(run):
            if missing[i]:
                j = i
                while j + 1 < len(run) and missing[j + 1]:
                    j += 1
                seg_with_pos.append((base + i, run[i : j + 1]))
                i = j + 1
            else:
                i += 1
    for m in _ASCII_TOKEN_RE.finditer(normalized):
        token = m.group().lower()
        if token not in covered_terms:
            ascii_tokens.append(token)
    # 截断时优先保留最长的缺失片段（信息量最大），展示时再按原文顺序排列
    selected = sorted(seg_with_pos, key=lambda p: (-len(p[1]), p[0]))
    selected = selected[:max_segments]
    selected.sort(key=lambda p: p[0])
    cjk_segments = [s for _, s in selected]
    ascii_tokens = list(dict.fromkeys(ascii_tokens))
    return cjk_segments, ascii_tokens


# ---------------------------------------------------------------------------
# 数值忠实性
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> list:
    """提取文本中的数值单元，返回去重保序的 [(数值字符串, 是否百分比), ...]。

    约定："70%" -> ("70", True)；"1500元" -> ("1500", False)；
    "3.5万" -> ("3.5", False)。中文数字（如 "百分之七十"）v1 暂不支持。
    """
    normalized = normalize_text(text)
    seen: list = []
    for m in _NUMBER_RE.finditer(normalized):
        pair = (m.group("value"), m.group("pct") is not None)
        if pair not in seen:
            seen.append(pair)
    return seen


def number_supported_in_context(value: str, is_percent: bool, context: str) -> bool:
    """边界敏感地判断数值是否在上下文中出现。

    严格规则：
    1. 数字边界："1500" 不得命中 "15000" / "31500" 中的子串；
    2. 单位一致：答案中的百分数必须以上下文中的百分数形式出现
       （防止 "85%" 被 "85元" 洗白）；普通数值允许命中普通数值或百分数。
    """
    normalized = normalize_text(context)
    esc = re.escape(value)
    if is_percent:
        pattern = rf"(?<![\d.]){esc}(?![\d.])\s*%"
    else:
        pattern = rf"(?<![\d.]){esc}(?![\d.])"
    return re.search(pattern, normalized) is not None


# ---------------------------------------------------------------------------
# 终端显示宽度（CJK 对齐）
# ---------------------------------------------------------------------------

def _char_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def display_width(text: str) -> int:
    """终端显示宽度：East Asian Wide/Fullwidth 记 2，其余记 1。"""
    return sum(_char_width(ch) for ch in text)


def pad_display(text: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格，支持 left/right/center。"""
    gap = max(width - display_width(text), 0)
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def truncate_display(text: str, width: int, suffix: str = "...") -> str:
    """按显示宽度截断并追加省略后缀，不会把一个宽字符劈成两半。"""
    if display_width(text) <= width:
        return text
    budget = max(width - display_width(suffix), 0)
    out: list = []
    acc = 0
    for ch in text:
        w = _char_width(ch)
        if acc + w > budget:
            break
        out.append(ch)
        acc += w
    return "".join(out) + suffix
