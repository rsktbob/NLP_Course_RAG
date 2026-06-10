from __future__ import annotations

import re
from collections import Counter


_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_:+./%-]*")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_SPACE_RE = re.compile(r"\s+")
_STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "is",
    "of",
    "the",
    "to",
    "what",
    "why",
    "一個",
    "什麼",
    "何為",
    "可以",
    "如何",
    "怎麼",
    "問題",
    "解決",
    "想解",
    "想解決",
    "決什",
    "為什",
    "為什麼",
    "甚麼",
    "要做",
    "說明",
    "請問",
    "麼是",
    "是什",
    "是什麼",
}


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u3000", " ")
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def _cjk_ngrams(run: str) -> list[str]:
    terms: list[str] = []
    if len(run) <= 2:
        terms.append(run)
        terms.extend(run)
        return terms

    terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    terms.extend(run[i : i + 3] for i in range(len(run) - 2))
    return terms


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English course text for lightweight BM25."""
    normalized = normalize_text(text).lower()
    terms: list[str] = []
    for token in _LATIN_TOKEN_RE.findall(normalized):
        if len(token) > 1 and token not in _STOP_TERMS:
            terms.append(token)
        for part in re.split(r"[-_/:+.]+", token):
            if len(part) > 1 and part not in _STOP_TERMS:
                terms.append(part)
    for run in _CJK_RUN_RE.findall(normalized):
        terms.extend(term for term in _cjk_ngrams(run) if term not in _STOP_TERMS)
    return terms


def token_counts(text: str) -> dict[str, int]:
    return dict(Counter(tokenize(text)))


_COMMON_SIMPLIFIED_TO_TRADITIONAL = str.maketrans(
    {
        "两": "兩",
        "个": "個",
        "为": "為",
        "与": "與",
        "这": "這",
        "应": "應",
        "学": "學",
        "习": "習",
        "轻": "輕",
        "实": "實",
        "体": "體",
        "类": "類",
        "据": "據",
        "无": "無",
        "论": "論",
        "过": "過",
        "国": "國",
        "组": "組",
        "间": "間",
        "关": "關",
        "统": "統",
        "数": "數",
        "库": "庫",
        "门": "門",
        "较": "較",
        "将": "將",
        "词": "詞",
        "汇": "彙",
        "维": "維",
        "码": "碼",
        "测": "測",
        "试": "試",
        "资": "資",
        "质": "質",
        "构": "構",
        "语": "語",
        "义": "義",
        "时": "時",
        "围": "圍",
        "范": "範",
        "问": "問",
        "题": "題",
        "处": "處",
        "转": "轉",
        "换": "換",
    }
)


def normalize_traditional(text: str) -> str:
    """Best-effort cleanup for occasional simplified Chinese from local LLMs."""
    return text.translate(_COMMON_SIMPLIFIED_TO_TRADITIONAL)


def short_title(text: str, fallback: str, limit: int = 80) -> str:
    text = normalize_text(text)
    if not text:
        return fallback
    title = text.split("●", 1)[0].strip() or text
    return title[:limit]
