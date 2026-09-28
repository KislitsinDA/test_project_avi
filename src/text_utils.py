import re
from typing import Optional

_WS = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")

# Длинные окончания раньше коротких. Стем короткий, без словаря — чтобы «бани» и «баня» сходились.
_ENDINGS = (
    "остями",
    "ениями",
    "аниями",
    "ость",
    "ение",
    "ание",
    "ами",
    "ями",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ий",
    "ый",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ия",
    "ам",
    "ям",
    "ом",
    "ем",
    "ую",
    "юю",
    "а",
    "я",
    "ы",
    "и",
    "у",
    "ю",
    "е",
    "о",
)

# Служебные слова из фильтров вроде «вид услуги» / «рейтинг 4 звезды» — в поиске только шумят.
_STOP = {
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "для",
    "от",
    "до",
    "или",
    "не",
    "за",
    "из",
    "к",
    "ко",
    "о",
    "об",
    "а",
    "но",
    "это",
    "как",
    "что",
    "вид",
    "услуги",
    "услуга",
    "рейтинг",
    "пользователя",
    "звезды",
    "звезда",
    "выше",
    "ниже",
}


def norm_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("ё", "е")
    text = _WS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def stem_token(token: str) -> str:
    if len(token) < 4:
        return token
    for ending in _ENDINGS:
        stem_len = len(token) - len(ending)
        if stem_len >= 3 and token.endswith(ending):
            return token[:stem_len]
    return token


def tokenize(value: str, max_tokens: int = 0) -> list[str]:
    text = norm_text(value)
    if not text:
        return []
    toks: list[str] = []
    for raw in text.split():
        if raw in _STOP or raw.isdigit():
            continue
        tok = stem_token(raw)
        if len(tok) < 2 or tok in _STOP:
            continue
        toks.append(tok)
        if max_tokens and len(toks) >= max_tokens:
            break
    return toks


def build_query_text(row) -> str:
    parts = [row.get("search_query", ""), row.get("search_infm_params_text", "")]
    return norm_text(" ".join(str(p) for p in parts if p))


def e5_query_text(row) -> str:
    """Префикс query: обязателен для multilingual-e5."""
    q = str(row.get("search_query") or "").strip()
    extra = str(row.get("search_infm_params_text") or "").strip()
    if extra:
        q = f"{q} {extra}"
    q = norm_text(q)
    return f"query: {q}" if q else "query:"


def e5_passage_text(title: str, desc: str, params: str, desc_chars: int) -> str:
    t = str(title or "").strip()
    d = str(desc or "")[:desc_chars].strip()
    p = str(params or "").strip()
    body = norm_text(f"{t} {p} {d}")
    return f"passage: {body}" if body else "passage:"


def item_search_text(title: str, desc: str, params: str, desc_chars: int) -> str:
    """Заголовок дважды — короткие запросы чаще совпадают с title, не с простынёй описания."""
    t = str(title or "")
    d = str(desc or "")[:desc_chars]
    p = str(params or "")
    return norm_text(f"{t} {t} {p} {d}")


def min_rating_from_params(params_text: Optional[str]) -> Optional[float]:
    """Фильтр вида «Рейтинг пользователя 4 звезды и выше»."""
    if not params_text:
        return None
    text = str(params_text).lower().replace("ё", "е")
    m = re.search(r"рейтинг[^\d]*(\d)\s*зв", text)
    if m:
        return float(m.group(1))
    return None
