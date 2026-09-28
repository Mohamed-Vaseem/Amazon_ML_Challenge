"""Text normalization helpers for business entity resolution.

The canonical normalized text intentionally preserves non-Latin scripts. The
dataset can contain business names and addresses in multiple scripts, so
destructive ASCII-only normalization would throw away useful signal. Separate
ASCII-folded helper fields are provided for blocking and fuzzy matching.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

LEGAL_SUFFIXES = {
    "ag",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "llc",
    "llc",
    "llp",
    "ltd",
    "limited",
    "lp",
    "pte",
    "pty",
    "pvt",
    "private",
    "plc",
    "sa",
    "sas",
    "sarl",
}

LEGAL_SUFFIX_REPLACEMENTS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
}

ADDRESS_REPLACEMENTS = {
    "addr": "address",
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
    "apt": "apartment",
    "apmt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "blk": "block",
    "ctr": "center",
    "ctre": "center",
    "dist": "district",
    "dt": "district",
    "nr": "near",
    "po": "post office",
    "rm": "room",
    "no": "number",
    "num": "number",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
}

STOP_TOKENS = {
    "the",
    "and",
    "of",
    "for",
    "near",
    "opp",
    "opposite",
    "nearby",
    "at",
    "in",
    "to",
    "by",
    "from",
    "a",
    "an",
}

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s,;]+|\b[A-Z0-9-]+\.(?:com|org|net|in|co|io|biz|info)\b",
    flags=re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
HOUSE_NUMBER_RE = re.compile(r"\b(?:no|number|#)?\s*([a-z]?\d+[a-z]?(?:[-/]\d+[a-z]?)?)\b")
US_STATE_RE = re.compile(r"\b(?:al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy|dc)\b")


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_text(value: object) -> str:
    """Return lowercase, punctuation-light text while preserving Unicode words."""
    if value is None:
        return ""
    text = str(value).strip().casefold()
    if not text or text == "nan":
        return ""
    text = text.replace("&", " and ")
    text = text.replace("_", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def ascii_fold(value: object) -> str:
    """Return a lowercase ASCII approximation for blocking/fuzzy features."""
    text = normalize_text(value)
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    folded = SPACE_RE.sub(" ", folded)
    return folded.strip()


def tokenize(value: object) -> list[str]:
    """Tokenize normalized text into Unicode word tokens."""
    text = normalize_text(value)
    if not text:
        return []
    return TOKEN_RE.findall(text)


def normalize_business_name(value: object) -> str:
    """Normalize a business name but keep common legal suffix meanings."""
    tokens = tokenize(value)
    normalized = [LEGAL_SUFFIX_REPLACEMENTS.get(token, token) for token in tokens]
    return " ".join(normalized)


def core_business_name(value: object) -> str:
    """Normalize a business name and remove legal suffixes for matching."""
    tokens = normalize_business_name(value).split()
    core = [token for token in tokens if token not in LEGAL_SUFFIXES]
    return " ".join(core)


def sorted_token_key(tokens: Iterable[str]) -> str:
    """Return a deterministic token-set key for order-insensitive matching."""
    return pipe_join(sorted(_stable_unique(tokens)))


def prefix_key(tokens: Iterable[str], *, max_tokens: int = 3, width: int = 4) -> str:
    """Return compact prefixes from important tokens for cheap blocking."""
    prefixes = [token[:width] for token in tokens if token]
    return pipe_join(prefixes[:max_tokens])


def soundex_token(token: str) -> str:
    """Small dependency-free Soundex implementation for Latin business names."""
    folded = ascii_fold(token)
    if not folded:
        return ""
    letters = re.sub(r"[^a-z]", "", folded)
    if not letters:
        return ""

    first = letters[0].upper()
    groups = {
        "b": "1",
        "f": "1",
        "p": "1",
        "v": "1",
        "c": "2",
        "g": "2",
        "j": "2",
        "k": "2",
        "q": "2",
        "s": "2",
        "x": "2",
        "z": "2",
        "d": "3",
        "t": "3",
        "l": "4",
        "m": "5",
        "n": "5",
        "r": "6",
    }
    codes = [groups.get(char, "0") for char in letters]
    deduped = [codes[0]]
    for code in codes[1:]:
        if code != deduped[-1]:
            deduped.append(code)
    tail = [code for code in deduped[1:] if code != "0"]
    return (first + "".join(tail) + "000")[:4]


def soundex_key(tokens: Iterable[str], *, max_tokens: int = 4) -> str:
    """Return Soundex codes for the most useful Latin name tokens."""
    codes = [soundex_token(token) for token in tokens]
    return pipe_join(_stable_unique(code for code in codes if code)[:max_tokens])


def normalize_address(value: object) -> str:
    """Normalize an address with lightweight common abbreviation expansion."""
    tokens = tokenize(value)
    normalized = [ADDRESS_REPLACEMENTS.get(token, token) for token in tokens]
    return " ".join(normalized)


def extract_numbers(value: object) -> list[str]:
    """Extract numeric address/name tokens such as street numbers and PIN codes."""
    return NUMBER_RE.findall(normalize_text(value))


def extract_house_number(address: object) -> str:
    """Extract the first likely building or street number from an address."""
    text = normalize_address(address)
    match = HOUSE_NUMBER_RE.search(text)
    return match.group(1) if match else ""


def extract_postal_code(address: object, country: object = "") -> str:
    """Extract a likely postal code.

    US and France often use 5 digits, while India uses 6-digit PIN codes. When
    several candidates exist, this returns the last candidate because postal
    codes often appear near the end of addresses.
    """
    text = normalize_text(address)
    country_norm = normalize_text(country)
    if not text:
        return ""

    if country_norm == "india":
        matches = re.findall(r"\b\d{6}\b", text)
    elif country_norm in {"us", "usa", "united states"}:
        matches = re.findall(r"\b\d{5}(?:\s\d{4})?\b", text)
    elif country_norm == "france":
        matches = re.findall(r"\b\d{5}\b", text)
    else:
        matches = re.findall(r"\b\d{5,6}\b", text)

    return matches[-1].replace(" ", "-") if matches else ""


def extract_region_hint(address: object, country: object = "") -> str:
    """Extract a lightweight state/region hint when it is obvious."""
    text = normalize_text(address)
    country_norm = normalize_text(country)
    if not text:
        return ""
    if country_norm in {"us", "usa", "united states"}:
        matches = US_STATE_RE.findall(text)
        return matches[-1].upper() if matches else ""
    return ""


def extract_emails(*values: object) -> list[str]:
    """Extract normalized email addresses from free text fields."""
    text = " ".join(str(value or "") for value in values)
    return _stable_unique(match.casefold() for match in EMAIL_RE.findall(text))


def extract_urls(*values: object) -> list[str]:
    """Extract normalized website-like URLs/domains from free text fields."""
    text = " ".join(str(value or "") for value in values)
    matches = []
    for match in URL_RE.finditer(text):
        if match.start() > 0 and text[match.start() - 1] == "@":
            continue
        url = match.group(0).casefold().rstrip(").,;")
        if url.startswith("http://"):
            url = url.removeprefix("http://")
        elif url.startswith("https://"):
            url = url.removeprefix("https://")
        url = url.removeprefix("www.")
        matches.append(url)
    return _stable_unique(matches)


def extract_phones(*values: object) -> list[str]:
    """Extract phone-like digit strings, keeping only plausible lengths."""
    text = " ".join(str(value or "") for value in values)
    phones = []
    for match in PHONE_RE.findall(text):
        digits = re.sub(r"\D", "", match)
        if 8 <= len(digits) <= 15:
            phones.append(digits)
    return _stable_unique(phones)


def important_tokens(tokens: Iterable[str]) -> list[str]:
    """Return longer, non-stopword tokens useful for blocking."""
    result = []
    for token in tokens:
        if token in STOP_TOKENS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        result.append(token)
    return result


def pipe_join(tokens: Iterable[str]) -> str:
    """Store token lists in a Parquet/CSV-friendly deterministic string format."""
    return "|".join(tokens)


def stable_hash(*values: object, length: int = 16) -> str:
    """Create a deterministic compact fingerprint for record-level checks."""
    joined = "\u241f".join("" if value is None else str(value) for value in values)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]
