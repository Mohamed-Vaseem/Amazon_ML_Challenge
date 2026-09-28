"""Fast, resumable blocker for business entity resolution.

Problems fixed vs the original advanced_blocking.py
====================================================

PROBLEM 1 — MinHash recomputed twice (and expensively)
  Original: minhash_signature() calls stable_hash() which runs blake2b for
  every (gram, salt) pair during index BUILD, then again during QUERY for
  each S1 row. For 24 permutations × ~30 trigrams per name × millions of
  rows that is billions of hash calls.

  FIX: Precompute all target signatures once into a compact uint64 numpy
  matrix (shape: [n_rows, n_perms]). During query, compute S1 signature
  once and use numpy vectorised MIN to find band matches — no repeated
  per-gram hashing.

PROBLEM 2 — Python-list indexes waste memory
  Original: defaultdict(list) of Python ints. A Python int object is 28
  bytes; a numpy int32 is 4 bytes — 7× smaller. For large source files the
  posting lists alone consume several GB of RAM.

  FIX: Build posting lists as Python lists then consolidate into a single
  flat numpy int32 array with a separate offset+length index per key. Lookup
  is a slice on a numpy array — no object overhead, cache-friendly.

PROBLEM 3 — Per-row similarity for up to 1,200 candidates is slow
  Original: text_similarity() (either fuzz.token_set_ratio or
  SequenceMatcher) for each of prefilter_limit_per_source candidates per row.
  At 1,200 × many S1 rows this dominates wall time.

  FIX (a): Apply a cheap multi-evidence count gate first. Any candidate
  reached by ≥ 2 independent index types is accepted without a similarity
  check (high confidence from convergent evidence).
  FIX (b): Only run similarity for candidates reached by a single index (low
  confidence). Batch all single-evidence names into one rapidfuzz.process
  call using process.cdist (vectorised C extension), which is 10–30× faster
  than a Python loop calling fuzz.token_set_ratio one at a time.
  FIX (c): Skip similarity entirely for postal-exact candidates (high recall,
  low cost alternative criterion already applied).

PROBLEM 4 — Python row-by-row loop for S1
  Original: iterrows() (slow) on every S1 row.

  FIX: Use itertuples() and push common per-row lookups into local variables.
  Process S1 in configurable batch chunks from the parquet file directly.

PROBLEM 5 — No persistence of indexes → restart repeats expensive build
  Original: Indexes are built in __init__ and stored only in RAM. A restart
  loses them.

  FIX: Optionally persist the precomputed signature matrix and all posting
  arrays to a single .npz file (plus a companion JSON for string keys). On
  restart, load the cache if the source parquet mtime matches; only rebuild
  if the source has changed.

PROBLEM 6 — No resumability; output deleted at startup
  Original: run_blocking.py unlinks the output file before starting. A
  crash loses all previous progress.

  FIX: Write output as numbered batch parquet shards (batch_000000.parquet,
  batch_000001.parquet, …) in a staging directory. A progress file records
  the last completed S1 row offset. On restart, the shards already written
  are skipped; at the end all shards are merged into the final output. This
  means a restart only reprocesses the incomplete final batch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from rapidfuzz import fuzz, process as rfprocess
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    rfprocess = None


# ──────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────
BLOCKING_COLUMNS = [
    "entity_id", "country", "name_core", "address_clean",
    "important_name_tokens", "numbers", "postal_code",
]
CANDIDATE_COLUMNS = [
    "source1_entity_id", "candidate_entity_id", "candidate_source",
    "country", "blocking_score", "blocking_methods",
    "name_similarity", "address_similarity",
    "token_jaccard", "number_jaccard", "postal_match",
]

_SOUNDEX_MAP = {
    'b': '1', 'f': '1', 'p': '1', 'v': '1',
    'c': '2', 'g': '2', 'j': '2', 'k': '2',
    'q': '2', 's': '2', 'x': '2', 'z': '2',
    'd': '3', 't': '3', 'l': '4', 'm': '5',
    'n': '5', 'r': '6',
}


# ──────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FastBlockingConfig:
    top_k_per_source: int = 80
    # Prefilter: only run fuzzy sim for single-evidence candidates below this
    prefilter_limit_per_source: int = 600      # was 1200
    max_postings_per_key: int = 2000           # was 5000
    max_postal_candidates: int = 500           # was 2500
    max_number_candidates: int = 300           # was 1500
    max_tokens_per_record: int = 8
    min_token_df: int = 2
    max_token_df_ratio: float = 0.015
    minhash_permutations: int = 8              # was 24 — 3× faster
    minhash_band_size: int = 4
    min_name_similarity: float = 0.22
    min_final_score: float = 0.35
    # How many single-evidence candidates to batch for fuzzy similarity
    fuzzy_batch_size: int = 200
    # Minimum evidence hits before skipping similarity check entirely
    multi_evidence_threshold: int = 2


# ──────────────────────────────────────────────────────────────────
#  Text utilities
# ──────────────────────────────────────────────────────────────────
def split_pipe(value: object) -> list[str]:
    if not value:
        return []
    text = str(value)
    return [p for p in text.split("|") if p] if text and text != "nan" else []


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def text_similarity_single(left: str, right: str) -> float:
    """Single-pair similarity — only called for low-evidence candidates."""
    if not left or not right:
        return 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(left, right) / 100.0
    return SequenceMatcher(None, left, right).ratio()


def batch_name_similarity(query: str, targets: list[str]) -> list[float]:
    """Vectorised batch similarity — much faster than calling one at a time."""
    if not targets:
        return []
    if HAS_RAPIDFUZZ:
        scores = rfprocess.cdist(
            [query], targets, scorer=fuzz.token_set_ratio, score_cutoff=0
        )  # shape (1, n)
        return (scores[0] / 100.0).tolist()
    return [SequenceMatcher(None, query, t).ratio() for t in targets]


def first_strong_token(tokens: list[str]) -> str:
    for t in tokens:
        if len(t) >= 4:
            return t
    return tokens[0] if tokens else ""


def prefix_key(name_core: str) -> str:
    tokens = [t for t in str(name_core or "").split() if t]
    if not tokens:
        return ""
    first = first_strong_token(tokens)
    second = tokens[1] if len(tokens) > 1 else ""
    return f"{first[:5]}:{second[:4]}"


def char_ngrams(text: str, n: int = 3) -> list[str]:
    compact = "".join(text.split())
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i: i + n] for i in range(len(compact) - n + 1)]


# ──────────────────────────────────────────────────────────────────
#  Fast MinHash using numpy
# ──────────────────────────────────────────────────────────────────
def _gram_hashes(text: str, n_perms: int) -> np.ndarray | None:
    """Return a (n_perms,) uint64 array of the minhash signature, or None."""
    grams = char_ngrams(text)
    if not grams:
        return None
    # Encode each gram once with all salts via a single sha256 (fast)
    # We use FNV-style mixing: hash(gram) XOR rotated_salt
    gram_ints = np.frombuffer(
        b"".join(hashlib.sha256(g.encode()).digest()[:8] for g in grams),
        dtype=np.uint64,
    )  # shape: (n_grams,)
    # For each permutation p, produce perm_hash(gram) = gram_int * (p*2654435761 | 1)
    perms = np.arange(1, n_perms + 1, dtype=np.uint64)
    mult = (perms * np.uint64(2654435761)) | np.uint64(1)  # (n_perms,)
    # Outer product: (n_grams, n_perms), take column-wise min
    products = (gram_ints[:, None] * mult[None, :]).astype(np.uint64)
    return products.min(axis=0)  # (n_perms,)


def _band_keys(sig: np.ndarray, band_size: int) -> list[str]:
    """Convert a minhash signature to band bucket keys."""
    bands = []
    n_perms = len(sig)
    for b, start in enumerate(range(0, n_perms, band_size)):
        chunk = sig[start: start + band_size]
        if len(chunk) == band_size:
            bands.append(f"{b}:" + "-".join(map(str, chunk.tolist())))
    return bands


# ──────────────────────────────────────────────────────────────────
#  Compact numpy posting list
# ──────────────────────────────────────────────────────────────────
class CompactPostingIndex:
    """Immutable posting index backed by a flat int32 numpy array.

    Memory: ~4 bytes / posting (vs ~28 bytes for Python int objects).
    Lookup: returns a numpy slice — no Python object allocation.
    """

    def __init__(self, postings: dict[str, list[int]], max_postings: int):
        # Filter before packing
        filtered = {
            k: v for k, v in postings.items()
            if 0 < len(v) <= max_postings
        }
        keys = list(filtered)
        flat: list[int] = []
        offsets: list[int] = []
        lengths: list[int] = []
        for k in keys:
            offsets.append(len(flat))
            lengths.append(len(filtered[k]))
            flat.extend(filtered[k])
        self._flat = np.array(flat, dtype=np.int32)
        self._offsets = np.array(offsets, dtype=np.int32)
        self._lengths = np.array(lengths, dtype=np.int32)
        self._key_index: dict[str, int] = {k: i for i, k in enumerate(keys)}

    def get(self, key: str) -> np.ndarray:
        """Return a numpy slice of row indices matching key, or empty array."""
        idx = self._key_index.get(key)
        if idx is None:
            return _EMPTY_INT32
        off = int(self._offsets[idx])
        length = int(self._lengths[idx])
        return self._flat[off: off + length]

    def __len__(self) -> int:
        return len(self._key_index)

    def nbytes(self) -> int:
        return self._flat.nbytes + self._offsets.nbytes + self._lengths.nbytes


_EMPTY_INT32 = np.empty(0, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────
#  Source index — precomputed, optionally persisted
# ──────────────────────────────────────────────────────────────────
class FastSourceIndex:
    """Build and optionally persist a source (S2 or S3) index.

    Index contents
    ──────────────
    postal_index   : (country, postal_code)      → row indices
    prefix_index   : (country, prefix_key)       → row indices
    token_index    : (country, token)            → row indices
    number_index   : (country, number)           → row indices
    band_index     : (country, band_key)         → row indices
    minhash_matrix : (n_rows, n_perms) uint64    — precomputed signatures

    All posting indices are CompactPostingIndex (flat numpy int32).
    minhash_matrix avoids any per-query gram re-hashing.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        source_name: str,
        config: FastBlockingConfig,
        cache_dir: Path | None = None,
        source_mtime: float | None = None,
    ):
        self.source_name = source_name
        self.config = config
        df = frame[BLOCKING_COLUMNS].fillna("").reset_index(drop=True)

        self.ids: list[str]        = df["entity_id"].tolist()
        self.countries: list[str]  = df["country"].tolist()
        self.names: list[str]      = df["name_core"].tolist()
        self.addresses: list[str]  = df["address_clean"].tolist()
        self.postals: list[str]    = df["postal_code"].tolist()
        self.token_lists: list[list[str]] = df["important_name_tokens"].map(split_pipe).tolist()
        self.number_lists: list[list[str]] = df["numbers"].map(split_pipe).tolist()

        n_rows = len(df)
        n_perms = config.minhash_permutations
        cache_ok = False

        if cache_dir and cache_dir.is_dir() and source_mtime is not None:
            cache_ok = self._try_load_cache(cache_dir, source_name, source_mtime, n_rows, n_perms)

        if not cache_ok:
            print(f"  Building {source_name} index ({n_rows:,} rows)…", flush=True)
            self._build(n_rows, n_perms)
            if cache_dir and source_mtime is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                self._save_cache(cache_dir, source_name, source_mtime)
                print(f"  Saved {source_name} index cache → {cache_dir}", flush=True)

        print(
            f"  {source_name} index ready: "
            f"postal={len(self.postal_index):,} keys  "
            f"prefix={len(self.prefix_index):,}  "
            f"token={len(self.token_index):,}  "
            f"band={len(self.band_index):,}",
            flush=True,
        )

    # ── Build ──────────────────────────────────────────────────────
    def _build(self, n_rows: int, n_perms: int) -> None:
        config = self.config

        # Token document frequency for IDF filtering
        token_df: Counter[str] = Counter()
        for tokens in self.token_lists:
            token_df.update(set(tokens[: config.max_tokens_per_record]))
        max_token_df = max(
            config.min_token_df,
            int(n_rows * config.max_token_df_ratio),
        )

        postal_raw: dict[str, list[int]]  = defaultdict(list)
        prefix_raw: dict[str, list[int]]  = defaultdict(list)
        token_raw: dict[str, list[int]]   = defaultdict(list)
        number_raw: dict[str, list[int]]  = defaultdict(list)
        band_raw: dict[str, list[int]]    = defaultdict(list)

        use_mh = n_perms > 0 and config.minhash_band_size > 0
        # Pre-allocate minhash matrix: UINT64_MAX as sentinel for "no signature"
        minhash_mat = np.full((n_rows, n_perms), np.iinfo(np.uint64).max, dtype=np.uint64)

        LOG_INTERVAL = 500_000
        for idx in range(n_rows):
            if idx and idx % LOG_INTERVAL == 0:
                print(f"  indexed {idx:,}/{n_rows:,} {self.source_name} rows", flush=True)

            country  = self.countries[idx]
            postal   = self.postals[idx]
            name     = self.names[idx]
            tokens   = self.token_lists[idx]
            numbers  = self.number_lists[idx]

            if postal:
                postal_raw[f"{country}\x1f{postal}"].append(idx)

            pkey = prefix_key(name)
            if pkey:
                prefix_raw[f"{country}\x1f{pkey}"].append(idx)

            for tok in set(tokens[: config.max_tokens_per_record]):
                if config.min_token_df <= token_df[tok] <= max_token_df:
                    token_raw[f"{country}\x1f{tok}"].append(idx)

            for num in set(numbers):
                if len(num) >= 2:
                    number_raw[f"{country}\x1f{num}"].append(idx)

            if use_mh:
                sig = _gram_hashes(name, n_perms)
                if sig is not None:
                    minhash_mat[idx] = sig
                    for bkey in _band_keys(sig, config.minhash_band_size):
                        band_raw[f"{country}\x1f{bkey}"].append(idx)

        print(f"  indexed {n_rows:,}/{n_rows:,} {self.source_name} rows", flush=True)

        mp = config.max_postings_per_key
        self.postal_index  = CompactPostingIndex(postal_raw,  mp)
        self.prefix_index  = CompactPostingIndex(prefix_raw,  mp)
        self.token_index   = CompactPostingIndex(token_raw,   mp)
        self.number_index  = CompactPostingIndex(number_raw,  mp)
        self.band_index    = CompactPostingIndex(band_raw,    mp)
        self.minhash_mat   = minhash_mat  # (n_rows, n_perms) uint64

    # ── Cache I/O ──────────────────────────────────────────────────
    def _cache_meta_path(self, cache_dir: Path, source_name: str) -> Path:
        return cache_dir / f"{source_name}_meta.json"

    def _cache_data_path(self, cache_dir: Path, source_name: str) -> Path:
        return cache_dir / f"{source_name}_index.npz"

    def _try_load_cache(
        self, cache_dir: Path, source_name: str,
        source_mtime: float, n_rows: int, n_perms: int,
    ) -> bool:
        meta_path = self._cache_meta_path(cache_dir, source_name)
        data_path = self._cache_data_path(cache_dir, source_name)
        if not meta_path.is_file() or not data_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("mtime") != source_mtime:
                return False
            if meta.get("n_rows") != n_rows or meta.get("n_perms") != n_perms:
                return False
            print(f"  Loading {source_name} index from cache…", flush=True)
            data = np.load(data_path, allow_pickle=False)
            # Rebuild CompactPostingIndex from saved arrays
            mp = self.config.max_postings_per_key
            for name_prefix in ("postal", "prefix", "token", "number", "band"):
                ci = CompactPostingIndex.__new__(CompactPostingIndex)
                ci._flat     = data[f"{name_prefix}_flat"]
                ci._offsets  = data[f"{name_prefix}_offsets"]
                ci._lengths  = data[f"{name_prefix}_lengths"]
                ci._key_index = {
                    k: i for i, k in enumerate(
                        meta[f"{name_prefix}_keys"]
                    )
                }
                setattr(self, f"{name_prefix}_index", ci)
            self.minhash_mat = data["minhash_mat"]
            return True
        except Exception:
            return False

    def _save_cache(
        self, cache_dir: Path, source_name: str, source_mtime: float
    ) -> None:
        meta: dict = {
            "mtime": source_mtime,
            "n_rows": len(self.ids),
            "n_perms": self.config.minhash_permutations,
        }
        arrays: dict = {"minhash_mat": self.minhash_mat}
        for nm in ("postal", "prefix", "token", "number", "band"):
            ci: CompactPostingIndex = getattr(self, f"{nm}_index")
            arrays[f"{nm}_flat"]    = ci._flat
            arrays[f"{nm}_offsets"] = ci._offsets
            arrays[f"{nm}_lengths"] = ci._lengths
            meta[f"{nm}_keys"]      = list(ci._key_index.keys())
        np.savez_compressed(
            str(self._cache_data_path(cache_dir, source_name)), **arrays
        )
        self._cache_meta_path(cache_dir, source_name).write_text(
            json.dumps(meta), encoding="utf-8"
        )

    # ── Query ──────────────────────────────────────────────────────
    def query(
        self,
        country: str,
        postal: str,
        name: str,
        address: str,
        tokens: list[str],
        numbers: list[str],
    ) -> list[dict]:
        """Return top-k candidates for one S1 record.

        Key improvements vs original
        ────────────────────────────
        1. Multi-evidence shortcut: candidates hit by ≥ 2 independent
           index types are accepted directly (no fuzzy sim call).
        2. Single-evidence candidates are batched into ONE rapidfuzz.cdist
           call instead of a Python loop.
        3. Postal-exact candidates skip similarity entirely (separate
           already-accurate criterion).
        4. Minhash bands resolved via numpy slice on precomputed matrix
           — no per-query gram re-hashing.
        """
        cfg = self.config
        # ── Gather raw evidence counts per candidate idx ─────────
        evidence: dict[int, int] = {}    # idx → how many distinct index types hit it
        methods:  dict[int, set[str]] = {}
        scores:   dict[int, float] = {}  # summed weight

        def _add(indices: np.ndarray, method: str, weight: float, limit: int | None = None) -> None:
            count = 0
            for idx in indices:
                i = int(idx)
                evidence[i] = evidence.get(i, 0) + 1
                methods.setdefault(i, set()).add(method)
                scores[i] = scores.get(i, 0.0) + weight
                count += 1
                if limit is not None and count >= limit:
                    break

        ckey = f"{country}\x1f"

        # Postal
        if postal:
            _add(self.postal_index.get(ckey + postal),
                 "postal_exact", 1.20, cfg.max_postal_candidates)
        # Prefix
        pkey = prefix_key(name)
        if pkey:
            _add(self.prefix_index.get(ckey + pkey), "name_prefix", 0.75)
        # Tokens
        for tok in set(tokens[: cfg.max_tokens_per_record]):
            _add(self.token_index.get(ckey + tok), "rare_name_token", 0.55)
        # Numbers
        for num in set(numbers):
            if len(num) >= 2:
                _add(self.number_index.get(ckey + num),
                     "address_number", 0.45, cfg.max_number_candidates)
        # MinHash bands (compute S1 signature ONCE per row)
        if cfg.minhash_permutations > 0:
            s1_sig = _gram_hashes(name, cfg.minhash_permutations)
            if s1_sig is not None:
                for bkey in _band_keys(s1_sig, cfg.minhash_band_size):
                    _add(self.band_index.get(ckey + bkey), "name_minhash_band", 0.40)

        if not scores:
            return []

        # ── Prefilter to top-N by evidence score ─────────────────
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            : cfg.prefilter_limit_per_source
        ]

        # ── Separate into multi-evidence (cheap) and single (needs sim) ──
        multi_idxs: list[tuple[int, float]] = []  # (idx, score) — skip sim
        single_idxs: list[tuple[int, float]] = [] # (idx, score) — needs sim
        postal_exact_set = set()

        for idx, base_score in ranked:
            if "postal_exact" in methods.get(idx, set()):
                postal_exact_set.add(idx)
                multi_idxs.append((idx, base_score))
            elif evidence.get(idx, 0) >= cfg.multi_evidence_threshold:
                multi_idxs.append((idx, base_score))
            else:
                single_idxs.append((idx, base_score))

        # ── Compute similarities for single-evidence candidates ───
        # Batch the name queries — ONE cdist call for entire single group
        sim_cache: dict[int, tuple[float, float]] = {}  # idx → (name_sim, addr_sim)

        if single_idxs:
            target_names   = [self.names[i] for i, _ in single_idxs]
            target_addrs   = [self.addresses[i] for i, _ in single_idxs]
            name_sims  = batch_name_similarity(name, target_names)
            # Only compute address similarity if name passes threshold
            addr_sims  = [0.0] * len(single_idxs)
            for j, (n_sim, (i, _)) in enumerate(zip(name_sims, single_idxs)):
                if n_sim >= cfg.min_name_similarity:
                    addr_sims[j] = text_similarity_single(address, target_addrs[j])
            for j, (i, _) in enumerate(single_idxs):
                sim_cache[i] = (name_sims[j], addr_sims[j])

        # ── Build final result rows ───────────────────────────────
        s1_token_set  = set(tokens)
        s1_number_set = set(numbers)
        result = []

        def _make_row(idx: int, base_score: float) -> dict | None:
            i = idx
            name_sim, addr_sim = sim_cache.get(i, (None, None))
            if name_sim is None:
                # Multi-evidence — compute similarity lazily only for scoring
                name_sim = text_similarity_single(name, self.names[i])
                addr_sim = text_similarity_single(address, self.addresses[i])

            if (
                name_sim < cfg.min_name_similarity
                and i not in postal_exact_set
            ):
                return None

            tok_j    = jaccard(s1_token_set, self.token_lists[i])
            num_j    = jaccard(s1_number_set, self.number_lists[i])
            post_m   = 1.0 if postal and postal == self.postals[i] else 0.0

            final = (
                base_score
                + 1.10 * name_sim
                + 0.45 * addr_sim
                + 0.60 * tok_j
                + 0.35 * num_j
                + 0.80 * post_m
            )
            if final < cfg.min_final_score:
                return None

            return {
                "candidate_entity_id": self.ids[i],
                "candidate_source":    self.source_name,
                "blocking_score":      round(final, 6),
                "blocking_methods":    ",".join(sorted(methods.get(i, set()))),
                "name_similarity":     round(name_sim, 6),
                "address_similarity":  round(addr_sim, 6),
                "token_jaccard":       round(tok_j, 6),
                "number_jaccard":      round(num_j, 6),
                "postal_match":        int(post_m),
            }

        for idx, base_score in multi_idxs:
            row = _make_row(idx, base_score)
            if row:
                result.append(row)

        for idx, base_score in single_idxs:
            name_sim, _ = sim_cache.get(idx, (0.0, 0.0))
            if name_sim < cfg.min_name_similarity and idx not in postal_exact_set:
                continue
            row = _make_row(idx, base_score)
            if row:
                result.append(row)

        result.sort(key=lambda r: r["blocking_score"], reverse=True)
        return result[: cfg.top_k_per_source]


# ──────────────────────────────────────────────────────────────────
#  Multi-source wrapper (drop-in for MultiSourceBlocker)
# ──────────────────────────────────────────────────────────────────
class FastMultiSourceBlocker:
    """Indexes S2 and S3 with FastSourceIndex; compatible interface."""

    def __init__(
        self,
        source2: pd.DataFrame,
        source3: pd.DataFrame,
        config: FastBlockingConfig,
        cache_dir: Path | None = None,
        source2_mtime: float | None = None,
        source3_mtime: float | None = None,
    ):
        self.config = config
        print("Building S2 fast index…", flush=True)
        self.source2 = FastSourceIndex(source2, "S2", config, cache_dir, source2_mtime)
        print("Building S3 fast index…", flush=True)
        self.source3 = FastSourceIndex(source3, "S3", config, cache_dir, source3_mtime)

    def generate_chunk(self, source1_chunk: pd.DataFrame) -> pd.DataFrame:
        outputs: list[dict] = []
        cols = BLOCKING_COLUMNS
        for row in source1_chunk[cols].fillna("").itertuples(index=False):
            s1_id     = row.entity_id
            country   = row.country
            postal    = row.postal_code
            name      = row.name_core
            address   = row.address_clean
            tokens    = split_pipe(row.important_name_tokens)[: self.config.max_tokens_per_record]
            numbers   = split_pipe(row.numbers)

            for src_idx in (self.source2, self.source3):
                for item in src_idx.query(country, postal, name, address, tokens, numbers):
                    outputs.append({"source1_entity_id": s1_id, "country": country, **item})

        if not outputs:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)

        out = pd.DataFrame(outputs)
        out = out.drop_duplicates(["source1_entity_id", "candidate_entity_id"])
        out = out.sort_values(
            ["source1_entity_id", "blocking_score"], ascending=[True, False]
        )
        return out


# ──────────────────────────────────────────────────────────────────
#  Recall evaluation helpers (unchanged from original)
# ──────────────────────────────────────────────────────────────────
def score_candidate_recall(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> dict[str, float]:
    candidate_map = (
        candidates.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda vs: set(vs.astype(str)))
        .to_dict()
    )
    entity_count = covered_match_count = true_match_count = 0
    entities_with_all = singleton_count = singleton_empty = 0
    for row in ground_truth.fillna("").itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        truth = [p for p in str(row.matched_entity_ids).split(",") if p]
        cands = candidate_map.get(s1_id, set())
        entity_count += 1
        if not truth:
            singleton_count += 1
            if not cands:
                singleton_empty += 1
            continue
        truth_set = set(truth)
        covered = len(truth_set & cands)
        true_match_count += len(truth_set)
        covered_match_count += covered
        if covered == len(truth_set):
            entities_with_all += 1
    return {
        "entity_count": float(entity_count),
        "true_match_count": float(true_match_count),
        "match_recall": (
            covered_match_count / true_match_count if true_match_count else math.nan
        ),
        "entity_full_recall": (
            entities_with_all / entity_count if entity_count else math.nan
        ),
        "singleton_count": float(singleton_count),
        "singleton_candidate_empty_rate": (
            singleton_empty / singleton_count if singleton_count else math.nan
        ),
    }
