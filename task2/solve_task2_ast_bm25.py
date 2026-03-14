"""
Task 2: AST-based Chunk Retrieval Pipeline
==========================================
Strategy: "Chunk-based retrieval using static analysis"

Algorithm:
1. Parse every Python file in the repo into semantic AST chunks
   (module-level functions, classes, class methods)
2. Score chunks with BM25 + multi-signal ranking:
   - BM25 similarity to prefix+suffix
   - Import analysis (is this file imported by the completion file?)
   - Recently modified files
   - Same-directory proximity
   - Identifier overlap (class/function names used in prefix/suffix)
   - __init__.py structural bonus
3. Greedily select top-K chunks within char budget
4. OUTPUT: least relevant first, most relevant LAST
   → context is trimmed from the LEFT by models (Mellum/Codestral/Qwen),
     so the END of context string is closest to the cursor.
"""

import os
import re
import ast
import json
import argparse
from collections import defaultdict
from pathlib import Path

# ── auto-install missing packages ──────────────────────────────────────────────
try:
    import jsonlines
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jsonlines", "-q"])
    import jsonlines

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rank_bm25", "-q"])
    from rank_bm25 import BM25Okapi

# ── constants ──────────────────────────────────────────────────────────────────
FILE_SEP_SYMBOL = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

# Budget: context is trimmed from left, so we can be generous.
# Mellum: 8K tokens ≈ 32K chars; Codestral/Qwen: 16K tokens ≈ 64K chars.
# 55K chars → full for Codestral/Qwen, only the tail (~23K) used for Mellum.
MAX_CONTEXT_CHARS = 55_000

# How many top-scored chunks to consider before budget trimming
TOP_K_CHUNKS = 40


# ── AST parsing ───────────────────────────────────────────────────────────────

def extract_ast_chunks(file_content: str, rel_path: str) -> list[dict]:
    """
    Parse a Python file into semantic chunks.
    Returns list of dicts with keys: file_path, name, type, content, start_line.
    """
    lines = file_content.splitlines(keepends=True)

    def slice_lines(node) -> str:
        return "".join(lines[node.lineno - 1 : node.end_lineno])

    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        # Unparseable → treat whole file as one chunk
        return [
            {
                "file_path": rel_path,
                "name": rel_path,
                "type": "module",
                "content": file_content,
                "start_line": 0,
            }
        ]

    chunks = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                {
                    "file_path": rel_path,
                    "name": node.name,
                    "type": "function",
                    "content": slice_lines(node),
                    "start_line": node.lineno,
                }
            )

        elif isinstance(node, ast.ClassDef):
            # Whole class as one chunk
            chunks.append(
                {
                    "file_path": rel_path,
                    "name": node.name,
                    "type": "class",
                    "content": slice_lines(node),
                    "start_line": node.lineno,
                }
            )
            # Individual methods for finer BM25 granularity
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(
                        {
                            "file_path": rel_path,
                            "name": f"{node.name}.{child.name}",
                            "type": "method",
                            "content": slice_lines(child),
                            "start_line": child.lineno,
                        }
                    )

    # Fallback: tiny or empty files with no top-level definitions
    if not chunks:
        chunks.append(
            {
                "file_path": rel_path,
                "name": rel_path,
                "type": "module",
                "content": file_content,
                "start_line": 0,
            }
        )

    return chunks


def collect_repo_chunks(root_dir: str, extension: str = ".py") -> list[dict]:
    """Walk repo and return all AST chunks from Python files."""
    all_chunks = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune noise directories
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in ("__pycache__", "node_modules", ".git", "venv", ".venv")
        ]
        for fname in filenames:
            if not fname.endswith(extension):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_dir)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if len(content.strip()) < 30:
                    continue
                all_chunks.extend(extract_ast_chunks(content, rel))
            except Exception:
                pass
    return all_chunks


# ── scoring helpers ────────────────────────────────────────────────────────────

def tokenize(s: str) -> list[str]:
    return re.sub(r"[^a-zA-Z0-9_]", " ", s.lower()).split()


def extract_imports(code: str) -> list[str]:
    imports = []
    for m in re.finditer(r"^\s*(?:from\s+(\S+)|import\s+(\S+))", code, re.MULTILINE):
        mod = m.group(1) or m.group(2)
        if mod:
            imports.append(mod.split(".")[0])
            imports.append(mod.replace(".", "/"))
    return imports


_BUILTINS = {
    "self", "None", "True", "False", "print", "len", "range", "str", "int",
    "float", "list", "dict", "set", "tuple", "type", "isinstance", "super",
    "return", "yield", "import", "from", "class", "def", "for", "while",
    "if", "else", "elif", "try", "except", "finally", "with", "as", "not",
    "and", "or", "pass", "break", "continue", "lambda", "raise", "assert",
}


def extract_identifiers(code: str) -> set[str]:
    idents: set[str] = set()
    for m in re.finditer(r"\b(?:class|def)\s+(\w+)", code):
        idents.add(m.group(1))
    for m in re.finditer(r"\b(\w{3,})\s*\(", code):
        idents.add(m.group(1))
    for m in re.finditer(r"\.(\w{3,})", code):
        idents.add(m.group(1))
    return idents - _BUILTINS


# ── main scoring ──────────────────────────────────────────────────────────────

def score_chunks(
    chunks: list[dict],
    completion_path: str,
    prefix: str,
    suffix: str,
    modified_files: list[str],
) -> list[tuple[dict, float]]:
    """
    Score every chunk and return sorted list [(chunk, score)] descending.
    """
    if not chunks:
        return []

    query_text = prefix + " " + suffix
    query_tokens = tokenize(query_text)
    completion_dir = os.path.dirname(completion_path)

    imports = extract_imports(prefix + "\n" + suffix)
    identifiers = extract_identifiers(prefix + "\n" + suffix)
    modified_set = set(modified_files)
    modified_basenames = {os.path.basename(f) for f in modified_files}

    # BM25 over chunk contents
    corpus = [tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    bm25_raw = bm25.get_scores(query_tokens)
    max_bm25 = float(max(bm25_raw)) if bm25_raw.max() > 0 else 1.0

    scored: list[tuple[dict, float]] = []

    for i, chunk in enumerate(chunks):
        fp = chunk["file_path"]

        # Skip the file being completed
        if fp == completion_path:
            continue

        score = 0.0

        # ── 1. BM25 (weight 3.0) ─────────────────────────────────────────────
        score += 3.0 * (bm25_raw[i] / max_bm25)

        # ── 2. Direct identifier match (weight 4.0) ──────────────────────────
        chunk_name = chunk["name"].split(".")[-1]
        if chunk_name in identifiers:
            score += 4.0
        else:
            for ident in identifiers:
                if len(ident) > 4 and (
                    ident.lower() in chunk_name.lower()
                    or chunk_name.lower() in ident.lower()
                ):
                    score += 1.5
                    break

        # ── 3. Import analysis (weight 5.0) ──────────────────────────────────
        path_stem = Path(fp).stem
        path_as_dotted = fp.replace(".py", "").replace("/", ".").replace("\\", ".")
        for imp in imports:
            if path_stem == imp.split(".")[-1]:
                score += 5.0
                break
            if imp and (imp in path_as_dotted or imp.replace(".", "/") in fp):
                score += 3.0
                break

        # ── 4. Modified files (weight 2.5) ───────────────────────────────────
        if fp in modified_set:
            score += 2.5
        elif os.path.basename(fp) in modified_basenames:
            score += 1.5

        # ── 5. Directory proximity (weight 2.0) ──────────────────────────────
        chunk_dir = os.path.dirname(fp)
        if chunk_dir == completion_dir:
            score += 2.0
        elif completion_dir and fp.startswith(completion_dir.split("/")[0] + "/"):
            score += 0.5

        # ── 6. Identifier overlap in chunk content (weight ≤ 2.0) ────────────
        chunk_idents = extract_identifiers(chunk["content"])
        overlap = len(identifiers & chunk_idents)
        if overlap > 0:
            score += min(2.0, overlap * 0.15)

        # ── 7. __init__.py bonus ──────────────────────────────────────────────
        if os.path.basename(fp) == "__init__.py":
            if chunk_dir in completion_dir or completion_dir.startswith(chunk_dir):
                score += 1.5

        # ── 8. Penalise test files when completion file isn't a test ─────────
        is_test_chunk = "test" in fp.lower()
        is_test_completion = "test" in completion_path.lower()
        if is_test_chunk and not is_test_completion:
            score -= 1.0
        elif not is_test_chunk and not is_test_completion:
            score += 0.3

        scored.append((chunk, score))

    scored.sort(key=lambda x: -x[1])
    return scored


# ── context composition ───────────────────────────────────────────────────────

def compose_context(
    scored_chunks: list[tuple[dict, float]],
    max_chars: int = MAX_CONTEXT_CHARS,
    top_k: int = TOP_K_CHUNKS,
) -> str:
    """
    Build context string from top-K scored chunks.

    Layout: least-relevant file … most-relevant file
    (most relevant is at the END → closest to cursor after left-trimming)
    """
    if not scored_chunks:
        return ""

    top = scored_chunks[:top_k]

    # Group chunks by file; track max score and all selected chunks
    file_chunks: dict[str, list[dict]] = defaultdict(list)
    file_max_score: dict[str, float] = defaultdict(float)

    for chunk, score in top:
        fp = chunk["file_path"]
        file_chunks[fp].append(chunk)
        file_max_score[fp] = max(file_max_score[fp], score)

    # Select files greedily in DESCENDING score order (highest priority first)
    sorted_desc = sorted(file_chunks, key=lambda fp: -file_max_score[fp])

    selected: list[tuple[str, str, float]] = []  # (fp, entry_text, score)
    total_chars = 0

    for fp in sorted_desc:
        # Sort chunks within file by line number for coherent output
        sorted_chunks = sorted(file_chunks[fp], key=lambda c: c.get("start_line", 0))

        # Deduplicate: if a class AND its methods are both selected, skip methods
        # (the whole-class chunk already contains the method bodies)
        deduped: list[dict] = []
        class_ranges: list[tuple[int, int]] = []
        for c in sorted_chunks:
            if c["type"] == "class":
                end = c.get("start_line", 0) + c["content"].count("\n")
                class_ranges.append((c.get("start_line", 0), end))
                deduped.append(c)
            elif c["type"] == "method":
                sl = c.get("start_line", 0)
                if any(s <= sl <= e for s, e in class_ranges):
                    continue  # already included in parent class chunk
                deduped.append(c)
            else:
                deduped.append(c)

        file_content = "\n".join(c["content"].rstrip() for c in deduped)
        entry = FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL, file_name=fp, file_content=file_content
        )

        if total_chars + len(entry) <= max_chars:
            selected.append((fp, entry, file_max_score[fp]))
            total_chars += len(entry)
        else:
            # Try truncated version — keep as much as fits
            remaining = max_chars - total_chars
            overhead = len(FILE_SEP_SYMBOL) + len(fp) + 2
            avail = remaining - overhead
            if avail > 300:
                trunc = file_content[:avail]
                last_nl = trunc.rfind("\n")
                if last_nl > 0:
                    trunc = trunc[:last_nl]
                entry_trunc = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=fp, file_content=trunc
                )
                selected.append((fp, entry_trunc, file_max_score[fp]))
                total_chars += len(entry_trunc)
            break  # budget exhausted

    if not selected:
        return ""

    # Output in ASCENDING score order → most relevant file ends up LAST (closest to cursor)
    selected.sort(key=lambda x: x[2])  # ascending

    return "".join(entry for _, entry, _ in selected)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AST+BM25 chunk-based context retrieval for Task 2"
    )
    parser.add_argument("--stage", type=str, default="public",
                        help="Dataset stage: practice | public")
    parser.add_argument("--lang", type=str, default="python",
                        help="Language: python | kotlin")
    parser.add_argument("--top-k", type=int, default=TOP_K_CHUNKS,
                        help="Max chunks to consider per datapoint")
    parser.add_argument("--max-chars", type=int, default=MAX_CONTEXT_CHARS,
                        help="Max characters in composed context")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory")
    args = parser.parse_args()

    stage = args.stage
    language = args.lang
    extension = ".py" if language == "python" else ".kt"

    completion_points_file = os.path.join(args.data_dir, f"{language}-{stage}.jsonl")
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_file = os.path.join(predictions_dir, f"{language}-{stage}-ast-bm25.jsonl")

    print(f"AST+BM25 pipeline: {language}-{stage}")
    print(f"  Completion points: {completion_points_file}")
    print(f"  Output:            {predictions_file}")
    print(f"  Budget:            {args.max_chars:,} chars | top_k={args.top_k}")

    with jsonlines.open(completion_points_file, "r") as reader:
        datapoints = list(reader)

    print(f"  Datapoints: {len(datapoints)}\n")

    results = []

    # Cache chunks per repo to avoid re-parsing the same repo for multiple datapoints
    repo_chunk_cache: dict[str, list[dict]] = {}

    for idx, dp in enumerate(datapoints):
        repo_key = dp["repo"].replace("/", "__")
        repo_revision = dp["revision"]
        root_dir = os.path.join(
            args.data_dir,
            f"repositories-{language}-{stage}",
            f"{repo_key}-{repo_revision}",
        )

        if not os.path.exists(root_dir):
            print(f"  [{idx+1:3d}] WARN repo not found: {root_dir}")
            results.append({"context": ""})
            continue

        # Parse repo chunks (cached)
        cache_key = f"{repo_key}-{repo_revision}"
        if cache_key not in repo_chunk_cache:
            repo_chunk_cache[cache_key] = collect_repo_chunks(root_dir, extension)

        chunks = repo_chunk_cache[cache_key]

        if not chunks:
            results.append({"context": ""})
            continue

        # Score chunks
        scored = score_chunks(
            chunks,
            completion_path=dp["path"],
            prefix=dp["prefix"],
            suffix=dp["suffix"],
            modified_files=dp.get("modified", []),
        )

        # Compose context
        context = compose_context(scored, max_chars=args.max_chars, top_k=args.top_k)

        results.append({"context": context})

        if (idx + 1) % 25 == 0 or idx == 0:
            top3 = [
                f"{c['file_path']}::{c['name']} ({s:.2f})"
                for c, s in scored[:3]
            ]
            ctx_kb = len(context) / 1024
            print(
                f"  [{idx+1:3d}/{len(datapoints)}] {dp['path']}"
                f" | ctx={ctx_kb:.1f}KB"
                f"\n           top3: {top3}"
            )

    # Write predictions
    with jsonlines.open(predictions_file, "w") as writer:
        for r in results:
            writer.write(r)

    print(f"\nDone. {len(results)} predictions → {predictions_file}")

    # Sanity check
    non_empty = sum(1 for r in results if r["context"])
    avg_chars = (
        sum(len(r["context"]) for r in results) / len(results) if results else 0
    )
    print(f"Non-empty contexts: {non_empty}/{len(results)}")
    print(f"Average context length: {avg_chars:,.0f} chars")


if __name__ == "__main__":
    main()
