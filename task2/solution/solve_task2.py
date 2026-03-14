"""
Task 2: Chunk-based Context Collection Pipeline for Code Completion.

Architecture (ASE 2025 SOTA-inspired):
1. Parse repository files into AST → extract semantic chunks (classes, methods, functions)
2. BM25 ranking on chunks (not whole files)
3. Import-graph boosting for directly imported modules
4. Context ordering: highest relevance chunks at END (closest to FIM cursor)
5. Budget management for 3 model context windows (8K/16K/16K tokens)

Key: context is trimmed from LEFT → most important content must be at END.
"""

import os
import sys
import ast
import json
import re
import argparse
import zipfile
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    os.system("pip install rank_bm25")
    from rank_bm25 import BM25Okapi

FILE_SEP_SYMBOL = "<|file_sep|>"

# Context budget: Mellum 8K tokens, Codestral/Qwen 16K tokens
# ~4 chars/token average for code → 8K = ~32K chars, 16K = ~64K chars
# We aim for ~60K: full use for 16K models, most relevant tail for 8K
MAX_CONTEXT_CHARS = 60000

# How many top chunks to include
MAX_CHUNKS = 30


# ────────────────────────────────────────────────────────────
# AST-based Chunking
# ────────────────────────────────────────────────────────────

class CodeChunk:
    """A semantic unit extracted from a Python file."""
    __slots__ = ['file_path', 'name', 'kind', 'code', 'start_line', 'end_line',
                 'parent_class', 'imports_in_chunk']

    def __init__(self, file_path, name, kind, code, start_line, end_line,
                 parent_class=None, imports_in_chunk=None):
        self.file_path = file_path
        self.name = name
        self.kind = kind  # 'class', 'function', 'method', 'module_header', 'top_level'
        self.code = code
        self.start_line = start_line
        self.end_line = end_line
        self.parent_class = parent_class
        self.imports_in_chunk = imports_in_chunk or []


def parse_file_to_chunks(file_path: str, content: str) -> list[CodeChunk]:
    """Parse a Python file into semantic chunks using AST."""
    chunks = []
    lines = content.split('\n')

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        # Fallback: treat entire file as one chunk
        if content.strip():
            chunks.append(CodeChunk(
                file_path=file_path, name=Path(file_path).stem,
                kind='module', code=content,
                start_line=1, end_line=len(lines)
            ))
        return chunks

    # Extract module-level imports and top-level code (header)
    header_end = 0
    module_imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header_end = max(header_end, getattr(node, 'end_lineno', node.lineno))
            if isinstance(node, ast.ImportFrom) and node.module:
                module_imports.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_imports.append(alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            # Constants, docstrings at module level
            if node.lineno <= header_end + 5:
                header_end = max(header_end, getattr(node, 'end_lineno', node.lineno))

    # Module header chunk (imports + module-level assignments)
    if header_end > 0:
        header_code = '\n'.join(lines[:header_end])
        if header_code.strip():
            chunks.append(CodeChunk(
                file_path=file_path, name=f"{Path(file_path).stem}:header",
                kind='module_header', code=header_code,
                start_line=1, end_line=header_end,
                imports_in_chunk=module_imports
            ))

    # Extract class and function definitions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', node.lineno)
            class_code = '\n'.join(lines[start:end])

            # Add full class as a chunk
            chunks.append(CodeChunk(
                file_path=file_path, name=node.name,
                kind='class', code=class_code,
                start_line=start + 1, end_line=end
            ))

            # Also extract individual methods as separate chunks
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_start = item.lineno - 1
                    m_end = getattr(item, 'end_lineno', item.lineno)
                    method_code = '\n'.join(lines[m_start:m_end])
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        name=f"{node.name}.{item.name}",
                        kind='method', code=method_code,
                        start_line=m_start + 1, end_line=m_end,
                        parent_class=node.name
                    ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', node.lineno)
            func_code = '\n'.join(lines[start:end])
            chunks.append(CodeChunk(
                file_path=file_path, name=node.name,
                kind='function', code=func_code,
                start_line=start + 1, end_line=end
            ))

    # If no chunks extracted (e.g. script with only top-level code), add whole file
    if not chunks and content.strip():
        chunks.append(CodeChunk(
            file_path=file_path, name=Path(file_path).stem,
            kind='module', code=content,
            start_line=1, end_line=len(lines)
        ))

    return chunks


# ────────────────────────────────────────────────────────────
# Import Analysis
# ────────────────────────────────────────────────────────────

def extract_imports_from_code(code: str) -> list[str]:
    """Extract import module names from Python code."""
    imports = []
    for line in code.split('\n'):
        line = line.strip()
        m = re.match(r'^from\s+([\w.]+)\s+import', line)
        if m:
            imports.append(m.group(1))
        m = re.match(r'^import\s+([\w.]+)', line)
        if m:
            imports.append(m.group(1))
    return imports


def module_matches_path(module: str, rel_path: str) -> bool:
    """Check if an import module matches a file path."""
    mod_path = module.replace('.', '/')
    clean_path = rel_path.replace('.py', '').replace('/__init__', '')

    if clean_path == mod_path or clean_path.endswith('/' + mod_path):
        return True

    mod_parts = module.split('.')
    path_stem = Path(rel_path).stem
    if path_stem == mod_parts[-1] and path_stem != '__init__':
        return True

    if mod_path in clean_path:
        return True

    return False


# ────────────────────────────────────────────────────────────
# Symbol / Name Extraction
# ────────────────────────────────────────────────────────────

def extract_referenced_names(code: str) -> set[str]:
    """Extract identifiers used in code (calls, attributes, types)."""
    names = set()
    for m in re.finditer(r'\b(\w{3,})\s*\(', code):
        names.add(m.group(1))
    for m in re.finditer(r'(\w{3,})\.(\w{3,})', code):
        names.add(m.group(1))
        names.add(m.group(2))
    for m in re.finditer(r':\s*(\w{3,})', code):
        names.add(m.group(1))
    for m in re.finditer(r'=\s*(\w{3,})\s*\(', code):
        names.add(m.group(1))

    noise = {
        'self', 'cls', 'None', 'True', 'False', 'print', 'len', 'range', 'str',
        'int', 'float', 'list', 'dict', 'set', 'tuple', 'type', 'isinstance',
        'super', 'open', 'map', 'filter', 'zip', 'enumerate', 'sorted', 'reversed',
        'any', 'all', 'min', 'max', 'abs', 'sum', 'hash', 'repr', 'iter',
        'next', 'getattr', 'setattr', 'hasattr', 'delattr', 'property',
        'staticmethod', 'classmethod', 'raise', 'return', 'yield', 'assert',
        'pass', 'break', 'continue', 'for', 'while', 'with', 'try', 'except',
        'finally', 'import', 'from', 'class', 'def', 'elif', 'else', 'and',
        'not', 'lambda', 'global', 'nonlocal', 'del', 'async', 'await',
    }
    return names - noise


# ────────────────────────────────────────────────────────────
# File Discovery
# ────────────────────────────────────────────────────────────

def get_all_python_files(root_dir: str) -> list[tuple[str, str]]:
    """Get all Python files as (relative_path, content) pairs."""
    files = []
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.tox', 'venv', 'env',
                 '.venv', '.eggs', '.mypy_cache', '.pytest_cache'}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith('.')]
        for fn in filenames:
            if fn.endswith('.py'):
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            rel = os.path.relpath(fp, root_dir)
                            files.append((rel, content))
                except:
                    pass
    return files


# ────────────────────────────────────────────────────────────
# BM25 Tokenizer
# ────────────────────────────────────────────────────────────

def tokenize_for_bm25(s: str) -> list[str]:
    """Tokenize for BM25 — split camelCase, snake_case, etc."""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return re.sub(r'[^a-zA-Z0-9_]', ' ', s.lower()).split()


# ────────────────────────────────────────────────────────────
# Chunk Scoring & Ranking
# ────────────────────────────────────────────────────────────

def score_chunks(
    chunks: list[CodeChunk],
    prefix: str,
    suffix: str,
    target_path: str,
    modified_files: list[str],
) -> list[tuple[CodeChunk, float]]:
    """
    Score chunks using multiple signals:
    1. BM25 similarity to prefix+suffix (3.0)
    2. Import resolution — chunk's file is imported by target (8.0)
    3. Symbol overlap — chunk defines symbols used in prefix/suffix (6.0)
    4. Same-file/directory proximity (3.0)
    5. Recently modified file (2.0)
    6. Chunk kind bonus (class headers, __init__.py) (1.0)
    """
    if not chunks:
        return []

    query_text = prefix + "\n" + suffix
    query_tokens = tokenize_for_bm25(query_text)
    query_imports = extract_imports_from_code(query_text)
    referenced_names = extract_referenced_names(query_text)
    target_dir = os.path.dirname(target_path)
    target_stem = Path(target_path).stem

    modified_set = set(modified_files) if modified_files else set()

    # Pre-compute import-matched files
    import_matched_files = set()
    for imp in query_imports:
        for chunk in chunks:
            if module_matches_path(imp, chunk.file_path):
                import_matched_files.add(chunk.file_path)

    # BM25 on chunk codes
    corpus = [tokenize_for_bm25(c.code) for c in chunks]
    # Filter out empty corpora
    valid_indices = [i for i, c in enumerate(corpus) if len(c) > 0]
    if not valid_indices:
        return []

    valid_corpus = [corpus[i] for i in valid_indices]
    valid_chunks = [chunks[i] for i in valid_indices]

    bm25 = BM25Okapi(valid_corpus)
    bm25_raw = bm25.get_scores(query_tokens)
    max_bm25 = max(bm25_raw) if len(bm25_raw) > 0 and max(bm25_raw) > 0 else 1.0

    scored = []
    for idx, (chunk, bm25_score) in enumerate(zip(valid_chunks, bm25_raw)):
        # Skip chunks from the target file itself
        if chunk.file_path == target_path:
            continue

        score = 0.0

        # 1. BM25 similarity (weight: 3.0)
        score += 3.0 * (bm25_score / max_bm25)

        # 2. Import resolution (weight: 8.0)
        if chunk.file_path in import_matched_files:
            score += 8.0

        # 3. Symbol overlap (weight: 6.0)
        # Check if chunk defines symbols referenced in prefix/suffix
        chunk_name = chunk.name.split('.')[-1]  # Get just the name part
        if chunk_name in referenced_names:
            score += 6.0
        elif chunk.parent_class and chunk.parent_class in referenced_names:
            score += 4.0
        else:
            # Check definitions inside chunk code
            chunk_defs = set()
            for m in re.finditer(r'(?:def|class)\s+(\w+)', chunk.code):
                chunk_defs.add(m.group(1))
            overlap = referenced_names & chunk_defs
            if overlap:
                score += min(5.0, len(overlap) * 1.5)

        # 4. Same directory proximity (weight: 3.0)
        chunk_dir = os.path.dirname(chunk.file_path)
        if chunk_dir == target_dir:
            score += 3.0
        elif target_dir and chunk_dir.startswith(target_dir.split('/')[0]):
            score += 1.0

        # 5. Recently modified (weight: 2.0)
        if chunk.file_path in modified_set:
            score += 2.0

        # 6. Kind bonuses
        if chunk.kind == 'module_header' and chunk.file_path in import_matched_files:
            score += 1.5  # Import headers of imported modules are useful
        if os.path.basename(chunk.file_path) == '__init__.py':
            init_dir = os.path.dirname(chunk.file_path)
            if target_dir.startswith(init_dir) or init_dir == target_dir:
                score += 1.0

        # 7. Reverse dependency: chunk imports the target
        if chunk.imports_in_chunk:
            for imp in chunk.imports_in_chunk:
                if target_stem in imp.split('.'):
                    score += 1.0
                    break

        scored.append((chunk, score))

    # Sort by score descending
    scored.sort(key=lambda x: -x[1])
    return scored


# ────────────────────────────────────────────────────────────
# Context Composition
# ────────────────────────────────────────────────────────────

def compose_context(
    scored_chunks: list[tuple[CodeChunk, float]],
    max_chars: int = MAX_CONTEXT_CHARS,
    max_chunks: int = MAX_CHUNKS,
) -> str:
    """
    Compose context from scored chunks.

    CRITICAL: Context is trimmed from LEFT.
    → Most relevant chunks must be at END (survive trimming for small models).

    Strategy:
    - Take top N chunks
    - Group by file for readability
    - Order: low-score chunks first (trimmed away), high-score chunks last
    """
    if not scored_chunks:
        return ""

    # Select top chunks within budget
    selected = []
    total_chars = 0
    seen_chunks = set()  # Avoid duplicates (method might overlap with class)

    for chunk, score in scored_chunks[:max_chunks * 2]:  # Consider more, filter dupes
        # Dedup: if we already have a class and its method, skip the method
        chunk_key = (chunk.file_path, chunk.start_line, chunk.end_line)
        if chunk_key in seen_chunks:
            continue

        # Check for overlapping ranges from same file
        skip = False
        for sel_chunk, _, _ in selected:
            if sel_chunk.file_path == chunk.file_path:
                # Check overlap
                if (chunk.start_line >= sel_chunk.start_line and
                    chunk.end_line <= sel_chunk.end_line):
                    skip = True
                    break
        if skip:
            continue

        entry_len = len(chunk.code) + len(chunk.file_path) + len(FILE_SEP_SYMBOL) + 2
        if total_chars + entry_len > max_chars:
            # Try truncating
            remaining = max_chars - total_chars
            if remaining > 300:
                truncated = chunk.code[:remaining - len(chunk.file_path) - 50]
                last_nl = truncated.rfind('\n')
                if last_nl > 0:
                    truncated = truncated[:last_nl]
                chunk_copy = CodeChunk(
                    file_path=chunk.file_path, name=chunk.name,
                    kind=chunk.kind, code=truncated,
                    start_line=chunk.start_line, end_line=chunk.end_line
                )
                selected.append((chunk_copy, score, entry_len))
            break

        seen_chunks.add(chunk_key)
        selected.append((chunk, score, entry_len))
        total_chars += entry_len

        if len(selected) >= max_chunks:
            break

    if not selected:
        return ""

    # Group chunks by file, maintaining score ordering
    # We want low-score files first, high-score files last
    file_scores = defaultdict(float)
    file_chunks = defaultdict(list)
    for chunk, score, _ in selected:
        file_scores[chunk.file_path] = max(file_scores[chunk.file_path], score)
        file_chunks[chunk.file_path].append((chunk, score))

    # Sort files by max score ascending (lowest first = trimmed first)
    sorted_files = sorted(file_scores.keys(), key=lambda f: file_scores[f])

    # Build context
    parts = []
    for file_path in sorted_files:
        file_chunk_list = file_chunks[file_path]
        # Sort chunks within file by line number for readability
        file_chunk_list.sort(key=lambda x: x[0].start_line)

        # Combine chunks from same file under one file_sep
        file_code_parts = []
        for chunk, score in file_chunk_list:
            file_code_parts.append(chunk.code)

        combined_code = '\n\n'.join(file_code_parts)
        entry = f"{FILE_SEP_SYMBOL}{file_path}\n{combined_code}"
        parts.append(entry)

    return "".join(parts)


# ────────────────────────────────────────────────────────────
# Main Processing
# ────────────────────────────────────────────────────────────

def process_datapoint(dp: dict, data_dir: str) -> dict:
    """Process a single datapoint: extract repo, chunk, rank, compose context."""
    archive_name = dp['archive']
    archive_path = os.path.join(data_dir, archive_name)

    prefix = dp['prefix']
    suffix = dp['suffix']
    target_path = dp['path']
    modified = dp.get('modified', [])

    # Try to find archive
    if not os.path.exists(archive_path):
        # Try subdirectories
        for sub in os.listdir(data_dir):
            candidate = os.path.join(data_dir, sub, archive_name)
            if os.path.exists(candidate):
                archive_path = candidate
                break
        else:
            # Try without revision suffix matching
            for f in Path(data_dir).rglob('*.zip'):
                if archive_name in f.name:
                    archive_path = str(f)
                    break

    if not os.path.exists(archive_path):
        print(f"  WARN: Archive not found: {archive_name}")
        return {"context": ""}

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(tmpdir)

        # Find root directory
        extracted = os.listdir(tmpdir)
        if len(extracted) == 1 and os.path.isdir(os.path.join(tmpdir, extracted[0])):
            root_dir = os.path.join(tmpdir, extracted[0])
        else:
            root_dir = tmpdir

        # Get all Python files
        all_files = get_all_python_files(root_dir)

        # Parse all files into chunks
        all_chunks = []
        for rel_path, content in all_files:
            chunks = parse_file_to_chunks(rel_path, content)
            all_chunks.extend(chunks)

        # Score chunks
        scored = score_chunks(all_chunks, prefix, suffix, target_path, modified)

        # Compose context
        context = compose_context(scored, max_chars=MAX_CONTEXT_CHARS)

        return {"context": context}

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"context": ""}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Task 2: Chunk-based Context Collection Pipeline")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input JSONL file")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing archive zip files")
    parser.add_argument("--output", type=str, default="predictions/python-public-smart.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS,
                       help="Maximum context characters")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS,
                       help="Maximum number of chunks to include")
    args = parser.parse_args()

    global MAX_CONTEXT_CHARS, MAX_CHUNKS
    MAX_CONTEXT_CHARS = args.max_context_chars
    MAX_CHUNKS = args.max_chunks

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with open(args.input, 'r') as f:
        datapoints = [json.loads(line) for line in f if line.strip()]

    print(f"Task 2: Chunk-based Context Collection Pipeline")
    print(f"Input: {args.input} ({len(datapoints)} datapoints)")
    print(f"Data dir: {args.data_dir}")
    print(f"Max context: {MAX_CONTEXT_CHARS} chars, Max chunks: {MAX_CHUNKS}")
    print()

    results = []
    for idx, dp in enumerate(datapoints):
        result = process_datapoint(dp, args.data_dir)
        results.append(result)

        ctx_len = len(result.get('context', ''))
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(datapoints)}] {dp['path']} -> {ctx_len} chars")

    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    # Stats
    ctx_lengths = [len(r.get('context', '')) for r in results]
    print(f"\nDone! Saved to {args.output}")
    print(f"Context lengths: avg={sum(ctx_lengths)/len(ctx_lengths):.0f}, "
          f"min={min(ctx_lengths)}, max={max(ctx_lengths)}")


if __name__ == "__main__":
    main()
