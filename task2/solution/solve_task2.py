"""
Task 2: Chunk-based Context Collection Pipeline for Code Completion (v3).

Architecture (ASE 2025 SOTA-inspired):
1. Parse repo files into AST → semantic chunks (classes, methods, functions)
2. Multi-signal ranking: BM25 + Jaccard + TF-IDF + imports + symbols + proximity
3. Prefix-weighted query (prefix 3x more important than suffix for FIM)
4. Test/doc file deprioritization
5. Context ordering: highest relevance at END (survives left-trimming)
6. Dual-budget optimization: 8K (Mellum) tail + 16K (Codestral/Qwen) full

Key: context is trimmed from LEFT → most important content must be at END.
"""

import os
import sys
import ast
import json
import re
import math
import argparse
import zipfile
import tempfile
import shutil
from collections import defaultdict, Counter
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    os.system("pip install rank_bm25")
    from rank_bm25 import BM25Okapi

FILE_SEP_SYMBOL = "<|file_sep|>"

# Context budget: Mellum 8K tokens, Codestral/Qwen 16K tokens
# ~4 chars/token for code. We fill ~60K chars total.
# The last ~30K chars (tail) should be optimized for Mellum (8K model).
MAX_CONTEXT_CHARS = 60000
MAX_CHUNKS = 35


# ────────────────────────────────────────────────────────────
# AST-based Chunking
# ────────────────────────────────────────────────────────────

class CodeChunk:
    """A semantic unit extracted from a Python file."""
    __slots__ = ['file_path', 'name', 'kind', 'code', 'start_line', 'end_line',
                 'parent_class', 'imports_in_chunk', 'identifiers']

    def __init__(self, file_path, name, kind, code, start_line, end_line,
                 parent_class=None, imports_in_chunk=None, identifiers=None):
        self.file_path = file_path
        self.name = name
        self.kind = kind
        self.code = code
        self.start_line = start_line
        self.end_line = end_line
        self.parent_class = parent_class
        self.imports_in_chunk = imports_in_chunk or []
        self.identifiers = identifiers or set()


def extract_chunk_identifiers(code: str) -> set[str]:
    """Extract all meaningful identifiers from a code chunk."""
    idents = set()
    # Definitions
    for m in re.finditer(r'(?:def|class)\s+(\w{2,})', code):
        idents.add(m.group(1))
    # Function calls
    for m in re.finditer(r'\b(\w{3,})\s*\(', code):
        idents.add(m.group(1))
    # Attribute access
    for m in re.finditer(r'\.(\w{3,})', code):
        idents.add(m.group(1))
    # Variable assignments
    for m in re.finditer(r'\b(\w{3,})\s*=', code):
        idents.add(m.group(1))

    noise = {
        'self', 'cls', 'None', 'True', 'False', 'print', 'len', 'range', 'str',
        'int', 'float', 'list', 'dict', 'set', 'tuple', 'type', 'isinstance',
        'super', 'open', 'map', 'filter', 'zip', 'enumerate', 'sorted', 'reversed',
        'any', 'all', 'min', 'max', 'abs', 'sum', 'hash', 'repr', 'iter',
        'next', 'getattr', 'setattr', 'hasattr', 'delattr', 'property',
        'staticmethod', 'classmethod', 'return', 'yield', 'assert',
        'pass', 'break', 'continue', 'for', 'while', 'with', 'try', 'except',
        'finally', 'import', 'from', 'class', 'def', 'elif', 'else', 'and',
        'not', 'lambda', 'global', 'nonlocal', 'del', 'async', 'await',
        'raise', 'args', 'kwargs', 'result', 'value', 'name', 'key', 'item',
        'data', 'info', 'error', 'msg', 'func', 'obj', 'ret', 'val', 'tmp',
    }
    return idents - noise


def parse_file_to_chunks(file_path: str, content: str) -> list[CodeChunk]:
    """Parse a Python file into semantic chunks using AST."""
    chunks = []
    lines = content.split('\n')

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        if content.strip():
            idents = extract_chunk_identifiers(content)
            chunks.append(CodeChunk(
                file_path=file_path, name=Path(file_path).stem,
                kind='module', code=content,
                start_line=1, end_line=len(lines), identifiers=idents
            ))
        return chunks

    # Module header: imports + module-level assignments/docstrings
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
            if node.lineno <= header_end + 5:
                header_end = max(header_end, getattr(node, 'end_lineno', node.lineno))

    if header_end > 0:
        header_code = '\n'.join(lines[:header_end])
        if header_code.strip():
            chunks.append(CodeChunk(
                file_path=file_path, name=f"{Path(file_path).stem}:header",
                kind='module_header', code=header_code,
                start_line=1, end_line=header_end,
                imports_in_chunk=module_imports,
                identifiers=extract_chunk_identifiers(header_code)
            ))

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', node.lineno)
            class_code = '\n'.join(lines[start:end])

            chunks.append(CodeChunk(
                file_path=file_path, name=node.name,
                kind='class', code=class_code,
                start_line=start + 1, end_line=end,
                identifiers=extract_chunk_identifiers(class_code)
            ))

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
                        parent_class=node.name,
                        identifiers=extract_chunk_identifiers(method_code)
                    ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', node.lineno)
            func_code = '\n'.join(lines[start:end])
            chunks.append(CodeChunk(
                file_path=file_path, name=node.name,
                kind='function', code=func_code,
                start_line=start + 1, end_line=end,
                identifiers=extract_chunk_identifiers(func_code)
            ))

    if not chunks and content.strip():
        chunks.append(CodeChunk(
            file_path=file_path, name=Path(file_path).stem,
            kind='module', code=content,
            start_line=1, end_line=len(lines),
            identifiers=extract_chunk_identifiers(content)
        ))

    return chunks


# ────────────────────────────────────────────────────────────
# Import Analysis
# ────────────────────────────────────────────────────────────

def extract_imports_from_code(code: str) -> list[str]:
    """Extract import module names."""
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


def extract_imported_names(code: str) -> set[str]:
    """Extract specific names imported (from X import Y, Z)."""
    names = set()
    for line in code.split('\n'):
        line = line.strip()
        m = re.match(r'^from\s+[\w.]+\s+import\s+(.+)', line)
        if m:
            for part in m.group(1).split(','):
                name = part.strip().split(' as ')[-1].strip()
                if name and name != '*':
                    names.add(name)
    return names


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
# TF-IDF & Jaccard
# ────────────────────────────────────────────────────────────

def tokenize_code(s: str) -> list[str]:
    """Tokenize code for BM25/TF-IDF — split camelCase, snake_case."""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return re.sub(r'[^a-zA-Z0-9_]', ' ', s.lower()).split()


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_tfidf_similarity(query_tokens: list[str], chunk_tokens: list[str],
                              idf: dict[str, float]) -> float:
    """Compute TF-IDF cosine similarity between query and chunk."""
    if not query_tokens or not chunk_tokens:
        return 0.0

    # TF for query
    q_tf = Counter(query_tokens)
    q_max = max(q_tf.values())

    # TF for chunk
    c_tf = Counter(chunk_tokens)
    c_max = max(c_tf.values())

    # Compute dot product
    dot = 0.0
    q_norm = 0.0
    c_norm = 0.0

    all_terms = set(q_tf.keys()) | set(c_tf.keys())
    for term in all_terms:
        q_w = (q_tf.get(term, 0) / q_max) * idf.get(term, 0)
        c_w = (c_tf.get(term, 0) / c_max) * idf.get(term, 0)
        dot += q_w * c_w
        q_norm += q_w ** 2
        c_norm += c_w ** 2

    if q_norm == 0 or c_norm == 0:
        return 0.0
    return dot / (math.sqrt(q_norm) * math.sqrt(c_norm))


# ────────────────────────────────────────────────────────────
# File Discovery
# ────────────────────────────────────────────────────────────

def is_test_or_doc_file(path: str) -> bool:
    """Check if a file is a test or documentation file."""
    lower = path.lower()
    parts = lower.split('/')
    # Test directories/files
    if any(p in ('test', 'tests', 'testing', 'test_utils') for p in parts):
        return True
    basename = os.path.basename(lower)
    if basename.startswith('test_') or basename.endswith('_test.py'):
        return True
    # Doc/example directories
    if any(p in ('doc', 'docs', 'examples', 'example', 'tutorial', 'tutorials',
                 'benchmarks', 'bench', 'scripts', 'tools') for p in parts):
        return True
    # Setup/config files
    if basename in ('setup.py', 'conftest.py', 'conf.py', 'fabfile.py'):
        return True
    return False


def get_all_python_files(root_dir: str) -> list[tuple[str, str]]:
    """Get all Python files as (relative_path, content) pairs."""
    files = []
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.tox', 'venv', 'env',
                 '.venv', '.eggs', '.mypy_cache', '.pytest_cache', 'dist', 'build',
                 'egg-info', '.github', '.circleci'}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs and not d.startswith('.')
                       and not d.endswith('.egg-info')]
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
# Chunk Scoring & Ranking (v3: BM25 + Jaccard + TF-IDF)
# ────────────────────────────────────────────────────────────

def score_chunks(
    chunks: list[CodeChunk],
    prefix: str,
    suffix: str,
    target_path: str,
    modified_files: list[str],
) -> list[tuple[CodeChunk, float]]:
    """
    Score chunks using 9 signals:
    1. BM25 (prefix-weighted query)          — 3.0
    2. TF-IDF cosine similarity              — 2.5
    3. Jaccard on identifiers                — 3.0
    4. Import resolution                     — 8.0
    5. Imported names overlap                — 4.0
    6. Symbol/definition overlap             — 6.0
    7. Same directory proximity              — 3.0
    8. Recently modified                     — 2.0
    9. Kind/structure bonuses                — 1.5
    + Test/doc penalty                       — -3.0
    """
    if not chunks:
        return []

    # Prefix is 3x more important for FIM completion
    query_text = (prefix + " ") * 3 + suffix
    query_text_balanced = prefix + "\n" + suffix
    query_tokens = tokenize_code(query_text)
    query_tokens_balanced = tokenize_code(query_text_balanced)
    query_imports = extract_imports_from_code(query_text_balanced)
    query_imported_names = extract_imported_names(query_text_balanced)
    query_identifiers = extract_chunk_identifiers(query_text_balanced)
    target_dir = os.path.dirname(target_path)
    target_stem = Path(target_path).stem
    target_is_test = is_test_or_doc_file(target_path)

    modified_set = set(modified_files) if modified_files else set()

    # Pre-compute import-matched files
    import_matched_files = set()
    for imp in query_imports:
        for chunk in chunks:
            if module_matches_path(imp, chunk.file_path):
                import_matched_files.add(chunk.file_path)

    # Build IDF from all chunks
    doc_count = len(chunks)
    df = defaultdict(int)
    chunk_token_cache = {}
    for i, c in enumerate(chunks):
        tokens = tokenize_code(c.code)
        chunk_token_cache[i] = tokens
        for term in set(tokens):
            df[term] += 1
    idf = {term: math.log((doc_count + 1) / (freq + 1)) + 1 for term, freq in df.items()}

    # BM25 on chunks (with prefix-weighted query)
    corpus = [chunk_token_cache.get(i, []) for i in range(len(chunks))]
    valid_indices = [i for i, c in enumerate(corpus) if len(c) > 0]
    if not valid_indices:
        return []

    valid_corpus = [corpus[i] for i in valid_indices]
    valid_chunks = [chunks[i] for i in valid_indices]
    valid_token_cache = {new_i: chunk_token_cache[old_i]
                         for new_i, old_i in enumerate(valid_indices)}

    bm25 = BM25Okapi(valid_corpus)
    bm25_raw = bm25.get_scores(query_tokens)
    max_bm25 = max(bm25_raw) if len(bm25_raw) > 0 and max(bm25_raw) > 0 else 1.0

    scored = []
    for idx, (chunk, bm25_score) in enumerate(zip(valid_chunks, bm25_raw)):
        if chunk.file_path == target_path:
            continue

        score = 0.0
        chunk_tokens = valid_token_cache.get(idx, [])

        # 1. BM25 (prefix-weighted) — 3.0
        score += 3.0 * (bm25_score / max_bm25)

        # 2. TF-IDF cosine similarity — 2.5
        tfidf_sim = compute_tfidf_similarity(query_tokens_balanced, chunk_tokens, idf)
        score += 2.5 * tfidf_sim

        # 3. Jaccard on identifiers — 3.0
        jaccard = compute_jaccard(query_identifiers, chunk.identifiers)
        score += 3.0 * jaccard

        # 4. Import resolution — 8.0
        if chunk.file_path in import_matched_files:
            score += 8.0

        # 5. Imported names overlap — 4.0
        # If code does "from foo import Bar, baz" and chunk defines Bar or baz
        if query_imported_names:
            chunk_defs = set()
            for m in re.finditer(r'(?:def|class)\s+(\w+)', chunk.code):
                chunk_defs.add(m.group(1))
            imported_overlap = query_imported_names & chunk_defs
            if imported_overlap:
                score += min(4.0, len(imported_overlap) * 2.0)

        # 6. Symbol/definition overlap — 6.0
        chunk_name = chunk.name.split('.')[-1]
        if chunk_name in query_identifiers and len(chunk_name) > 3:
            score += 6.0
        elif chunk.parent_class and chunk.parent_class in query_identifiers:
            score += 4.0
        else:
            chunk_defs = set()
            for m in re.finditer(r'(?:def|class)\s+(\w{3,})', chunk.code):
                chunk_defs.add(m.group(1))
            overlap = query_identifiers & chunk_defs
            if overlap:
                score += min(5.0, len(overlap) * 1.5)

        # 7. Same directory proximity — 3.0
        chunk_dir = os.path.dirname(chunk.file_path)
        if chunk_dir == target_dir:
            score += 3.0
        elif target_dir:
            # Shared path prefix
            target_parts = target_dir.split('/')
            chunk_parts = chunk_dir.split('/')
            shared = sum(1 for a, b in zip(target_parts, chunk_parts) if a == b)
            if shared > 0:
                score += min(2.0, shared * 0.5)

        # 8. Recently modified — 2.0
        if chunk.file_path in modified_set:
            score += 2.0

        # 9. Kind/structure bonuses — 1.5
        if chunk.kind == 'module_header' and chunk.file_path in import_matched_files:
            score += 1.5
        if os.path.basename(chunk.file_path) == '__init__.py':
            init_dir = os.path.dirname(chunk.file_path)
            if target_dir.startswith(init_dir) or init_dir == target_dir:
                score += 1.0
        # Reverse dependency
        if chunk.imports_in_chunk:
            for imp in chunk.imports_in_chunk:
                if target_stem in imp.split('.'):
                    score += 1.0
                    break

        # Penalty: test/doc files (unless target is also test)
        if not target_is_test and is_test_or_doc_file(chunk.file_path):
            score -= 3.0

        # Penalty: very large chunks (>200 lines) get slight penalty
        # as they're likely noisy full classes
        chunk_lines = chunk.code.count('\n') + 1
        if chunk_lines > 200:
            score -= 0.5

        scored.append((chunk, score))

    scored.sort(key=lambda x: -x[1])
    return scored


# ────────────────────────────────────────────────────────────
# Context Composition (v3: dual-budget aware)
# ────────────────────────────────────────────────────────────

def compose_context(
    scored_chunks: list[tuple[CodeChunk, float]],
    max_chars: int = MAX_CONTEXT_CHARS,
    max_chunks: int = MAX_CHUNKS,
) -> str:
    """
    Compose context from scored chunks.

    CRITICAL: Context is trimmed from LEFT.
    → Most relevant chunks at END (survive trimming for Mellum 8K).

    Strategy:
    - Select top chunks within budget
    - Group by file, sort by line number within file
    - Files sorted: lowest-score first (trimmed first), highest last
    - Prefer methods/functions over full classes (less noise)
    """
    if not scored_chunks:
        return ""

    selected = []
    total_chars = 0
    seen_ranges = set()

    for chunk, score in scored_chunks[:max_chunks * 2]:
        chunk_key = (chunk.file_path, chunk.start_line, chunk.end_line)
        if chunk_key in seen_ranges:
            continue

        # Skip if overlaps with already-selected chunk from same file
        skip = False
        for sel_chunk, _, _ in selected:
            if sel_chunk.file_path == chunk.file_path:
                if (chunk.start_line >= sel_chunk.start_line and
                    chunk.end_line <= sel_chunk.end_line):
                    skip = True
                    break
                # Also skip if selected chunk is inside this one (prefer smaller)
                if (sel_chunk.start_line >= chunk.start_line and
                    sel_chunk.end_line <= chunk.end_line and
                    chunk.kind == 'class' and sel_chunk.kind in ('method', 'function')):
                    skip = True
                    break
        if skip:
            continue

        entry_len = len(chunk.code) + len(chunk.file_path) + len(FILE_SEP_SYMBOL) + 2
        if total_chars + entry_len > max_chars:
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
                selected.append((chunk_copy, score, len(truncated)))
            break

        seen_ranges.add(chunk_key)
        selected.append((chunk, score, entry_len))
        total_chars += entry_len

        if len(selected) >= max_chunks:
            break

    if not selected:
        return ""

    # Group by file, sort files by max score ascending (low first = trimmed first)
    file_scores = defaultdict(float)
    file_chunks = defaultdict(list)
    for chunk, score, _ in selected:
        file_scores[chunk.file_path] = max(file_scores[chunk.file_path], score)
        file_chunks[chunk.file_path].append((chunk, score))

    sorted_files = sorted(file_scores.keys(), key=lambda f: file_scores[f])

    parts = []
    for file_path in sorted_files:
        file_chunk_list = file_chunks[file_path]
        file_chunk_list.sort(key=lambda x: x[0].start_line)

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

def process_datapoint(dp: dict, data_dir: str, max_context_chars: int = 60000) -> dict:
    """Process a single datapoint."""
    archive_name = dp['archive']
    archive_path = os.path.join(data_dir, archive_name)

    prefix = dp['prefix']
    suffix = dp['suffix']
    target_path = dp['path']
    modified = dp.get('modified', [])

    # Find archive
    if not os.path.exists(archive_path):
        for sub in os.listdir(data_dir):
            candidate = os.path.join(data_dir, sub, archive_name)
            if os.path.exists(candidate):
                archive_path = candidate
                break
        else:
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

        extracted = os.listdir(tmpdir)
        if len(extracted) == 1 and os.path.isdir(os.path.join(tmpdir, extracted[0])):
            root_dir = os.path.join(tmpdir, extracted[0])
        else:
            root_dir = tmpdir

        all_files = get_all_python_files(root_dir)

        all_chunks = []
        for rel_path, content in all_files:
            chunks = parse_file_to_chunks(rel_path, content)
            all_chunks.extend(chunks)

        scored = score_chunks(all_chunks, prefix, suffix, target_path, modified)
        context = compose_context(scored, max_chars=max_context_chars)
        return {"context": context}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"context": ""}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Task 2: Context Collection Pipeline v3")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions/python-public-smart.jsonl")
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    args = parser.parse_args()

    max_context_chars = args.max_context_chars
    max_chunks = args.max_chunks

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with open(args.input, 'r') as f:
        datapoints = [json.loads(line) for line in f if line.strip()]

    print(f"Task 2: Context Collection Pipeline v3 (BM25 + TF-IDF + Jaccard)")
    print(f"Input: {args.input} ({len(datapoints)} datapoints)")
    print(f"Data dir: {args.data_dir}")
    print(f"Max context: {max_context_chars} chars, Max chunks: {max_chunks}")
    print()

    results = []
    for idx, dp in enumerate(datapoints):
        result = process_datapoint(dp, args.data_dir, max_context_chars=max_context_chars)
        results.append(result)

        ctx_len = len(result.get('context', ''))
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(datapoints)}] {dp['path']} -> {ctx_len} chars")

    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    ctx_lengths = [len(r.get('context', '')) for r in results]
    print(f"\nDone! Saved to {args.output}")
    print(f"Context lengths: avg={sum(ctx_lengths)/len(ctx_lengths):.0f}, "
          f"min={min(ctx_lengths)}, max={max(ctx_lengths)}")


if __name__ == "__main__":
    main()
