"""
Task 2: High-Precision Context Collection Pipeline for Code Completion.

Key insights:
- Context is trimmed from LEFT -> most relevant files must be at the END
- 3 models with different context windows: Mellum 8K, Codestral/Qwen 16K tokens
- We optimize for all three by putting highest-signal content last
- Evaluation: ChrF score (character n-gram F-score)

Strategy:
1. Multi-signal file ranking (imports, BM25, path proximity, recently modified, symbol overlap)
2. Smart truncation with relevant code sections
3. Careful context budget management
4. Priority ordering: imported files > same-package files > BM25-similar files
"""

import os
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

# Budget: we want to fill context windows optimally.
# Mellum: 8K tokens (~32K chars), Codestral/Qwen: 16K tokens (~64K chars)
# Since context is trimmed from left, we can overshoot — the most important
# files at the end will survive trimming. We aim for ~60K chars which gives
# full coverage for 16K models and the most relevant tail for 8K.
MAX_CONTEXT_CHARS = 60000

# For the most important files (top tier), we keep full content
# For lower tier files, we can truncate to key sections
TOP_TIER_COUNT = 3  # Full content for top 3 files
MID_TIER_COUNT = 5  # Truncated content for next 5


def extract_imports(code: str) -> list[dict]:
    """Extract structured import info from Python code."""
    imports = []
    for line in code.split('\n'):
        line = line.strip()
        # from X import Y, Z
        m = re.match(r'^from\s+([\w.]+)\s+import\s+(.+)', line)
        if m:
            module = m.group(1)
            names = [n.strip().split(' as ')[0].strip() for n in m.group(2).split(',')]
            imports.append({'module': module, 'names': names, 'type': 'from'})
            continue
        # import X, Y
        m = re.match(r'^import\s+([\w., ]+)', line)
        if m:
            modules = [n.strip().split(' as ')[0].strip() for n in m.group(1).split(',')]
            for mod in modules:
                imports.append({'module': mod, 'names': [], 'type': 'import'})
    return imports


def extract_definitions(code: str) -> dict:
    """Extract class and function definitions with their line ranges."""
    defs = {}
    lines = code.split('\n')
    current_class = None

    for i, line in enumerate(lines):
        # Class definition
        m = re.match(r'^class\s+(\w+)', line)
        if m:
            current_class = m.group(1)
            defs[current_class] = {'type': 'class', 'line': i, 'name': current_class}

        # Function definition
        m = re.match(r'^(\s*)def\s+(\w+)', line)
        if m:
            indent = len(m.group(1))
            fname = m.group(2)
            if indent == 0:
                current_class = None
            full_name = f"{current_class}.{fname}" if current_class and indent > 0 else fname
            defs[full_name] = {'type': 'method' if indent > 0 else 'function', 'line': i, 'name': fname}

    return defs


def extract_referenced_names(code: str) -> set[str]:
    """Extract identifiers referenced in code (function calls, attribute access, class usage)."""
    names = set()
    # Function/method calls: foo(...), bar.baz(...)
    for m in re.finditer(r'\b(\w{2,})\s*\(', code):
        names.add(m.group(1))
    # Attribute access: foo.bar
    for m in re.finditer(r'(\w{2,})\.(\w{2,})', code):
        names.add(m.group(1))
        names.add(m.group(2))
    # Type hints / class references
    for m in re.finditer(r':\s*(\w{2,})', code):
        names.add(m.group(1))
    # Assignment targets that look like class instantiation
    for m in re.finditer(r'=\s*(\w{2,})\s*\(', code):
        names.add(m.group(1))

    # Remove Python builtins and keywords
    noise = {
        'self', 'cls', 'None', 'True', 'False', 'print', 'len', 'range', 'str',
        'int', 'float', 'list', 'dict', 'set', 'tuple', 'type', 'isinstance',
        'super', 'open', 'map', 'filter', 'zip', 'enumerate', 'sorted', 'reversed',
        'any', 'all', 'min', 'max', 'abs', 'sum', 'hash', 'id', 'repr', 'iter',
        'next', 'getattr', 'setattr', 'hasattr', 'delattr', 'property', 'staticmethod',
        'classmethod', 'raise', 'return', 'yield', 'assert', 'pass', 'break', 'continue',
        'for', 'while', 'with', 'try', 'except', 'finally', 'import', 'from', 'class',
        'def', 'if', 'elif', 'else', 'and', 'or', 'not', 'in', 'is', 'lambda',
        'global', 'nonlocal', 'del', 'async', 'await',
    }
    return names - noise


def get_all_python_files(root_dir: str) -> list[tuple[str, str]]:
    """Get all Python files with relative path and content."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith('.') and d not in {'__pycache__', 'node_modules', '.git', '.tox', 'venv', 'env'}]
        for fn in filenames:
            if fn.endswith('.py'):
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():  # skip empty files
                            rel = os.path.relpath(fp, root_dir)
                            files.append((rel, content))
                except:
                    pass
    return files


def tokenize_for_bm25(s: str) -> list[str]:
    """Tokenize for BM25 — split on non-alphanumeric, handle camelCase/snake_case."""
    # Split camelCase
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    # Split on non-alphanumeric
    return re.sub(r'[^a-zA-Z0-9_]', ' ', s.lower()).split()


def module_matches_path(module: str, rel_path: str) -> bool:
    """Check if an import module name matches a file path."""
    # Convert module to path-like: foo.bar.baz -> foo/bar/baz
    mod_path = module.replace('.', '/')
    clean_path = rel_path.replace('.py', '').replace('/__init__', '')

    # Exact match
    if clean_path == mod_path or clean_path.endswith('/' + mod_path):
        return True

    # Last component match (relative imports often just use the last part)
    mod_parts = module.split('.')
    path_stem = Path(rel_path).stem
    if path_stem == mod_parts[-1] and path_stem != '__init__':
        return True

    # Check if path contains the module hierarchy
    if mod_path in clean_path:
        return True

    return False


def score_and_rank_files(
    prefix: str,
    suffix: str,
    target_path: str,
    all_files: list[tuple[str, str]],
    modified_files: list[str],
) -> list[tuple[str, float, str, dict]]:
    """
    Score files with multiple signals. Returns [(rel_path, score, content, signals), ...].

    Signal weights optimized for code completion context:
    - Import resolution: 10.0 (strongest — directly imported files are most likely needed)
    - Symbol overlap: 5.0 (code that defines symbols used in prefix/suffix)
    - BM25 similarity: 3.0 (textual similarity to completion context)
    - Same directory: 3.0 (nearby files in same package)
    - Recently modified: 2.0 (developer's recent focus)
    - Path similarity: 1.0 (structural proximity in repo)
    - __init__.py in parent: 1.5 (module interface)
    """
    if not all_files:
        return []

    query_text = prefix + "\n" + suffix
    imports = extract_imports(query_text)
    referenced_names = extract_referenced_names(query_text)
    target_dir = os.path.dirname(target_path)

    # BM25
    corpus = [tokenize_for_bm25(content) for _, content in all_files]
    query_tokens = tokenize_for_bm25(query_text)

    bm25 = BM25Okapi(corpus)
    bm25_raw = bm25.get_scores(query_tokens)
    max_bm25 = max(bm25_raw) if len(bm25_raw) > 0 and max(bm25_raw) > 0 else 1.0

    # Modified files set
    modified_set = set(modified_files) if modified_files else set()

    results = []
    for i, (rel_path, content) in enumerate(all_files):
        # Skip the target file
        if rel_path == target_path:
            continue

        score = 0.0
        signals = {}

        # 1. Import resolution (weight: 10.0)
        import_score = 0.0
        for imp in imports:
            if module_matches_path(imp['module'], rel_path):
                import_score = 10.0
                break
        # Also check if first component of relative import matches
        if import_score == 0:
            for imp in imports:
                if imp['module'].startswith('.'):
                    # Relative import — resolve against target dir
                    mod_name = imp['module'].lstrip('.')
                    if mod_name:
                        candidate = os.path.join(target_dir, mod_name.replace('.', '/'))
                        clean = rel_path.replace('.py', '').replace('/__init__', '')
                        if candidate == clean or clean.endswith(candidate):
                            import_score = 10.0
                            break
        score += import_score
        signals['import'] = import_score

        # 2. Symbol overlap (weight: up to 5.0)
        file_defs = extract_definitions(content)
        defined_names = {d['name'] for d in file_defs.values()}
        overlap = referenced_names & defined_names
        if overlap:
            symbol_score = min(5.0, len(overlap) * 0.5)
            score += symbol_score
            signals['symbols'] = symbol_score

        # 3. BM25 (weight: 3.0)
        bm25_score = 3.0 * (bm25_raw[i] / max_bm25)
        score += bm25_score
        signals['bm25'] = round(bm25_score, 2)

        # 4. Same directory bonus (weight: 3.0)
        file_dir = os.path.dirname(rel_path)
        if file_dir == target_dir:
            score += 3.0
            signals['same_dir'] = 3.0
        elif target_dir and file_dir.startswith(target_dir.split('/')[0]):
            score += 1.0
            signals['near_dir'] = 1.0

        # 5. Recently modified (weight: 2.0)
        if rel_path in modified_set:
            score += 2.0
            signals['modified'] = 2.0
        # Also check basename match for modified files
        elif any(os.path.basename(m) == os.path.basename(rel_path) for m in modified_set):
            score += 1.0
            signals['modified_basename'] = 1.0

        # 6. Path similarity (weight: 1.0)
        target_parts = Path(target_path).parts
        file_parts = Path(rel_path).parts
        shared = sum(1 for a, b in zip(target_parts, file_parts) if a == b)
        path_sim = shared / max(len(target_parts), len(file_parts)) if target_parts else 0
        score += path_sim
        signals['path_sim'] = round(path_sim, 2)

        # 7. __init__.py of parent package (weight: 1.5)
        if os.path.basename(rel_path) == '__init__.py':
            init_dir = os.path.dirname(rel_path)
            if target_dir.startswith(init_dir) or init_dir == target_dir:
                score += 1.5
                signals['init_parent'] = 1.5

        # 8. Reverse dependency: file imports the target module
        target_module = target_path.replace('/', '.').replace('.py', '')
        target_stem = Path(target_path).stem
        file_imports = extract_imports(content)
        for fi in file_imports:
            if target_stem in fi['module'].split('.') or module_matches_path(fi['module'], target_path):
                score += 1.0
                signals['reverse_dep'] = 1.0
                break

        results.append((rel_path, score, content, signals))

    # Sort by score descending
    results.sort(key=lambda x: -x[1])
    return results


def smart_truncate_file(content: str, referenced_names: set, max_chars: int = 8000) -> str:
    """Truncate file content keeping the most relevant sections."""
    if len(content) <= max_chars:
        return content

    lines = content.split('\n')

    # Always keep: imports, class/function definitions that match referenced names
    keep_ranges = []

    # Keep first 10 lines (imports, module docstring)
    keep_ranges.append((0, min(10, len(lines))))

    # Keep definitions that are referenced
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)(class|def)\s+(\w+)', line)
        if m:
            name = m.group(3)
            if name in referenced_names:
                # Keep this definition + some body (up to 30 lines)
                start = max(0, i - 1)
                end = min(len(lines), i + 30)
                keep_ranges.append((start, end))

    if not keep_ranges:
        # No specific matches — keep head and tail
        head = '\n'.join(lines[:max_chars // 160])
        tail_lines = max_chars // 160
        tail = '\n'.join(lines[-tail_lines:]) if tail_lines > 0 else ''
        return head + '\n# ... truncated ...\n' + tail

    # Merge overlapping ranges
    keep_ranges.sort()
    merged = [keep_ranges[0]]
    for start, end in keep_ranges[1:]:
        if start <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Build truncated content
    parts = []
    for start, end in merged:
        parts.append('\n'.join(lines[start:end]))

    result = '\n# ...\n'.join(parts)
    if len(result) > max_chars:
        result = result[:max_chars]
        last_nl = result.rfind('\n')
        if last_nl > max_chars * 0.8:
            result = result[:last_nl]

    return result


def compose_context(
    ranked_files: list[tuple[str, float, str, dict]],
    prefix: str,
    suffix: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Compose context from ranked files.

    CRITICAL: Context is trimmed from LEFT by the evaluation system.
    So we put MOST RELEVANT files at the END (they survive trimming).

    Layout:
    [low-relevance files ... | mid-relevance files ... | HIGH-relevance files]
    ^--- trimmed first                                   kept for all models ---^
    """
    referenced_names = extract_referenced_names(prefix + "\n" + suffix)

    # Separate files into tiers
    top_tier = ranked_files[:TOP_TIER_COUNT]
    mid_tier = ranked_files[TOP_TIER_COUNT:TOP_TIER_COUNT + MID_TIER_COUNT]
    low_tier = ranked_files[TOP_TIER_COUNT + MID_TIER_COUNT:]

    # Build context parts (will be assembled: low -> mid -> top, so top is at END)
    parts_low = []
    parts_mid = []
    parts_top = []

    total_chars = 0

    # Low tier: aggressively truncated, filled first (will be trimmed away for small models)
    for rel_path, score, content, signals in low_tier:
        if total_chars >= max_chars:
            break
        truncated = smart_truncate_file(content, referenced_names, max_chars=3000)
        entry = f"{FILE_SEP_SYMBOL}{rel_path}\n{truncated}"
        if total_chars + len(entry) > max_chars:
            break
        parts_low.append(entry)
        total_chars += len(entry)

    # Mid tier: moderately truncated
    for rel_path, score, content, signals in mid_tier:
        if total_chars >= max_chars:
            break
        truncated = smart_truncate_file(content, referenced_names, max_chars=6000)
        entry = f"{FILE_SEP_SYMBOL}{rel_path}\n{truncated}"
        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 800:
                truncated2 = truncated[:remaining - len(rel_path) - 50]
                last_nl = truncated2.rfind('\n')
                if last_nl > 0:
                    truncated2 = truncated2[:last_nl]
                entry = f"{FILE_SEP_SYMBOL}{rel_path}\n{truncated2}"
                parts_mid.append(entry)
            break
        parts_mid.append(entry)
        total_chars += len(entry)

    # Top tier: full content (these survive trimming and are closest to prefix/suffix)
    for rel_path, score, content, signals in top_tier:
        if total_chars >= max_chars:
            break
        entry = f"{FILE_SEP_SYMBOL}{rel_path}\n{content}"
        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 500:
                truncated = smart_truncate_file(content, referenced_names, max_chars=remaining - len(rel_path) - 50)
                entry = f"{FILE_SEP_SYMBOL}{rel_path}\n{truncated}"
                parts_top.append(entry)
            break
        parts_top.append(entry)
        total_chars += len(entry)

    # Assemble: low (trimmed first) -> mid -> top (trimmed last, closest to completion)
    # Top tier is REVERSED so #1 best file is very last (closest to prefix/suffix)
    all_parts = parts_low + parts_mid + list(reversed(parts_top))
    return "".join(all_parts)


def process_datapoint(dp: dict, data_dir: str) -> dict:
    """Process a single datapoint and return prediction."""
    repo_name = dp['archive'].replace('.zip', '')
    archive_path = os.path.join(data_dir, dp['archive'])

    prefix = dp['prefix']
    suffix = dp['suffix']
    target_path = dp['path']
    modified = dp.get('modified', [])

    # Extract repo from archive to temp dir
    if os.path.exists(archive_path):
        tmpdir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(tmpdir)

            # Find the actual root (might be nested in a folder)
            extracted = os.listdir(tmpdir)
            if len(extracted) == 1 and os.path.isdir(os.path.join(tmpdir, extracted[0])):
                root_dir = os.path.join(tmpdir, extracted[0])
            else:
                root_dir = tmpdir

            # Get all Python files
            all_files = get_all_python_files(root_dir)

            # Score and rank
            ranked = score_and_rank_files(prefix, suffix, target_path, all_files, modified)

            # Compose context
            context = compose_context(ranked, prefix, suffix)

            return {"context": context}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        # Try pre-extracted directory
        repo_path = dp['repo'].replace("/", "__")
        repo_revision = dp['revision']
        root_dir = os.path.join(data_dir, f"repositories-python-test",
                               f"{repo_path}-{repo_revision}")

        if not os.path.exists(root_dir):
            print(f"  WARN: No archive or extracted dir found")
            return {"context": ""}

        all_files = get_all_python_files(root_dir)
        ranked = score_and_rank_files(prefix, suffix, target_path, all_files, modified)
        context = compose_context(ranked, prefix, suffix)
        return {"context": context}


def main():
    parser = argparse.ArgumentParser(description="Task 2: Context Collection Pipeline")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input JSONL file (e.g., python-test.jsonl)")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing archive files")
    parser.add_argument("--output", type=str, default="predictions/python-test-smart.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS,
                       help="Maximum context characters")
    args = parser.parse_args()

    global MAX_CONTEXT_CHARS
    MAX_CONTEXT_CHARS = args.max_context_chars

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load datapoints
    with open(args.input, 'r') as f:
        datapoints = [json.loads(line) for line in f if line.strip()]

    print(f"Processing {len(datapoints)} datapoints from {args.input}")
    print(f"Data directory: {args.data_dir}")
    print(f"Max context chars: {MAX_CONTEXT_CHARS}")

    results = []
    for idx, dp in enumerate(datapoints):
        result = process_datapoint(dp, args.data_dir)
        results.append(result)

        if (idx + 1) % 10 == 0 or idx == 0:
            ctx_len = len(result.get('context', ''))
            print(f"  [{idx+1}/{len(datapoints)}] {dp['path']} -> context: {ctx_len} chars")

    # Write predictions
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\nDone! Predictions saved to {args.output}")
    print(f"Total: {len(results)} predictions")


if __name__ == "__main__":
    main()
