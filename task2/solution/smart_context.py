"""
Task 2: Smart Context Collection Pipeline
Strategy: Multi-signal file ranking + intelligent context composition

Signals for ranking files:
1. BM25 similarity to prefix+suffix
2. Import analysis (files imported by the completion file)
3. Recently modified files
4. Same-directory proximity
5. File name similarity to completion file
6. Structural analysis (classes/functions referenced in prefix/suffix)
"""

import os
import re
import json
import argparse
import jsonlines
from collections import defaultdict
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    os.system("pip install rank_bm25")
    from rank_bm25 import BM25Okapi

FILE_SEP_SYMBOL = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

# Maximum context budget (characters) - we want to stay well within token limits
# Mellum: 8K tokens, Codestral/Qwen: 16K tokens
# ~4 chars per token on average for code
MAX_CONTEXT_CHARS = 50000  # ~12.5K tokens, safe for all models


def extract_imports(code: str, language: str = "python") -> list[str]:
    """Extract imported module names from Python code."""
    imports = []
    if language == "python":
        # import foo, from foo import bar
        for match in re.finditer(r'^\s*(?:from\s+(\S+)|import\s+(\S+))', code, re.MULTILINE):
            mod = match.group(1) or match.group(2)
            if mod:
                # Get top-level module
                parts = mod.split('.')
                imports.append(parts[0])
                # Also keep full path for local imports
                imports.append(mod.replace('.', '/'))
    return imports


def extract_identifiers(code: str) -> set[str]:
    """Extract identifiers (class/function names) from code."""
    # Find class and function definitions and usages
    idents = set()
    # Definitions
    for match in re.finditer(r'\b(?:class|def)\s+(\w+)', code):
        idents.add(match.group(1))
    # Function calls
    for match in re.finditer(r'\b(\w{3,})\s*\(', code):
        idents.add(match.group(1))
    # Attribute access
    for match in re.finditer(r'\.(\w{3,})', code):
        idents.add(match.group(1))
    # Remove common Python builtins
    builtins = {'self', 'None', 'True', 'False', 'print', 'len', 'range', 'str', 'int',
                'float', 'list', 'dict', 'set', 'tuple', 'type', 'isinstance', 'super',
                'return', 'yield', 'import', 'from', 'class', 'def', 'for', 'while', 'if',
                'else', 'elif', 'try', 'except', 'finally', 'with', 'as', 'not', 'and', 'or'}
    return idents - builtins


def get_all_code_files(root_dir: str, extension: str = ".py", min_lines: int = 3) -> dict[str, str]:
    """Get all code files in the repository."""
    files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden dirs, __pycache__, .git, etc.
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__' and d != 'node_modules']
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if len(content.split('\n')) >= min_lines:
                            rel_path = os.path.relpath(file_path, root_dir)
                            files[rel_path] = content
                except:
                    pass
    return files


def score_files(
    repo_files: dict[str, str],
    completion_path: str,
    prefix: str,
    suffix: str,
    modified_files: list[str],
    language: str = "python"
) -> list[tuple[str, float, str]]:
    """Score and rank all files by relevance. Returns [(path, score, content), ...]."""

    if not repo_files:
        return []

    extension = ".py" if language == "python" else ".kt"
    query_text = prefix + " " + suffix
    completion_dir = os.path.dirname(completion_path)

    # Extract signals from the completion file
    imports = extract_imports(prefix + "\n" + suffix, language)
    identifiers = extract_identifiers(prefix + "\n" + suffix)

    # Prepare BM25
    def tokenize(s: str) -> list[str]:
        return re.sub(r'[^a-zA-Z0-9_]', ' ', s.lower()).split()

    file_paths = list(repo_files.keys())
    file_contents = [repo_files[p] for p in file_paths]

    corpus = [tokenize(c) for c in file_contents]
    query_tokens = tokenize(query_text)

    # BM25 scores
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query_tokens)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

    scores = []
    for i, (path, content) in enumerate(zip(file_paths, file_contents)):
        score = 0.0

        # Skip the completion file itself
        if path == completion_path:
            continue

        # 1. BM25 similarity (weight: 3.0)
        score += 3.0 * (bm25_scores[i] / max_bm25)

        # 2. Import analysis (weight: 5.0) - very strong signal
        path_parts = path.replace(extension, '').replace('/', '.').replace('\\', '.')
        path_stem = Path(path).stem
        for imp in imports:
            if path_stem == imp.split('.')[-1]:
                score += 5.0
                break
            if imp in path_parts or imp.replace('.', '/') in path:
                score += 4.0
                break

        # 3. Recently modified files (weight: 3.0)
        if path in modified_files:
            score += 3.0
        # Also check by filename match
        for mod_file in modified_files:
            if os.path.basename(mod_file) == os.path.basename(path):
                score += 2.0
                break

        # 4. Same directory proximity (weight: 2.0)
        if os.path.dirname(path) == completion_dir:
            score += 2.0
        elif completion_dir and path.startswith(completion_dir.split('/')[0] + '/'):
            score += 0.5

        # 5. Identifier overlap (weight: 2.0)
        file_idents = extract_identifiers(content)
        overlap = len(identifiers & file_idents)
        if overlap > 0:
            score += min(2.0, overlap * 0.2)

        # 6. Test file heuristic
        if 'test' in path.lower() and 'test' in completion_path.lower():
            score += 1.0
        elif 'test' not in path.lower() and 'test' not in completion_path.lower():
            score += 0.5

        # 7. __init__.py files are often useful for understanding module structure
        if os.path.basename(path) == '__init__.py':
            if os.path.dirname(path) in completion_dir or completion_dir.startswith(os.path.dirname(path)):
                score += 1.5

        scores.append((path, score, content))

    # Sort by score descending
    scores.sort(key=lambda x: -x[1])
    return scores


def compose_context(
    ranked_files: list[tuple[str, float, str]],
    prefix: str,
    suffix: str,
    max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """Compose context from ranked files, fitting within budget."""
    context_parts = []
    total_chars = 0

    for path, score, content in ranked_files:
        # Format this file's contribution
        entry = FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL,
            file_name=path,
            file_content=content
        )

        if total_chars + len(entry) > max_chars:
            # Try to include a truncated version
            remaining = max_chars - total_chars
            if remaining > 500:  # Worth including partial
                truncated_content = content[:remaining - len(path) - 50]
                # Try to cut at a line boundary
                last_newline = truncated_content.rfind('\n')
                if last_newline > 0:
                    truncated_content = truncated_content[:last_newline]
                entry = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL,
                    file_name=path,
                    file_content=truncated_content
                )
                context_parts.append(entry)
            break
        else:
            context_parts.append(entry)
            total_chars += len(entry)

    return "".join(context_parts)


def smart_prefix_suffix(prefix: str, suffix: str) -> tuple[str, str]:
    """Optionally trim prefix/suffix to focus on the most relevant parts."""
    # Keep full prefix (most recent context matters for FIM)
    # Trim suffix if very long - keep first 30 lines
    suffix_lines = suffix.split('\n')
    if len(suffix_lines) > 30:
        suffix = '\n'.join(suffix_lines[:30])
    return prefix, suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="practice")
    parser.add_argument("--lang", type=str, default="python")
    args = parser.parse_args()

    stage = args.stage
    language = args.lang
    extension = ".py" if language == "python" else ".kt"

    completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")
    predictions_file = os.path.join("predictions", f"{language}-{stage}-smart.jsonl")

    os.makedirs("predictions", exist_ok=True)

    print(f"Running smart context pipeline for {language}-{stage}")

    with jsonlines.open(completion_points_file, 'r') as reader:
        datapoints = list(reader)

    print(f"Processing {len(datapoints)} datapoints...")

    results = []
    for idx, dp in enumerate(datapoints):
        repo_path = dp['repo'].replace("/", "__")
        repo_revision = dp['revision']
        root_dir = os.path.join("data", f"repositories-{language}-{stage}", f"{repo_path}-{repo_revision}")

        if not os.path.exists(root_dir):
            print(f"  [{idx+1}] WARN: repo dir not found: {root_dir}")
            results.append({"context": ""})
            continue

        # Get all code files
        repo_files = get_all_code_files(root_dir, extension)

        # Score and rank files
        ranked = score_files(
            repo_files,
            completion_path=dp['path'],
            prefix=dp['prefix'],
            suffix=dp['suffix'],
            modified_files=dp.get('modified', []),
            language=language
        )

        # Compose context from top files
        context = compose_context(ranked, dp['prefix'], dp['suffix'])

        # Optionally adjust prefix/suffix
        new_prefix, new_suffix = smart_prefix_suffix(dp['prefix'], dp['suffix'])

        submission = {"context": context}
        if new_prefix != dp['prefix']:
            submission["prefix"] = new_prefix
        if new_suffix != dp['suffix']:
            submission["suffix"] = new_suffix

        results.append(submission)

        if (idx + 1) % 10 == 0 or idx == 0:
            top3 = [(p, f"{s:.2f}") for p, s, _ in ranked[:3]]
            print(f"  [{idx+1}/{len(datapoints)}] {dp['path']} -> top files: {top3}")

    # Write predictions
    with jsonlines.open(predictions_file, 'w') as writer:
        for r in results:
            writer.write(r)

    print(f"\nPredictions saved to {predictions_file}")
    print(f"Total predictions: {len(results)}")


if __name__ == "__main__":
    main()
