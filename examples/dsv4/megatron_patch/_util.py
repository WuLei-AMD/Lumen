from __future__ import annotations


def patch_file(path: str, replacements: list[tuple[str, str]]) -> bool:
    with open(path) as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False
