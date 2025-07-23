import os

def read_code_files(directory: str, extensions=None):
    if extensions is None:
        extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts", ".json"]

    files = {}
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if any(fname.endswith(ext) for ext in extensions):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        files[fpath] = f.read()
                except Exception:
                    continue
    return files
