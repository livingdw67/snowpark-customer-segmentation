"""Run the test suite inside Snowflake's Python runtime (the one production uses).

Useful where scikit-learn can't load locally; CI runs the same suite on GitHub Actions.
Usage: python tests/run_in_snowflake.py
"""
import io
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snowflake.snowpark import Session  # noqa: E402
from snowflake.snowpark.types import StringType  # noqa: E402

from common import DATABASE, SPROC_PACKAGES, get_session  # noqa: E402

STAGE_PATH = f"@{DATABASE}.ML.CODE/test_bundle.zip"


def bundle() -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("segmentation", "tests"):
            for name in sorted(os.listdir(os.path.join(ROOT, folder))):
                if name.endswith(".py") and name != "run_in_snowflake.py":
                    zf.write(os.path.join(ROOT, folder, name), f"{folder}/{name}")
    buf.seek(0)
    return buf


def run_pytest(session: Session) -> str:
    import contextlib
    import sys
    import tempfile
    import zipfile

    import pytest

    import_dir = sys._xoptions["snowflake_import_directory"]
    work = tempfile.mkdtemp()
    zipfile.ZipFile(f"{import_dir}/test_bundle.zip").extractall(work)
    sys.path.insert(0, work)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = pytest.main([f"{work}/tests", "-q", "-p", "no:cacheprovider", "--color=no"])
    return f"exit code {int(code)}\n{out.getvalue()}"


if __name__ == "__main__":
    session = get_session()
    session.file.put_stream(bundle(), STAGE_PATH, auto_compress=False, overwrite=True)
    proc = session.sproc.register(
        run_pytest, return_type=StringType(), input_types=[], is_permanent=False, replace=True,
        packages=SPROC_PACKAGES + ["pytest==9.1.1"], imports=[STAGE_PATH],
    )
    report = proc()
    print(report)
    session.close()
    sys.exit(0 if report.startswith("exit code 0") else 1)
