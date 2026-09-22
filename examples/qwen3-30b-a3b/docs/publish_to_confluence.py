#!/usr/bin/env python3
"""Publish the Qwen3-30B-A3B optimization write-up to Confluence via the REST API.

Confluence Cloud removed the storage-format source editor from the page menu, so
the storage XHTML has to go in over the API. This uploads the chart PNGs as page
attachments first, then writes the body that references them.

Stdlib only -- no pip install needed.

    export CONFLUENCE_EMAIL='you@example.com'
    export CONFLUENCE_API_TOKEN='...'      # id.atlassian.com/manage-profile/security/api-tokens

    python3 examples/qwen3-30b-a3b/docs/publish_to_confluence.py \
        --base-url https://your-domain.atlassian.net/wiki \
        --space PERF \
        --title 'Qwen3-30B-A3B MoE 训练性能优化'

Re-running is safe: it finds the page by title and bumps the version instead of
creating a duplicate, and attachments are replaced in place.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DOCS = Path(__file__).parent
BODY_FILE = DOCS / "qwen3-30b-a3b-perf-optimization.confluence.xhtml"
CHARTS_DIR = DOCS / "charts"


class Confluence:
    def __init__(self, base_url: str, email: str, token: str):
        self.base = base_url.rstrip("/")
        cred = base64.b64encode(f"{email}:{token}".encode()).decode()
        self.auth = f"Basic {cred}"

    def _send(self, req: urllib.request.Request):
        req.add_header("Authorization", self.auth)
        req.add_header("X-Atlassian-Token", "nocheck")
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:800]
            sys.exit(f"HTTP {exc.code} {exc.reason} on {req.get_method()} {req.full_url}\n{detail}")

    def get(self, path: str, **params):
        url = f"{self.base}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return self._send(urllib.request.Request(url, method="GET"))

    def json_call(self, method: str, path: str, payload: dict):
        req = urllib.request.Request(
            f"{self.base}{path}", data=json.dumps(payload).encode(), method=method)
        req.add_header("Content-Type", "application/json")
        return self._send(req)

    def upload(self, page_id: str, path: Path):
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
            f"Content-Type: {ctype}\r\n\r\n".encode(),
            path.read_bytes(),
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="minorEdit"\r\n\r\ntrue\r\n',
            f"--{boundary}--\r\n".encode(),
        ])
        # PUT upserts by filename; POST would 400 on the second run.
        req = urllib.request.Request(
            f"{self.base}/rest/api/content/{page_id}/child/attachment",
            data=body, method="PUT")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        return self._send(req)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="e.g. https://your-domain.atlassian.net/wiki")
    ap.add_argument("--space", help="space key; required when creating a new page")
    ap.add_argument("--title", default="Qwen3-30B-A3B MoE 训练性能优化")
    ap.add_argument("--parent-id", help="parent page id, only used on create")
    ap.add_argument("--page-id", help="update this page directly, skipping title lookup")
    ap.add_argument("--dry-run", action="store_true", help="validate inputs and exit")
    args = ap.parse_args()

    email = os.environ.get("CONFLUENCE_EMAIL")
    token = os.environ.get("CONFLUENCE_API_TOKEN")
    if not (email and token):
        sys.exit("set CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN")

    body = BODY_FILE.read_text()
    charts = sorted(CHARTS_DIR.glob("*.png"))
    missing = [f for f in charts if not f.stat().st_size]
    if not charts:
        sys.exit(f"no charts in {CHARTS_DIR}; run make_charts.py first")
    if missing:
        sys.exit(f"empty chart files: {missing}")

    referenced = {n for n in (c.name for c in charts) if f'ri:filename="{n}"' in body}
    if len(referenced) != len(charts):
        print(f"warning: {len(charts) - len(referenced)} chart(s) not referenced by the page body")

    print(f"body   : {BODY_FILE.name} ({len(body):,} chars)")
    print(f"charts : {len(charts)} PNG(s), {len(referenced)} referenced")
    if args.dry_run:
        print("dry run, nothing sent")
        return

    api = Confluence(args.base_url, email, token)

    page_id, version = args.page_id, 0
    if page_id:
        info = api.get(f"/rest/api/content/{page_id}")
        version = info["version"]["number"]
        title = info["title"]
    else:
        if not args.space:
            sys.exit("--space is required unless --page-id is given")
        hits = api.get("/rest/api/content", spaceKey=args.space,
                       title=args.title, expand="version").get("results", [])
        if hits:
            page_id = hits[0]["id"]
            version = hits[0]["version"]["number"]
            title = hits[0]["title"]
            print(f"found existing page {page_id} (v{version})")
        else:
            payload = {
                "type": "page",
                "title": args.title,
                "space": {"key": args.space},
                "body": {"storage": {"value": "<p>uploading…</p>", "representation": "storage"}},
            }
            if args.parent_id:
                payload["ancestors"] = [{"id": args.parent_id}]
            created = api.json_call("POST", "/rest/api/content", payload)
            page_id, version, title = created["id"], created["version"]["number"], created["title"]
            print(f"created page {page_id}")

    # Attachments must exist before the body's ri:attachment refs will resolve.
    for chart in charts:
        api.upload(page_id, chart)
        print(f"  uploaded {chart.name}")

    api.json_call("PUT", f"/rest/api/content/{page_id}", {
        "id": page_id,
        "type": "page",
        "title": args.title,
        "version": {"number": version + 1},
        "body": {"storage": {"value": body, "representation": "storage"}},
    })
    print(f"\ndone -> {api.base}/pages/viewpage.action?pageId={page_id}")


if __name__ == "__main__":
    main()
