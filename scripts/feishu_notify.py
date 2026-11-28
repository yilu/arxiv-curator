"""Send newly ranked papers to the curator owner's Feishu bot chat."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from archive_store import paper_rank_key


FEISHU_API = "https://open.feishu.cn/open-apis"
SITE_URL = "https://yilu.me/arxiv/"
MAX_TEXT_BYTES = 12_000


def format_messages(papers):
    if not papers:
        return []

    chunks = []
    body = b""
    # Reserve space for the title, part number, and site link.
    limit = MAX_TEXT_BYTES - 256
    for rank, paper in enumerate(sorted(papers, key=paper_rank_key), start=1):
        names = [author["name"] if isinstance(author, dict) else author for author in paper["authors"]]
        block = (
            f"{rank}. {paper['title']}\n{', '.join(names)}\n"
            f"Score: {paper['score']:.3f} · https://arxiv.org/abs/{paper['id']}\n\n"
            f"{paper['summary']}"
        ).encode()
        if body and len(body) + 2 + len(block) > limit:
            chunks.append(body.decode())
            body = b""
        while len(block) > limit:
            prefix = block[:limit].decode(errors="ignore")
            split_at = prefix.rfind(" ")
            if split_at >= len(prefix) // 2:
                prefix = prefix[:split_at + 1]
            chunks.append(prefix)
            block = block[len(prefix.encode()):]
        body += (b"\n\n" if body else b"") + block
    if body:
        chunks.append(body.decode())

    return [
        f"arXiv Curator · {len(papers)} new papers · {i}/{len(chunks)}\n{SITE_URL}\n\n{chunk}"
        for i, chunk in enumerate(chunks, start=1)
    ]


def post_json(path, payload, token=None):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(FEISHU_API + path, json.dumps(payload, ensure_ascii=False).encode(), headers)
    try:
        with urlopen(req, timeout=25) as res:
            data = json.load(res)
    except HTTPError as exc:
        raise RuntimeError(f"Feishu returned HTTP {exc.code} for {path}") from None
    except (URLError, TimeoutError, ValueError):
        raise RuntimeError(f"Feishu request failed for {path}") from None
    if data.get("code") != 0:
        raise RuntimeError(f"Feishu returned API code {data.get('code')} for {path}")
    return data


def tenant_access_token():
    data = post_json("/auth/v3/tenant_access_token/internal", {
        "app_id": os.environ["FEISHU_APP_ID"],
        "app_secret": os.environ["FEISHU_APP_SECRET"],
    })
    return data["tenant_access_token"]


def notify(papers, publication):
    messages = format_messages(papers)
    if not messages:
        return 0
    recipient = os.environ["FEISHU_RECIPIENT_OPEN_ID"]
    if not recipient.startswith("ou_"):
        raise ValueError("Feishu recipient must be the paired user's Open ID")
    token = tenant_access_token()
    for part, message in enumerate(messages, start=1):
        request_id = hashlib.sha256(f"{publication}:{part}:{message}".encode()).hexdigest()[:32]
        post_json("/im/v1/messages?receive_id_type=open_id", {
            "receive_id": recipient,
            "msg_type": "text",
            "content": json.dumps({"text": message}, ensure_ascii=False),
            "uuid": request_id,
        }, token)
        print(f"Feishu part {part}/{len(messages)} accepted", flush=True)
    return len(messages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers", type=Path)
    parser.add_argument("publication", help="Published site commit used for Feishu deduplication")
    args = parser.parse_args()
    papers = json.loads(args.papers.read_text(encoding="utf-8"))
    print(f"Feishu sent {notify(papers, args.publication)} messages for {len(papers)} papers")


if __name__ == "__main__":
    main()
