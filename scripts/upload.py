import boto3
from pathlib import Path
import mimetypes
import hashlib
import os
import json
from multiprocessing.pool import ThreadPool

AKEY = os.environ["AWS_S3_ACCESS_KEY"]
SKEY = os.environ["AWS_S3_SECRET_KEY"]

BUCKET = "keras.io"
USE_THREADING = True
HASH_CACHE = "contents_hashes.json"

s3 = boto3.client("s3", aws_access_key_id=AKEY, aws_secret_access_key=SKEY)


def hash_file(fpath):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(fpath, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()[:8]


def upload_file(bucket, fpath, key_name, redirect=None):
    print(f"...Upload to {bucket}:{key_name}")
    mime = mimetypes.guess_type(fpath)[0]
    extra_args = {"ContentType": mime, "ACL": "public-read"}
    if redirect:
        extra_args["WebsiteRedirectLocation"] = redirect
    s3.upload_file(
        fpath, bucket, key_name, ExtraArgs={"ContentType": mime, "ACL": "public-read"}
    )


def load_hash_cache():
    try:
        s3.download_file(BUCKET, HASH_CACHE, HASH_CACHE)
    except:
        print(f"[ERROR] Could not dowload hash cache {HASH_CACHE}")
        return {}
    with open(HASH_CACHE) as f:
        contents = f.read()
        return json.loads(contents)


def save_hash_cache(hash_cache):
    with open(HASH_CACHE, "w") as f:
        f.write(json.dumps(hash_cache))
    upload_file(BUCKET, HASH_CACHE, HASH_CACHE)


def wrapped_upload_file(args):
    bucket, fpath, key_name = args
    upload_file(bucket, fpath, key_name)


def cleanup(site_directory, redirect_directory):
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=BUCKET)
    for page in page_iterator:
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".html"):
                site_fpath = os.path.join(site_directory, key)
                redirect_fpath = os.path.join(redirect_directory, key)
                if not os.path.exists(site_fpath) and not os.path.exists(
                    redirect_fpath
                ):
                    print(f"[DELETE] {key}")
                    s3.delete_object(Bucket=BUCKET, Key=key)


def upload_dir(directory, include_img=True, hash_cache=None):
    print(f"Uploading files from '{directory}'...")
    all_targets = []
    for dp, _, fn in os.walk(directory):
        if fn:
            for f in fn:
                fpath = os.path.join(dp, f)
                if f.startswith("."):
                    continue
                if not include_img and "/img/" in fpath:
                    continue
                key_name = fpath[len(directory) :]
                key_name = key_name.removeprefix("/")
                print(f"...{key_name}")
                all_targets.append((BUCKET, fpath, key_name))

    if hash_cache is not None:
        filtered_targets = []
        new_hash_cache = {}
        for bucket, fpath, key_name in all_targets:
            new_hash = hash_file(fpath)
            old_hash = hash_cache.get(key_name)
            if new_hash != old_hash:
                filtered_targets.append((bucket, fpath, key_name))
            new_hash_cache[key_name] = new_hash
        all_targets = filtered_targets

    if USE_THREADING:
        pool = ThreadPool(processes=8)
        pool.map(wrapped_upload_file, all_targets)
    else:
        for args in all_targets:
            wrapped_upload_file(args)

    if hash_cache is not None:
        return new_hash_cache


def upload_redirects(directory):
    print("Uploading redirects...")
    for dp, _, fn in os.walk(directory):
        if fn:
            for f in fn:
                fpath = os.path.join(dp, f)
                if not f == "index.html":
                    continue
                content = open(fpath).read()
                url = content[content.find("URL=") + 5 :]
                url = url[: url.find("'")]
                print(fpath)
                print(url)
                key_name = fpath[len(directory) :]
                upload_file(BUCKET, fpath, key_name, redirect=url)


if __name__ == "__main__":
    root = Path(__file__).parent.parent.resolve()
    hash_cache = load_hash_cache()
    hash_cache = upload_dir(
        os.path.join(root, "site"), include_img=True, hash_cache=hash_cache
    )
    save_hash_cache(hash_cache)
