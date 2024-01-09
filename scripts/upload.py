import boto3
from pathlib import Path
import mimetypes
import os
from multiprocessing.pool import ThreadPool

AKEY = os.environ["AWS_S3_ACCESS_KEY"]
SKEY = os.environ["AWS_S3_SECRET_KEY"]
BUCKET = "keras.io"
USE_THREADING = True

s3 = boto3.client("s3", aws_access_key_id=AKEY, aws_secret_access_key=SKEY)


def upload_file(bucket, fpath, key_name, redirect=None):
    print(f"...Upload to {bucket}:{key_name}")
    mime = mimetypes.guess_type(fpath)[0]
    extra_args = {"ContentType": mime, "ACL": "public-read"}
    if redirect:
        extra_args["WebsiteRedirectLocation"] = redirect
    s3.upload_file(
        fpath, bucket, key_name, ExtraArgs={"ContentType": mime, "ACL": "public-read"}
    )


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


def upload_dir(directory, include_img=True):
    print(f"Uploading files from '{directory}'...")
    all_targets = []
    for dp, dn, fn in os.walk(directory):
        if fn:
            for f in fn:
                fpath = os.path.join(dp, f)
                if f.startswith("."):
                    continue
                if not include_img and "/img/" in fpath:
                    continue
                key_name = fpath[len(directory) :]
                print("> " + fpath)
                print(">>>>>> " + key_name)
                all_targets.append((BUCKET, fpath, key_name))
    if USE_THREADING:
        pool = ThreadPool(processes=8)
        pool.map(wrapped_upload_file, all_targets)
    else:
        for args in all_targets:
            wrapped_upload_file(args)


def upload_redirects(directory):
    print("Uploading redirects...")
    for dp, dn, fn in os.walk(directory):
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
    upload_dir(os.path.join(root, "site"), include_img=True)
