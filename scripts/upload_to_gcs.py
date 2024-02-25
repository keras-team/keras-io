"""Script to upload post-build `site/` contents to GCS.

Prerequisite steps:

```
gcloud auth login
gcloud config set project keras-io
```

The site can be previewed at http://keras.io.storage.googleapis.com/index.html

NOTE that when previewing through the storage.googleapis.com URL,
there is no redirect to `index.html` or `404.html`; you'd have to navigate directly
to these pages. From the docs:

```
The MainPageSuffix and NotFoundPage website configurations
are only used for requests that come to Cloud Storage through a CNAME or A redirect.
For example, a request to www.example.com shows the index page,
but an equivalent request to storage.googleapis.com/www.example.com does not.
```

After upload, you may need to invalidate the CDN cache.
"""

import os
import pathlib

bucket = "keras.io"  # Bucket under `keras-io` project
scripts_path = pathlib.Path(os.path.dirname(__file__))
base_path = scripts_path.parent
site_path = base_path / "site"
site_dir = site_path.absolute()

os.system(f"gsutil -m rsync -R {site_dir} gs://{bucket}")
