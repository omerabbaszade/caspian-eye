# Helper to open remote NISAR GCOV HDF5 files by byte range; the token is read from a file and never printed
import requests, fsspec, h5py
import os
TOKEN = open(os.environ.get("EARTHDATA_TOKEN_FILE", os.path.expanduser("~/.earthdata_token"))).read().strip()  # the token lives in a file, never in the code
def signed_url(url):
    """Resolve the ASF redirect with an EDL token and return the signed CloudFront URL."""
    r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}", "Range": "bytes=0-0"}, allow_redirects=False, timeout=60)
    hops = 0
    while r.status_code in (301, 302, 303, 307, 308) and hops < 5:
        loc = r.headers["Location"]
        if "cloudfront.net" in loc or "amazonaws.com" in loc:
            return loc
        r = requests.get(loc, headers={"Authorization": f"Bearer {TOKEN}", "Range": "bytes=0-0"}, allow_redirects=False, timeout=60); hops += 1
    return url
def open_h5(url, block=8 * 2**20):
    su = signed_url(url)
    fs = fsspec.filesystem("http")
    f = fs.open(su, "rb", block_size=block, cache_type="blockcache")
    return h5py.File(f, "r")
