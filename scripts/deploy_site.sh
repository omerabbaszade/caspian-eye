#!/bin/bash
# Publishes the Caspian Eye web page to ceye.online (/var/www/ceye.online, served as static files by nginx)
# Usage: bash deploy_site.sh [source web folder, default /root/caspian-eye/web]
set -euo pipefail
SRC="${1:-/root/caspian-eye/web}"
DST=/var/www/ceye.online
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
rsync -a --exclude '*.bak*' "$SRC"/ "$TMP"/
# The source page has no doctype/head; add them for the standalone site
{
cat <<'EOF'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Caspian Eye: radar monitoring of water, wetland and coastline change in the Kura River delta, Azerbaijan, using NISAR L-band and Sentinel-1/2 data. Prototype for the 2026 NASA International Space Apps Challenge.">
<meta property="og:type" content="website">
<meta property="og:title" content="Caspian Eye: Kura Delta radar monitoring">
<meta property="og:description" content="NISAR L-band and Sentinel radar evidence of water and coastline change in the Kura River delta, 2017–2026.">
<meta property="og:url" content="https://ceye.online/">
<meta property="og:image" content="https://ceye.online/img/trend.png">
<style>html,body{margin:0}</style>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='14' fill='%230d3b4f'/%3E%3Ccircle cx='16' cy='16' r='9' fill='none' stroke='%237fd1c7' stroke-width='2'/%3E%3Ccircle cx='16' cy='16' r='3.5' fill='%237fd1c7'/%3E%3C/svg%3E">
EOF
grep -v 'viewport-fit-sides' "$SRC/index.html"
} > "$TMP/index.html"
# Source code archive: tracked and non-ignored files (images, GIFs and backups excluded)
( cd /root/caspian-eye && git ls-files --cached --others --exclude-standard \
  | grep -vE '^web/img/|^media/.*\.gif$|\.bak|^web/index\.html$' \
  | python3 -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED); [z.write(f.strip(), "caspian-eye/" + f.strip()) for f in sys.stdin if f.strip()]; z.write(sys.argv[2], "caspian-eye/web/index.html"); z.close()' "$TMP/caspian-eye-source.zip" "$TMP/index.html" )
# 30-second demo video (from media/, if present)
[ -f /root/caspian-eye/media/caspian_eye_demo.mp4 ] && cp /root/caspian-eye/media/caspian_eye_demo.mp4 "$TMP/demo.mp4"
mkdir -p "$DST"
rsync -a --delete "$TMP"/ "$DST"/
chown -R root:root "$DST"; chmod -R u=rwX,go=rX "$DST"
echo "Published: $DST ($(du -sh "$DST" | cut -f1))"
