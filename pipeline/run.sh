#!/bin/sh
# Rebuild every table and the site's data file from the raw exports.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY parse.py
$PY parse_dms.py
$PY embed.py
$PY classify.py
$PY analyze.py > /dev/null
echo "site/data.js rebuilt"
