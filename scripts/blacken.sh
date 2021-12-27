#!/usr/bin/env bash

# ref: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

find . \( -name '*.py' -or -name '*.pyi' \) -print0 | xargs -0 sed -i.sedbak 's/coding: future_annotations/coding: utf-8/'
find . -type f -name '*.sedbak' -print0 | xargs -0 rm
black .
find . \( -name '*.py' -or -name '*.pyi' \) -print0 | xargs -0 sed -i.sedbak 's/coding: utf-8/coding: future_annotations/'
find . -type f -name '*.sedbak' -print0 | xargs -0 rm
