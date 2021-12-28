#!/usr/bin/env bash

# ref: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

DIRS="./pyccolo ./test ./examples"

function cleanup()
{
    find $DIRS -name '*.py' -print0 | xargs -0 sed -i.sedbak 's/coding: utf-8/coding: future_annotations/'
    find $DIRS -type f -name '*.sedbak' -print0 | xargs -0 rm
}

trap cleanup EXIT

find $DIRS -name '*.py' -print0 | xargs -0 sed -i.sedbak 's/coding: future_annotations/coding: utf-8/'
black $DIRS $@
