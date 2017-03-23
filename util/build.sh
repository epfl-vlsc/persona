#!/usr/bin/env bash

set -eu
set -o pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
persona_root="$DIR/.."
docker_tag="persona_shell"
docker_file="$DIR/persona_shell.docker"
if [ ! -f "$docker_file" ]; then
    echo "Can't find dockerfile: $docker_file"
    exit 1
fi

clean_python() {
    clean_root="$1"
    find "$1" -regex "\(.*__pycache__.*\|*.py[co]\)" -delete
}

clean_python "$persona_root"
docker build -t "$docker_tag" -f "$docker_file" "$persona_root"
