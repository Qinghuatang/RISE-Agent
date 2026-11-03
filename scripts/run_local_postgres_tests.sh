#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${RISE_TEST_PG_BIN:-}" ]]; then
    if command -v initdb >/dev/null 2>&1; then
        RISE_TEST_PG_BIN="$(dirname "$(command -v initdb)")"
    else
        echo "Set RISE_TEST_PG_BIN to a PostgreSQL bin directory." >&2
        exit 2
    fi
fi

for command_name in initdb pg_ctl createdb postgres; do
    if [[ ! -x "$RISE_TEST_PG_BIN/$command_name" ]]; then
        echo "Missing executable: $RISE_TEST_PG_BIN/$command_name" >&2
        exit 2
    fi
done

if [[ -n "${RISE_TEST_PG_LIBS:-}" ]]; then
    export LD_LIBRARY_PATH="${RISE_TEST_PG_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
test_python="${RISE_TEST_PYTHON:-python3}"

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/rise-postgres-test.XXXXXX")"
data_dir="$temporary_root/data"
log_file="$temporary_root/postgres.log"
db_user="$(id -un)"
db_name="rise_test_$$"
port="$("$test_python" - <<'PY'
import socket

with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY
)"
server_started=0

cleanup() {
    if [[ "$server_started" == 1 ]]; then
        "$RISE_TEST_PG_BIN/pg_ctl" -D "$data_dir" -m fast -w stop >/dev/null || true
    fi
    rm -rf "$temporary_root"
}
trap cleanup EXIT INT TERM

init_args=(
    -D "$data_dir"
    --no-locale
    --encoding=UTF8
    --auth-local=trust
    --auth-host=trust
    --username="$db_user"
)
if [[ -n "${RISE_TEST_PG_SHARE:-}" ]]; then
    init_args+=( -L "$RISE_TEST_PG_SHARE" )
fi
"$RISE_TEST_PG_BIN/initdb" "${init_args[@]}"

"$RISE_TEST_PG_BIN/pg_ctl" -D "$data_dir" -l "$log_file" \
    -o "-h 127.0.0.1 -p $port -c listen_addresses=127.0.0.1 -c unix_socket_directories='$temporary_root'" \
    -w start
server_started=1
"$RISE_TEST_PG_BIN/createdb" -h 127.0.0.1 -p "$port" -U "$db_user" "$db_name"

export RISE_TEST_POSTGRES_URL="postgresql://${db_user}@127.0.0.1:${port}/${db_name}"
echo "Using isolated PostgreSQL $("$RISE_TEST_PG_BIN/postgres" --version) on 127.0.0.1:$port"
echo "Running pytest with RISE_TEST_POSTGRES_URL set to the temporary database."

if [[ "$#" -gt 0 ]]; then
    "$test_python" -m pytest "$@"
else
    "$test_python" -m pytest -q -rs
fi
