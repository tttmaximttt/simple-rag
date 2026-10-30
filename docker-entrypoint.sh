#!/bin/sh
# Starts as root only to fix the ownership of /data (volumes created by older root-run images),
# then drops to the unprivileged "app" user.
set -e
if [ "$(id -u)" = 0 ]; then
  [ "$(stat -c %U /data)" = app ] || chown -R app:app /data
  exec setpriv --reuid=app --regid=app --init-groups --no-new-privs --bounding-set=-all "$@"
fi
exec "$@"
