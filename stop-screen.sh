#!/bin/bash
PIDS=$(screen -ls | grep "notebook" | awk -F"." '{print $1}' | sed -E "s/\s+//g" | xargs)

if [[ -n "${PIDS}" ]]; then
    echo "Terminating process: ${PIDS}"
    kill -TERM ${PIDS}
fi
