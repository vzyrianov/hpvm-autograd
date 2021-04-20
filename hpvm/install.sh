#!/bin/bash
# Run installer script and pass on args to installer that can parse them
"${BASH_SOURCE%/*}/scripts/hpvm_installer.py" "$@"
ret_code=$?
echo "Installer returned with code $ret_code"
exit $ret_code
