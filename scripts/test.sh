#!/bin/bash
# Runs the unit tests. With only the Command Line Tools, Swift Testing lives outside the default search paths.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/check-agents.py
D=${DEVELOPER_DIR:-$(xcode-select -p)}
F=$D/Library/Developer/Frameworks
L=$D/Library/Developer/usr/lib
extra=()
P=$(dirname "$(xcrun --find swiftc)")/../lib/swift/host/plugins/testing
if [[ -f "$P/libTestingMacros.dylib" && -f "$F/_Testing_Foundation.framework/_Testing_Foundation" && ! -d "$F/_Testing_Foundation.framework/Modules" ]]; then
    # Some CLT releases ship the overlay binary but not its Swift module.
    extra+=(--disable-xctest -Xswiftc -Xfrontend -Xswiftc -disable-cross-import-overlays -Xswiftc -plugin-path -Xswiftc "$P")
fi
exec swift test -Xswiftc -F -Xswiftc "$F" -Xlinker -F -Xlinker "$F" -Xlinker -rpath -Xlinker "$F" -Xlinker -rpath -Xlinker "$L" ${extra[@]+"${extra[@]}"} "$@"
