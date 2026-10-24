#!/usr/bin/env bash
# Bump the shared version of every crate in the llm-suite workspace.
#
# All internal crates share one version (workspace.package.version) and all
# internal path dependencies reference that same version in
# [workspace.dependencies]. This script updates both in one pass.
#
# Usage:
#   scripts/bump-version.sh <new-version> [--dry-run] [--no-check]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <new-version> [--dry-run] [--no-check]" >&2
    exit 2
fi

NEW_VERSION="$1"
shift

DRY_RUN=false
RUN_CHECK=true
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --no-check) RUN_CHECK=false ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/Cargo.toml"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

# Current version = workspace.package.version
OLD_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$MANIFEST" | head -1)"
if [[ -z "$OLD_VERSION" ]]; then
    echo "Could not detect current workspace version" >&2
    exit 1
fi

if [[ "$OLD_VERSION" == "$NEW_VERSION" ]]; then
    echo "Version is already $NEW_VERSION; nothing to do."
    exit 0
fi

# Validate a simple semver-ish version string.
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
    echo "Invalid version: $NEW_VERSION" >&2
    exit 1
fi

# Apply replacement only to lines that carry a version literal. We replace the
# old version token everywhere it appears in the manifest, which covers
# workspace.package.version and every workspace.dependencies entry.
if $DRY_RUN; then
    echo "Dry run: $OLD_VERSION -> $NEW_VERSION in $MANIFEST"
    echo "----- proposed changes -----"
    diff <(cat "$MANIFEST") <(sed "s/${OLD_VERSION//./\\.}/${NEW_VERSION}/g" "$MANIFEST") || true
    echo "----- end -----"
    echo "No files modified."
    exit 0
fi

cp "$MANIFEST" "$MANIFEST.bak"
sed -i "s/${OLD_VERSION//./\\.}/${NEW_VERSION}/g" "$MANIFEST"
rm -f "$MANIFEST.bak"

echo "Bumped $OLD_VERSION -> $NEW_VERSION in $MANIFEST"

if $RUN_CHECK; then
    echo "Running cargo check --workspace to refresh Cargo.lock..."
    (cd "$ROOT" && cargo check --workspace)
fi