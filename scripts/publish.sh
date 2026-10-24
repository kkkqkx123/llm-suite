#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

CRATES=(
    "crates/llm-suite-types"
    "crates/llm-suite-common"
    "crates/llm-suite-auth"
    "crates/llm-suite-proxy"
    "crates/llm-suite-token"
    "crates/llm-suite-message"
    "crates/llm-suite-tool-call"
    "crates/llm-suite-codec"
    "crates/llm-suite-embedding"
    "crates/llm-suite-chat-basic"
    "crates/llm-suite-rerank"
    "crates/llm-suite-client"
    "crates/llm-suite-config"
    "crates/llm-suite-gateway"
)

for crate in "${CRATES[@]}"; do
    # Package name may differ from the directory name; read it from the
    # crate's own Cargo.toml.
    name=$(sed -n 's/^name *= *"\(.*\)".*/\1/p' "$crate/Cargo.toml" | head -1)
    # Extract the package version; fall back to the workspace version
    # for crates using `version.workspace = true`.
    version=$(sed -n 's/^version *= *"\(.*\)".*/\1/p' "$crate/Cargo.toml" | head -1)
    if [[ -z "$version" ]]; then
        version=$(sed -n 's/^version *= *"\(.*\)".*/\1/p' Cargo.toml | head -1)
    fi
    echo "=== Publishing $name ==="
    # Skip if this version is already on crates.io (User-Agent is
    # required by the crates.io API).
    if curl -sf -A "llm-suite-publish-script" -o /dev/null "https://crates.io/api/v1/crates/$name/$version"; then
        echo "Skipping $name v$version: already published."
        echo ""
        continue
    fi
    set +e
    if $DRY_RUN; then
        (cd "$crate" && cargo publish --dry-run --allow-dirty --registry crates-io)
    else
        (cd "$crate" && cargo publish --allow-dirty --registry crates-io)
    fi
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        # In dry-run mode, unpublished internal dependencies cannot be
        # resolved from the real index yet; warn instead of aborting.
        if $DRY_RUN; then
            echo "WARNING: dry-run failed for $name (likely unpublished internal deps); continuing."
        else
            exit $rc
        fi
    fi
done

echo "All crates published successfully."
