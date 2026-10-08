#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

CRATES=(
    "crates/llm-types"
    "crates/llm-common"
    "crates/llm-token"
    "crates/llm-message"
    "crates/llm-tool-call"
    "crates/llm-codec"
    "crates/llm-embedding"
    "crates/llm-chat-basic"
    "crates/llm-rerank"
    "crates/llm-client"
    "crates/llm-config"
    "crates/llm-gateway"
)

for crate in "${CRATES[@]}"; do
    name=$(basename "$crate")
    echo "=== Publishing $name ==="
    if $DRY_RUN; then
        (cd "$crate" && cargo publish --dry-run --allow-dirty --registry crates-io)
    else
        (cd "$crate" && cargo publish --allow-dirty --registry crates-io)
    fi
    echo ""
done

echo "All crates published successfully."
