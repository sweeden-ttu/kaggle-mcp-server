#!/usr/bin/env bash
# .antigravity/bootstrap.sh — auto-bootstrap launcher for kagg
#
# Detects available runtimes and launches the appropriate implementation.
# Falls through: Rust (compiled) → Ruby → Python (mlsyseng_mcp).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KAGG_ROOT="$(dirname "$SCRIPT_DIR")"

# -----------------------------------------------------------------------
# Runtime detection
# -----------------------------------------------------------------------

has_rust() {
    [ -x "$KAGG_ROOT/rust/target/release/kagg" ] || command -v cargo >/dev/null 2>&1
}

has_ruby() {
    command -v ruby >/dev/null 2>&1 && ruby -e "require 'sqlite3'" 2>/dev/null
}

has_python() {
    command -v python3 >/dev/null 2>&1
}

# -----------------------------------------------------------------------
# Build helpers
# -----------------------------------------------------------------------

build_rust() {
    if [ ! -x "$KAGG_ROOT/rust/target/release/kagg" ]; then
        echo "[antigravity] Building Rust crate..." >&2
        (cd "$KAGG_ROOT/rust" && cargo build --release 2>&1) >&2
    fi
}

# -----------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------

if has_rust; then
    build_rust
    exec "$KAGG_ROOT/rust/target/release/kagg" "$@"
fi

if has_ruby; then
    exec ruby -I"$KAGG_ROOT/ruby/lib" -rkagg -e "
      case ARGV.first
      when 'kmap'
        vars = (ARGV.index('--vars') ? ARGV[ARGV.index('--vars')+1].to_i : 4)
        mt = ARGV[ARGV.index('--minterms')+1].split(',').map(&:to_i)
        km = Kagg::KMap::Map.new(mt, vars)
        puts km.display
        puts 'Simplified: ' + Kagg::KMap.simplify(km).sop
      when 'equiv'
        a_str = ARGV[ARGV.index('--a')+1]
        b_str = ARGV[ARGV.index('--b')+1]
        ea = Kagg::KMap.parse(a_str)
        eb = Kagg::KMap.parse(b_str)
        puts Kagg::KMap.equivalence_table(ea, eb).display
      when 'stats'
        db = Kagg::Database.new
        require 'json'
        puts JSON.pretty_generate(db.stats)
      else
        STDERR.puts 'Usage: antigravity {kmap|equiv|stats} [options]'
        exit 1
      end
    " -- "$@"
fi

if has_python; then
    exec python3 -m mlsyseng_mcp.server "$@"
fi

echo "[antigravity] No supported runtime found (need cargo, ruby+sqlite3, or python3)." >&2
exit 1
