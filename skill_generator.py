#!/usr/bin/env python3
"""Generate platform-specific skill configurations from skills.yaml.

Usage:
    python skill_generator.py [--yaml-path PATH]
"""

import argparse
import sys

from mlsyseng_moe.skill_generator import generate_all


def main():
    parser = argparse.ArgumentParser(
        description="Generate MLSysEng MoE skill configurations for all platforms"
    )
    parser.add_argument(
        "--yaml-path",
        default="skills.yaml",
        help="Path to skills.yaml (default: skills.yaml)",
    )
    args = parser.parse_args()

    print("Generating skill configurations for all platforms...")
    results = generate_all(args.yaml_path)

    print("\nResults:")
    for platform, path in results.items():
        status = "✓" if not path.startswith("error:") else "✗"
        print(f"  {status} {platform}: {path}")

    errors = [k for k, v in results.items() if v.startswith("error:")]
    if errors:
        print(f"\n{len(errors)} platform(s) had errors (may be expected if platform not installed)")
    else:
        print("\nAll platforms configured successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
