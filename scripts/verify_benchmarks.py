#!/usr/bin/env python3
"""Verification script to test benchmark suite installation and imports.

This script verifies that:
1. All benchmark modules can be imported
2. Registry is properly configured
3. All benchmarks are registered
4. Configuration files are valid
"""

import sys
from pathlib import Path


def verify_imports():
    """Verify all benchmark imports."""
    print("=" * 80)
    print("Verifying Benchmark Suite Imports")
    print("=" * 80)

    try:
        # Core imports
        print("\n1. Testing core imports...")
        from abprop.benchmarks import (
            Benchmark,
            BenchmarkConfig,
            BenchmarkRegistry,
            BenchmarkResult,
            get_registry,
        )
        print("   ✓ Core classes imported successfully")

        # Registry
        print("\n2. Testing registry...")
        registry = get_registry()
        print(f"   ✓ Registry instantiated: {registry}")

        # List benchmarks
        print("\n3. Checking registered benchmarks...")
        available = registry.list_benchmarks()
        print(f"   Available benchmarks: {available}")

        expected = [
            "perplexity",
            "cdr_classification",
            "liability",
            "developability",
            "zero_shot",
        ]

        for bench in expected:
            if bench in available:
                print(f"   ✓ {bench}")
            else:
                print(f"   ✗ {bench} - NOT FOUND")
                return False

        # Test benchmark creation
        print("\n4. Testing benchmark instantiation...")
        config = BenchmarkConfig(
            data_path=Path("data/processed/oas"),
            batch_size=32,
        )

        for bench_name in available:
            try:
                benchmark = registry.create(bench_name, config)
                print(f"   ✓ {bench_name}: {benchmark.__class__.__name__}")
            except Exception as e:
                print(f"   ✗ {bench_name}: Failed - {e}")
                return False

        print("\n" + "=" * 80)
        print("✓ All verifications passed!")
        print("=" * 80)
        return True

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nPossible solutions:")
        print("1. Ensure you're in the correct environment")
        print("2. Install AbProp: pip install -e .")
        print("3. Check PYTHONPATH includes src/")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_config_files():
    """Verify configuration files exist and are valid."""
    print("\n" + "=" * 80)
    print("Verifying Configuration Files")
    print("=" * 80)

    config_files = [
        "configs/benchmarks.yaml",
        "configs/model.yaml",
        "configs/data.yaml",
    ]

    all_exist = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"   ✓ {config_file}")
        else:
            print(f"   ✗ {config_file} - NOT FOUND")
            all_exist = False

    if all_exist:
        # Try loading benchmark config
        try:
            from abprop.utils import load_yaml_config

            config = load_yaml_config(Path("configs/benchmarks.yaml"))
            print(f"\n   ✓ Benchmark config loaded: {len(config)} entries")
        except Exception as e:
            print(f"\n   ✗ Failed to load config: {e}")
            return False

    return all_exist


def verify_scripts():
    """Verify benchmark scripts exist."""
    print("\n" + "=" * 80)
    print("Verifying Benchmark Scripts")
    print("=" * 80)

    scripts = [
        "scripts/run_benchmarks.py",
        "examples/run_benchmark_example.py",
    ]

    all_exist = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            print(f"   ✓ {script}")
        else:
            print(f"   ✗ {script} - NOT FOUND")
            all_exist = False

    return all_exist


def verify_documentation():
    """Verify documentation files exist."""
    print("\n" + "=" * 80)
    print("Verifying Documentation")
    print("=" * 80)

    docs = [
        "src/abprop/benchmarks/README.md",
        "docs/README.md",
        "docs/RESULTS.md",
        "docs/CASE_STUDIES.md",
    ]

    all_exist = True
    for doc in docs:
        path = Path(doc)
        if path.exists():
            print(f"   ✓ {doc}")
        else:
            print(f"   ✗ {doc} - NOT FOUND")
            all_exist = False

    return all_exist


def print_usage_examples():
    """Print example usage commands."""
    print("\n" + "=" * 80)
    print("Example Usage")
    print("=" * 80)

    examples = [
        ("Run all benchmarks", "python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --all"),
        ("Run specific benchmark", "python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --benchmarks perplexity"),
        ("Quick test", "python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --benchmarks perplexity --max-samples 100"),
        ("Generate HTML report", "python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --all --html-report"),
        ("Programmatic usage", "python examples/run_benchmark_example.py"),
    ]

    for desc, cmd in examples:
        print(f"\n{desc}:")
        print(f"  {cmd}")


def main():
    """Run all verifications."""
    print("\n" + "=" * 80)
    print("AbProp Benchmark Suite Verification")
    print("=" * 80)
    print()

    results = []

    # Run verifications
    results.append(("Imports", verify_imports()))
    results.append(("Config Files", verify_config_files()))
    results.append(("Scripts", verify_scripts()))
    results.append(("Documentation", verify_documentation()))

    # Print summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n" + "=" * 80)
        print("✓ All verifications passed! Benchmark suite is ready to use.")
        print("=" * 80)
        print_usage_examples()
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ Some verifications failed. Please check the errors above.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
