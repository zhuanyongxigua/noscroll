"""Real integration tests for NoScroll CLI.

These tests actually execute the full workflow (not dry-run).
They use --serial mode to avoid rate limiting.

Run with: pytest tests/integration/test_real_run.py -v -s
"""

import pytest
import sys
import os
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from noscroll.cli import main


# Test output directory
TEST_OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_outputs"


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    """Create and clean test output directory."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Keep outputs for inspection after tests


class TestRealRun:
    """Real integration tests that actually fetch and process data.
    
    These tests run serially to avoid rate limiting.
    Each test depends on the previous one completing.
    """

    def test_01_last_1d_bucket_day_all_sources(self):
        """Test 1: 过去1天所有资源，每天一份
        
        Command: noscroll run --last 1d --bucket day --serial --delay 1000 --debug
        """
        output_dir = TEST_OUTPUT_DIR / "test1_daily"
        
        result = main([
            "run",
            "--last", "1d",
            "--bucket", "day",
            "--out", str(output_dir),
            "--serial",
            "--delay", "1000",
            "--debug",
        ])
        
        assert result == 0, "Command should succeed"
        assert output_dir.exists(), f"Output directory should exist: {output_dir}"
        
        # Check that files were created
        md_files = list(output_dir.glob("*.md"))
        print(f"\nTest 1 output: {output_dir}")
        print(f"Files created: {len(md_files)}")
        for f in md_files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        
        assert len(md_files) > 0, "Should create at least one output file"

    def test_02_last_5d_single_output_all_sources(self):
        """Test 2: 过去5天所有资源，合并成一份
        
        Command: noscroll run --last 5d --serial --delay 1000
        """
        output_file = TEST_OUTPUT_DIR / "test2_all.md"
        
        result = main([
            "run",
            "--last", "5d",
            "--out", str(output_file),
            "--serial",
            "--delay", "1000",
        ])
        
        assert result == 0, "Command should succeed"
        assert output_file.exists(), f"Output file should exist: {output_file}"
        
        size = output_file.stat().st_size
        print(f"\nTest 2 output: {output_file}")
        print(f"File size: {size} bytes")
        
        # Read and show first few lines
        content = output_file.read_text(encoding="utf-8")
        lines = content.split("\n")[:10]
        print("First 10 lines:")
        for line in lines:
            print(f"  {line}")
        
        assert size > 0, "Output file should not be empty"

    def test_03_last_5d_hn_bucket_day_chinese(self):
        """Test 3: 过去5天 HN，每天一份，中文输出
        
        Command: noscroll run --last 5d --source-types hn --bucket day --lang zh --serial --delay 1000
        """
        output_dir = TEST_OUTPUT_DIR / "test3_hn_daily_zh"
        
        result = main([
            "run",
            "--last", "5d",
            "--source-types", "hn",
            "--bucket", "day",
            "--lang", "zh",
            "--out", str(output_dir),
            "--serial",
            "--delay", "1000",
        ])
        
        assert result == 0, "Command should succeed"
        assert output_dir.exists(), f"Output directory should exist: {output_dir}"
        
        md_files = list(output_dir.glob("*.md"))
        print(f"\nTest 3 output: {output_dir}")
        print(f"Files created: {len(md_files)}")
        for f in md_files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        
        assert len(md_files) > 0, "Should create at least one output file"

    def test_04_last_5d_rss_single_output(self):
        """Test 4: 过去5天所有 RSS，合并一份
        
        Command: noscroll run --last 5d --source-types rss --serial --delay 1000
        """
        output_file = TEST_OUTPUT_DIR / "test4_rss.md"
        
        result = main([
            "run",
            "--last", "5d",
            "--source-types", "rss",
            "--out", str(output_file),
            "--serial",
            "--delay", "1000",
        ])
        
        assert result == 0, "Command should succeed"
        assert output_file.exists(), f"Output file should exist: {output_file}"
        
        size = output_file.stat().st_size
        print(f"\nTest 4 output: {output_file}")
        print(f"File size: {size} bytes")
        
        content = output_file.read_text(encoding="utf-8")
        lines = content.split("\n")[:10]
        print("First 10 lines:")
        for line in lines:
            print(f"  {line}")
        
        assert size > 0, "Output file should not be empty"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])
