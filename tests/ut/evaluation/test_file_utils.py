# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for evaluation file safety utilities."""

import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from mindspeed_llm.tasks.evaluation import file_utils
from tests.test_tools.utils import create_testconfig


class TestEvaluationFileUtils:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def test_standardize_path_rejects_none_and_overlong_paths(self):
        with pytest.raises(TypeError, match="should not be None"):
            file_utils.standardize_path(None)

        with pytest.raises(ValueError, match="greater than 3"):
            file_utils.standardize_path("abcd", max_path_length=3)

    def test_standardize_path_accepts_length_boundary_and_resolves_path(self, tmp_path):
        relative_path = os.path.join(str(tmp_path), ".", "input.txt")

        standardized = file_utils.standardize_path(
            relative_path,
            max_path_length=len(relative_path),
        )

        assert standardized == os.path.realpath(relative_path)

    def test_standardize_path_rejects_symlink_and_can_skip_link_check(self, monkeypatch):
        fake_path = SimpleNamespace(
            normpath=lambda path: f"normalized:{path}",
            islink=lambda path: path == "normalized:linked-file",
            realpath=lambda path: f"real:{path}",
        )
        monkeypatch.setattr(file_utils, "os", SimpleNamespace(path=fake_path))

        with pytest.raises(ValueError, match="symbolic link"):
            file_utils.standardize_path("linked-file")

        assert file_utils.standardize_path("linked-file", check_link=False) == "real:linked-file"

    def test_standardize_path_rejects_real_symlink(self, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("content", encoding="utf-8")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(target)
        except (NotImplementedError, OSError):
            pytest.skip("symbolic links are not available on this platform")

        with pytest.raises(ValueError, match="symbolic link"):
            file_utils.standardize_path(str(link))

        assert file_utils.standardize_path(str(link), check_link=False) == str(target.resolve())

    def test_is_path_exists_reports_files_and_missing_paths(self, tmp_path):
        existing_path = tmp_path / "present.txt"
        existing_path.write_text("data", encoding="utf-8")

        assert file_utils.is_path_exists(str(existing_path))
        assert not file_utils.is_path_exists(str(tmp_path / "missing.txt"))

    def test_check_file_size_accepts_limit_and_rejects_larger_file(self, tmp_path):
        file_path = tmp_path / "sized.bin"
        file_path.write_bytes(b"1234")

        file_utils.check_file_size_lt(str(file_path), max_file_size=4)

        with pytest.raises(ValueError, match="greater than 3"):
            file_utils.check_file_size_lt(str(file_path), max_file_size=3)

    @pytest.mark.parametrize(
        ("effective_uid", "effective_gid"),
        test_config["test_check_owner_accepts_root_owner_or_group"],
        ids=["root", "owner", "group"],
    )
    def test_check_owner_accepts_root_owner_or_group(self, monkeypatch, effective_uid, effective_gid):
        fake_os = SimpleNamespace(
            stat=lambda _path: SimpleNamespace(st_uid=101, st_gid=202),
            geteuid=lambda: effective_uid,
            getgid=lambda: effective_gid,
        )
        monkeypatch.setattr(file_utils, "os", fake_os)

        file_utils.check_owner("input.txt")

    def test_check_owner_rejects_unrelated_user(self, monkeypatch):
        fake_os = SimpleNamespace(
            stat=lambda _path: SimpleNamespace(st_uid=101, st_gid=202),
            geteuid=lambda: 303,
            getgid=lambda: 404,
        )
        monkeypatch.setattr(file_utils, "os", fake_os)

        with pytest.raises(PermissionError, match="Access denied"):
            file_utils.check_owner("input.txt")

    @pytest.mark.parametrize("mode", test_config["test_check_other_write_permission_accepts_private_modes"])
    def test_check_other_write_permission_accepts_private_modes(self, monkeypatch, mode):
        monkeypatch.setattr(
            file_utils,
            "os",
            SimpleNamespace(stat=lambda _path: SimpleNamespace(st_mode=mode)),
        )

        file_utils.check_other_write_permission("input.txt")

    def test_check_other_write_permission_rejects_world_writable_file(self, monkeypatch):
        mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IWOTH
        monkeypatch.setattr(
            file_utils,
            "os",
            SimpleNamespace(stat=lambda _path: SimpleNamespace(st_mode=mode)),
        )

        with pytest.raises(PermissionError, match="should not be writable by others"):
            file_utils.check_other_write_permission("input.txt")

    @pytest.mark.parametrize(
        ("environment_value", "is_internal_file", "should_check"),
        test_config["test_check_path_permission_honors_environment_switch"],
    )
    def test_check_path_permission_honors_environment_switch(
        self,
        monkeypatch,
        environment_value,
        is_internal_file,
        should_check,
    ):
        calls = []
        monkeypatch.setattr(file_utils, "check_owner", lambda path: calls.append(("owner", path)))
        monkeypatch.setattr(
            file_utils,
            "check_other_write_permission",
            lambda path: calls.append(("other-write", path)),
        )
        if environment_value is None:
            monkeypatch.delenv("MINDIE_CHECK_INPUTFILES_PERMISSION", raising=False)
        else:
            monkeypatch.setenv("MINDIE_CHECK_INPUTFILES_PERMISSION", environment_value)

        file_utils.check_path_permission("input.txt", is_internal_file=is_internal_file)

        expected = [("owner", "input.txt"), ("other-write", "input.txt")] if should_check else []
        assert calls == expected

    def test_check_file_safety_handles_existing_file_and_size_checks(self, tmp_path, monkeypatch):
        file_path = tmp_path / "existing.txt"
        file_path.write_bytes(b"1234")
        permission_checks = []
        monkeypatch.setattr(file_utils, "check_path_permission", permission_checks.append)

        file_utils.check_file_safety(str(file_path), max_file_size=4)

        assert permission_checks == [str(file_path)]

        with pytest.raises(FileExistsError, match="already does"):
            file_utils.check_file_safety(str(file_path), is_exist_ok=False)

        with pytest.raises(ValueError, match="greater than 3"):
            file_utils.check_file_safety(str(file_path), max_file_size=3)

    def test_check_file_safety_can_skip_existing_file_size_check(self, tmp_path, monkeypatch):
        file_path = tmp_path / "existing.txt"
        file_path.write_text("data", encoding="utf-8")
        permission_checks = []
        monkeypatch.setattr(file_utils, "check_path_permission", permission_checks.append)
        monkeypatch.setattr(
            file_utils,
            "check_file_size_lt",
            lambda *_args, **_kwargs: pytest.fail("size check should be skipped"),
        )

        file_utils.check_file_safety(str(file_path), is_check_file_size=False)

        assert permission_checks == [str(file_path)]

    @pytest.mark.parametrize("mode", test_config["test_check_file_safety_requires_read_target_to_exist"])
    def test_check_file_safety_requires_read_target_to_exist(self, tmp_path, monkeypatch, mode):
        permission_checks = []
        monkeypatch.setattr(file_utils, "check_path_permission", permission_checks.append)
        missing_path = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError, match="does not"):
            file_utils.check_file_safety(str(missing_path), mode=mode)

        assert not permission_checks

    def test_check_file_safety_checks_parent_for_new_write_target(self, tmp_path, monkeypatch):
        permission_checks = []
        monkeypatch.setattr(file_utils, "check_path_permission", permission_checks.append)
        new_path = tmp_path / "new.txt"

        file_utils.check_file_safety(str(new_path), mode="w")

        assert permission_checks == [str(tmp_path)]

    def test_safe_open_reads_and_writes_text_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_utils, "check_path_permission", lambda _path: None)
        source_path = tmp_path / "source.txt"
        source_path.write_text("source", encoding="utf-8")

        with file_utils.safe_open(str(source_path), "r", encoding="utf-8") as source:
            assert source.read() == "source"

        output_path = tmp_path / "output.txt"
        with file_utils.safe_open(str(output_path), "w", encoding="utf-8") as output:
            assert output.write("output") == 6

        assert output_path.read_text(encoding="utf-8") == "output"

    def test_safe_open_supports_update_and_append_modes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_utils, "check_path_permission", lambda _path: None)
        file_path = tmp_path / "mutable.txt"
        file_path.write_text("first", encoding="utf-8")

        with file_utils.safe_open(str(file_path), "r+", encoding="utf-8") as stream:
            assert stream.read() == "first"
            stream.write("-second")

        with file_utils.safe_open(str(file_path), "a", encoding="utf-8") as stream:
            stream.write("-third")

        assert file_path.read_text(encoding="utf-8") == "first-second-third"

    def test_safe_open_supports_binary_modes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_utils, "check_path_permission", lambda _path: None)
        file_path = tmp_path / "payload.bin"

        with file_utils.safe_open(str(file_path), "wb") as stream:
            stream.write(b"\x00\x01")

        with file_utils.safe_open(str(file_path), "rb") as stream:
            assert stream.read() == b"\x00\x01"

    def test_safe_listdir_enforces_entry_limit(self, tmp_path):
        (tmp_path / "one.txt").write_text("1", encoding="utf-8")
        (tmp_path / "two.txt").write_text("2", encoding="utf-8")

        assert set(file_utils.safe_listdir(str(tmp_path), max_file_num=2)) == {"one.txt", "two.txt"}

        with pytest.raises(ValueError, match="exceeds the limit 1"):
            file_utils.safe_listdir(str(tmp_path), max_file_num=1)

    def test_safe_chmod_validates_canonical_path_before_changing_original(self, monkeypatch):
        calls = []
        monkeypatch.setattr(file_utils, "standardize_path", lambda path: f"canonical:{path}")
        monkeypatch.setattr(
            file_utils,
            "check_path_permission",
            lambda path: calls.append(("permission", path)),
        )
        monkeypatch.setattr(
            file_utils,
            "os",
            SimpleNamespace(chmod=lambda path, mode: calls.append(("chmod", path, mode))),
        )

        file_utils.safe_chmod("input.txt", 0o640)

        assert calls == [
            ("permission", "canonical:input.txt"),
            ("chmod", "input.txt", 0o640),
        ]

    @pytest.mark.parametrize(
        ("mode", "expected"),
        test_config["test_has_owner_write_permission"],
    )
    def test_has_owner_write_permission(self, monkeypatch, mode, expected):
        monkeypatch.setattr(
            file_utils,
            "os",
            SimpleNamespace(stat=lambda _path: SimpleNamespace(st_mode=mode)),
        )

        assert file_utils.has_owner_write_permission("input.txt") == expected

    def test_safe_readlines_accepts_limit_and_rejects_excess(self, tmp_path):
        file_path = tmp_path / "lines.txt"
        file_path.write_text("first\nsecond\n", encoding="utf-8")

        with file_path.open("r", encoding="utf-8") as stream:
            assert file_utils.safe_readlines(stream, max_line_num=2) == ["first\n", "second\n"]

        with file_path.open("r", encoding="utf-8") as stream:
            with pytest.raises(ValueError, match="exceeds the limit 1"):
                file_utils.safe_readlines(stream, max_line_num=1)
