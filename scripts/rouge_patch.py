"""Make pyrouge work on Windows with Cygwin Perl.

Windows cannot CreateProcess a `.pl` file directly (WinError 193), and the
ROUGE-1.5.5 binary is a Perl script. This module monkeypatches the
`check_output` symbol inside pyrouge.Rouge155 so every ROUGE invocation is
prepended with the `perl` interpreter.

Usage:
    import rouge_patch  # noqa: F401  (import for side effect)
    rouge_patch.apply(rouge_home="/abs/path/to/ROUGE-1.5.5")
    from pyrouge import Rouge155
    ...
"""
from __future__ import annotations

import os
import shutil
import subprocess

from pyrouge import Rouge155 as _Rouge155

# ROUGE-1.5.5 needs XML::DOM / XML::Parser / DB_File, which Git's bundled Perl
# lacks. Strawberry Perl ships them, but its XML::Parser::Expat DLL depends on
# libexpat in C:\Strawberry\c\bin, so that dir must be on PATH at runtime.
_STRAWBERRY_PERL_CANDIDATES = [
    r"C:\Strawberry\perl\bin\perl.exe",
    r"C:\strawberry\perl\bin\perl.exe",
]
_STRAWBERRY_CBIN = r"C:\Strawberry\c\bin"


def _resolve_perl() -> str:
    for cand in _STRAWBERRY_PERL_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return shutil.which("perl") or "perl"


_PERL = _resolve_perl()
_orig_check_output = subprocess.check_output


def _patched_check_output(command, *args, **kwargs):
    # pyrouge builds command as [bin_path, *options] where bin_path is a .pl
    if isinstance(command, (list, tuple)) and command and str(command[0]).endswith(".pl"):
        command = [_PERL] + list(command)
    return _orig_check_output(command, *args, **kwargs)


def apply(rouge_home: str | None = None) -> None:
    """Patch pyrouge's check_output and optionally set ROUGE_HOME.

    `evaluate.__globals__` is the real `pyrouge.Rouge155` module namespace,
    so patching the symbol there is what actually intercepts the call (the
    `pyrouge.Rouge155` attribute is the re-exported class, not the module).
    """
    _Rouge155.evaluate.__globals__["check_output"] = _patched_check_output
    if os.path.isdir(_STRAWBERRY_CBIN) and _STRAWBERRY_CBIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _STRAWBERRY_CBIN + os.pathsep + os.environ.get("PATH", "")
    if rouge_home:
        os.environ["ROUGE_HOME"] = rouge_home
