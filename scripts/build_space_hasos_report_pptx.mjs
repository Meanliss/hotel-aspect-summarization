#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const scriptDir = path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1");
const bundledPython = path.join(
  os.homedir(),
  ".cache",
  "codex-runtimes",
  "codex-primary-runtime",
  "dependencies",
  "python",
  "python.exe",
);
const python = process.env.PYTHON_EXE || (fs.existsSync(bundledPython) ? bundledPython : "python");
const target = path.join(scriptDir, "build_space_hasos_report_pptx.py");
const result = spawnSync(python, [target, ...process.argv.slice(2)], {
  stdio: "inherit",
  shell: process.platform === "win32",
});
process.exit(result.status ?? 1);
