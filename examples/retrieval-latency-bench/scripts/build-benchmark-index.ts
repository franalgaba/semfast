const docsPath = "examples/retrieval-latency-bench/data/scifact/documents.jsonl";
const outPath = "examples/retrieval-latency-bench/data/index";

const result = Bun.spawnSync({
  cmd: [
    "cargo",
    "run",
    "-p",
    "semfast-cli",
    "--",
    "index",
    "build",
    docsPath,
    "--out",
    outPath,
  ],
  cwd: "../..",
  stdout: "inherit",
  stderr: "inherit",
});

if (!result.success) {
  process.exit(result.exitCode);
}

console.log(`Built SciFact index at ${outPath}`);
