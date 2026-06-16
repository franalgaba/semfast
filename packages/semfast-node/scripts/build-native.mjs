const packageRoot = new URL("..", import.meta.url).pathname;
const repositoryRoot = new URL("../../..", import.meta.url).pathname;
const nativeOutputPath = `${trimTrailingSlash(packageRoot)}/native/semfast_node_native.node`;
const releaseLibraryPath = `${trimTrailingSlash(repositoryRoot)}/${platformLibraryPath(process.platform)}`;

const cargoResult = Bun.spawnSync({
  cmd: ["cargo", "build", "--release", "-p", "semfast-node-native"],
  cwd: repositoryRoot,
  stdout: "inherit",
  stderr: "inherit",
});

if (!cargoResult.success) {
  process.exit(cargoResult.exitCode);
}

if (!(await Bun.file(releaseLibraryPath).exists())) {
  throw new Error(`Native library was not built at ${releaseLibraryPath}`);
}

await Bun.write(nativeOutputPath, Bun.file(releaseLibraryPath));
console.log(`Wrote ${nativeOutputPath}`);

function platformLibraryPath(platform) {
  switch (platform) {
    case "darwin":
      return "target/release/libsemfast_node_native.dylib";
    case "linux":
      return "target/release/libsemfast_node_native.so";
    case "win32":
      return "target/release/semfast_node_native.dll";
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}

function trimTrailingSlash(path) {
  return path.endsWith("/") ? path.slice(0, -1) : path;
}
