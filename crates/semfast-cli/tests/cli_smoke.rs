use std::process::Command;

#[test]
fn cli_builds_queries_inspects_and_benches() {
    let temp_dir = tempfile::tempdir().unwrap();
    let docs_dir = temp_dir.path().join("docs");
    let index_dir = temp_dir.path().join("index");
    let queries_path = temp_dir.path().join("queries.jsonl");
    std::fs::create_dir(&docs_dir).unwrap();
    std::fs::write(
        docs_dir.join("support.md"),
        "Refunds are available for damaged shipments.",
    )
    .unwrap();
    std::fs::write(
        &queries_path,
        "{\"text\":\"damaged shipment refund\",\"expected_chunk_id\":1}\n",
    )
    .unwrap();

    run(["index", "build"], [&docs_dir, &index_dir], Some("--out"));

    let query_output = command()
        .arg("query")
        .arg(&index_dir)
        .arg("damaged shipment refund")
        .output()
        .unwrap();
    assert!(query_output.status.success());
    assert!(String::from_utf8_lossy(&query_output.stdout).contains("Refunds"));

    let inspect_output = command().arg("inspect").arg(&index_dir).output().unwrap();
    assert!(inspect_output.status.success());
    assert!(
        String::from_utf8_lossy(&inspect_output.stdout)
            .contains("\"vector_backend\": \"turbovec\"")
    );

    let bench_output = command()
        .arg("bench")
        .arg(&index_dir)
        .arg("--queries")
        .arg(&queries_path)
        .output()
        .unwrap();
    assert!(bench_output.status.success());
    assert!(String::from_utf8_lossy(&bench_output.stdout).contains("\"search_only\""));
}

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_semfast-cli"))
}

fn run<const N: usize, const M: usize>(
    static_args: [&str; N],
    path_args: [&std::path::Path; M],
    named_path_arg: Option<&str>,
) {
    let mut command = command();
    for arg in static_args {
        command.arg(arg);
    }
    command.arg(path_args[0]);
    if let Some(name) = named_path_arg {
        command.arg(name);
    }
    command.arg(path_args[1]);
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
