# Security Policy

## Reporting

Please report security issues privately to the repository owner before public disclosure.

Include:

- affected component,
- reproduction steps,
- expected impact,
- whether the issue requires a malicious index artifact, malicious query input, or compromised dependency.

## Scope

Security-sensitive areas include:

- loading local Semfast artifacts,
- native N-API bindings,
- generated benchmark data,
- downloaded embedding/runtime dependencies,
- Hono server endpoints in the TypeScript package and example app.

Do not include secrets, API keys, private model tokens, or proprietary datasets in issues or pull requests.
