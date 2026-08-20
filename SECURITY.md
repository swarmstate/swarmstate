# Security policy

## Reporting a vulnerability

Report privately through
[GitHub's private vulnerability reporting](https://github.com/swarmstate/swarmstate/security/advisories/new)
rather than in a public issue. Include what you did, what happened, and the version and
platform you saw it on; a crashing input or a small reproducer is worth more than a
description.

Expect an acknowledgement within a few days. Fixes ship in a patch release with the
advisory published alongside it.

## Supported versions

Only the latest release is supported. There are no long-term-support branches yet.

## What is in scope

swarmstate is a state backend with a native core, so the interesting surface is anything
that turns bytes or user data into native work:

- **The msgpack decoder** (`swarmstate.loads`, and every read from a store). It parses
  bytes that may come from Redis, Postgres or a file written by something else, so treat
  untrusted input to it as in scope: a panic, an abort, an out-of-bounds read, unbounded
  allocation from a declared length, or a stack overflow through nesting. Nesting is capped
  at 128 levels in both directions; a way past that cap is a bug worth reporting.
- **The condition evaluator** (`HandoffGraph(when=...)`). Conditions are parsed into an
  expression tree and evaluated in Rust — never `eval`. An identifier is only ever a state
  lookup. Anything that gets a condition to import, call, read an attribute, or escape that
  model is a vulnerability, not a feature request.
- **The store's concurrency.** Data races, use-after-free or corruption reachable from
  Python, including on free-threaded (no-GIL) builds where the GIL is not serializing
  callers.
- **SQL and command construction in the backends.** Namespaces, keys and table names come
  from callers. Values and identifiers are bound as parameters; table names are validated
  against an identifier pattern. An injection through any of those paths is in scope, as is
  a prefix filter that treats caller data as a wildcard.

## What is not in scope

- **Anything the caller chose to store.** swarmstate persists what it is given; it does not
  encrypt values, and it makes no attempt to keep secrets out of a store, a snapshot, a
  metrics label or a log line.
- **Access control.** There is none. A store is reachable by anything that can reach the
  process, the Redis instance or the database — securing those is the deployment's job. The
  connection string you hand a backend is trusted input.
- **Denial of service through legitimate use.** An unbounded number of checkpoints will
  exhaust memory; that is what `max_checkpoints_per_thread` and `max_history` are for.
- Vulnerabilities in LangGraph, CrewAI, redis-py, psycopg or msgpack themselves — report
  those upstream.
