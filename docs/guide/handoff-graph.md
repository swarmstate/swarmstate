# Handoff graph

Many "which agent gets control next" decisions are rules over the state, not judgment
calls. Paying an LLM for them is slow, non-deterministic and billable. `HandoffGraph`
resolves them in Rust, in microseconds.

```python
import swarmstate as ss

g = ss.HandoffGraph()
g.add_edge("triage", "billing", when="category == 'billing'")
g.add_edge("triage", "technical", when="category == 'technical' and priority >= 2")
g.add_edge("triage", "human")                       # no condition: the default

g.route("triage", {"category": "billing"})          # -> "billing"
g.route("triage", {"category": "technical", "priority": 3})   # -> "technical"
g.route("triage", {"category": "other"})            # -> "human"
```

Edges are evaluated **in insertion order** and the first match wins, so put the fallback
last. `route` returns `None` when nothing matches (and when the node has no edges).

## The condition mini-language

Conditions are parsed once, at `add_edge`, into an expression tree and evaluated against
the state. This is never Python `eval`: an identifier is only ever a state lookup, so a
condition cannot import, call or mutate anything.

| | |
| --- | --- |
| Literals | `'text'`, `"text"`, `42`, `-3.5`, `true`/`false`, `null`/`none` |
| State access | `category`, `user.tier`, `data.user.role` (dotted paths) |
| Comparisons | `==` `!=` `<` `<=` `>` `>=` |
| Membership | `in` — over a list, a string (substring) or a dict's keys |
| Logic | `and`, `or`, `not`, parentheses |

```python
g.add_edge("n", "escalate", when="user.tier == 'gold' or 'urgent' in tags")
g.add_edge("n", "review", when="not approved and score > 0.8")
```

Evaluation is total: a missing key or a type mismatch yields `false` rather than raising,
so routing cannot crash on unexpected state. An invalid condition raises `ValueError` at
`add_edge` — mistakes surface when you build the graph, not when it runs.

## Only what the conditions read

`route` materializes just the paths its conditions name. A realistic agent state is mostly
message history that no condition looks at, and converting all of it used to dominate the
call — a 500-message state took ~165 µs, against 0.6 µs now, the same as a tiny one.

A practical consequence: a value the codec cannot serialize sitting in an unrelated field
does not block routing. If a *referenced* path holds one, it still raises.

## Cycles

By default the graph refuses an edge that would close a cycle, so a routing loop is a
build-time error:

```python
g = ss.HandoffGraph()                # on_cycle="error" (default)
g.add_edge("a", "b")
g.add_edge("b", "a")                 # ValueError: would create a cycle

g = ss.HandoffGraph(on_cycle="allow")   # cycles permitted (retry loops)
g.add_edge("a", "b")
g.add_edge("b", "a")
g.is_dag()                              # -> False
```

`is_dag()` is a topological sort: O(V+E) and iterative, so it stays fast and stack-safe on
large graphs.

## Introspection

```python
g.nodes()              # every node, sorted
g.edges("triage")      # [(to, when), ...] in insertion order
g.has_node("triage")   # or: "triage" in g
len(g)                 # node count
```
