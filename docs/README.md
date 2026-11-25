# Docs

These pages explain how the tasks were chosen, what can make their scores misleading, what hardware they need, and how they were independently checked. Each page opens with a short answer, and the detail follows.

| Page | Read it if you want to know |
|---|---|
| [selection/](selection/) | Where the 21 tasks came from, and why most candidates were dropped |
| [findings.md](findings.md) | What can make a speedup score misleading: gameable verifiers, wrong human fixes, environment effects, non-repeatable engines, harness bugs |
| [hardware.md](hardware.md) | Our GPUs, which tasks run on the RTX 4090 and RTX 5090 (with numbers), CPU pinning, `TZ=UTC`, and what the 5090 needs |
| [acceptance/](acceptance/) | Which tasks an independent tester rebuilt from scratch, what they got, and what they found broken |

**Where to start:**
- **Before running a task:** read hardware.md.
- **Before using a score to reward or rank agents:** read findings.md. Its first four sections matter most.
- **Deciding how far to trust one task:** read its row in acceptance/README.md.
