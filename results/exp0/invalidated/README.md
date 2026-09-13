# Invalidated artifacts — retained deliberately, not deleted

`exp0-INVALID.json` printed a confident verdict (SEPARABLE; cross-AUROC A->B 0.546,
B->A 0.558). It is **not a result**, for three independent reasons, any one sufficient:

1. **Duplicate episodes.** It analysed 18 groups, not 16. Two cells (`blocked-s4`,
   `blocked-s5`) were run twice by two concurrent processes and both copies kept. The
   episode is the unit of independence for every interval in this project, so a
   duplicate is a correlated observation inflating n — pseudo-replication.

2. **Contaminated SIM arm.** 35% of simulated tool outputs were rate-limit error
   strings rather than fabrications, so `B_empirical` was partly measuring "real output
   vs error message" — a contrast a probe separates trivially and for the wrong reason.

3. **Missing arm.** No `tempting` episodes: the class where crossing the scope boundary
   is instrumentally rewarded.

Produced by a process launched before the dedupe, locking and `SimulatorUnavailable`
fixes existed, which the supervisor had terminated and which kept running to completion.

Kept because a discarded result that is not recorded looks, later, like a result that
was never run. The guards that now make each failure impossible: `agent.dedupe_episodes`,
`agent.lock_log`, `arms.sim_stub.SimulatorUnavailable`, and the `SIM_CONTAMINATION_LIMIT`
gate in `experiments/exp0_construct_separation.py`.
