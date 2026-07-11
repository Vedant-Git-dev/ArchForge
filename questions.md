# Stress-test question bank — sample2.txt (Haber-Bosch)

Eight questions designed to break different parts of the pipeline. Each run
goes through the full 6-node default pipeline (reader → chunker → classifier
→ summarizer → fact_checker → writer) + one judge call.

## How correctness is judged
Each "Expected" line states the criterion an honest, faithful answer must
meet — what it MUST contain and, where relevant, what it MUST NOT do (invent
facts not in the input). The pipeline passes if its output meets the criterion;
it fails on hallucination if it asserts something the input does not support.

---

### Q1 — trivial single-fact extraction  (`--type extraction`)
**Q:** "In what year did the first commercial ammonia plant open, and which country imported the nitrate that the process replaced?"
**Expected:** 1913; Germany (imported Chilean saltpeter across sea lanes controlled by the Royal Navy). Must NOT cite a different year.
**Stresses:** over_composed — a 6-agent pipeline for one lookable fact is wasteful; classifier + fact_checker + summarizer arguably don't earn their place.

### Q2 — specific number / hallucination risk  (`--type extraction`)
**Q:** "About how much of the nitrogen in the average human body comes from the Haber-Bosch process, and what share of global energy does the process consume?"
**Expected:** ~50% ("about half") of human-body nitrogen; ~1–2% of global energy (and ~1.4% of CO2). Must NOT invent unrelated numbers.
**Stresses:** fact_checker — concrete numbers whose witness is the text the pipeline already saw; the validator must not flag its own source.

### Q3 — comparative/analytical  (`--type analysis`)
**Q:** "Compare the role of nitrogen fertilizer and the role of explosives in explaining why Haber-Bosch was strategically important."
**Expected:** Identify the dual-use: the SAME synthetic ammonia fixed BOTH the fertilizer shortage and the munitions-nitrate shortage; without it Germany lacked both food and explosives; one chemistry serves both. Must tie fertilizer and explosives together, not treat them separately.
**Stresses:** classifier + summarizer + writer coherence on a reasoned-comparison task; the multi-node chain is genuinely loaded here.

### Q4 — multi-hop causal trace  (`--type analysis`)
**Q:** "Trace the chain from Germany's dependence on Chilean saltpeter to its ability to sustain World War I for four years."
**Expected:** naval-import vulnerability (Royal Navy) → no synthetic ammonia means no food + no explosives within months → Haber/Bosch synthesis → Germany sustained ~4 years. Must connect ≥3 links.
**Stresses:** the linear chain — multi-step reasoning is what the serial path is FOR, so serial_bottleneck would be a spurious diagnosis here; tests whether the judge confuses genuine multi-hop work with a bottleneck defect.

### Q5 — counterfactual (beyond text)  (`--type analysis`)
**Q:** "If BASF had failed to scale Haber's process before 1914, what would have happened to Germany's war capacity?"
**Expected:** One defensible hop: Germany would have exhausted food + munitions nitrogen within months, crippling its ability to wage a sustained war. Must NOT overclaim specifics beyond this — the input only says "within months."
**Stresses:** fact_checker conservativeness — counterfactuals have no textual witness; a strict validator may emit "unverified" verdicts. accuracy if the writer extrapolates wildly.

### Q6 — open synthesis / judgment  (`--type analysis`)
**Q:** "What is the single most important downstream consequence of the Haber-Bosch process, and why?"
**Expected:** A defensible pick, justified from the text: (a) feeding ~half the world / enabling population growth to ~8B, or (b) the dual-use strategic lock (food + munitions), or (c) the ~1–2% energy / carbon burden. Must give a concrete reason grounded in the text.
**Stresses:** writer judgment on an open question with no single right answer; an over-strict judge could unfairly mark a valid answer low (accuracy-floor calibration).

### Q7 — unanswerable / not-in-text (faithfulness stress)  (`--type general`)
**Q:** "What was Fritz Haber's wife's name, and what did she do on the night chlorine gas was first deployed?"
**Expected:** The text mentions Haber and chlorine gas at Ypres 1915 but gives NO information about any wife. A faithful answer states the text provides no such information. Must NOT invent a name. (Historically she was Clara Immerwahr; a faithful pipeline does not know this from the input.)
**Stresses:** hallucination — the harshest test. Does the writer fabricate famous history the input doesn't contain?

### Q8 — "give why X is important" whole-doc synthesis  (`--type summary`)
**Q:** "Give why synthetic nitrogen is important according to the text, and what the main lever for reducing its environmental cost is."
**Expected:** importance — feeds ~half the world, enabled population growth to ~8B, dual-use strategic asset; env lever — switch hydrogen source to green hydrogen (electrolytic, renewable), plus address runoff/eutrophication (Gulf dead zone).
**Stresses:** mirrors the user's original EV question exactly ("give why ev is useful according to text"), so a direct before/after comparison against the diagnoses-empty bug.
