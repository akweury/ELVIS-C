## Pattern: `proximity_group_trigger`

**Goal**  
Test whether a model can detect proximity-based group formation and reason about its causal role in triggering an effect.

---

### 🔧 Setup

- `N = 6` objects: 
  - `L = {l₁, l₂, l₃}` on left half `x ∈ [0.1, 0.4]`
  - `R = {r₁, r₂, r₃}` on right half `x ∈ [0.6, 0.9]`
- All objects are randomly placed in their respective regions at `T = 0`
- Scene center is `C = (0.5, 0.5)`

---

### ✅ Positive Case

- `T = 0–10`: `L` moves toward `C` and forms a visible, **non-overlapping group** arranged in a circle
- `T = 12`: green trigger flash appears at `C`
- `T ≥ 13`: only grouped `L` objects change (e.g., color turns red)
- `R` remains static and unaffected

---

### ❌ Negative Case

Generated using one or more of the following violations:

- No convergence: `L` and `R` jitter randomly; no group forms
- Wrong trigger timing: flash appears too early (`T = 5`)
- Wrong effect: either `R` or **all** objects change color after `T ≥ 13`
- Effect occurs without meaningful group formation

---

### 🧠 Principle

- **Gestalt principle**: Proximity (emergent, motion-based grouping)
- **Causal structure**: Group → Trigger → Selective effect
