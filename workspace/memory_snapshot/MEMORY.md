# Memory index — ARX real robot operating wisdom

Each entry is a short markdown file in this directory.

## Calibration

- [Workspace calibration](../env_calibration.md) — workspace bounds, gripper values, camera mapping

## Core operating rules

### Pi0.5 prompt ladder
When pi0_pick fails, escalate through these rungs in order:
1. Sub-instruction: "pick up the {object}"
2. Full task language (describe object + context + target)
3. Spatial qualifier ("...on the left" / "...next to the box")
4. Re-position pre-pos (adjust approach height/offset by 5cm) and retry

### Error recovery patterns
| Failure mode | Diagnostic signal | Recovery |
|---|---|---|
| VLA grasp miss | EEF Z no change, gripper not closed | Prompt Ladder next rung → re-pre-pos → retry |
| Object slip | Object Z drops back to table | release → re-pre-pos → pi0_pick (full task language) |
| Move stall | EEF not converging in max_steps | Split into smaller waypoints → rotate wrist → retry |

### Calibration must-dos (before first real run)
1. Workspace bounds (teach pendant → each extreme → record xyz)
2. Table surface height (z_min)
3. Gripper open/close values
4. EEF-object offsets per object type (grasp once, measure delta)

## Experiment memories

(Add entries here after each experiment session.)
