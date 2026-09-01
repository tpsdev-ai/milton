/**
 * F32-discriminator pass condition. Both gate implementations
 * (discriminate-f32.mjs and crate embed-gate) must use this predicate
 * and read `ratio_max` from quant-budget.json — one source, no drift.
 *
 *   pass = (cos_dist <= gate_cos_dist) AND (ratio <= ratio_max)
 *   ratio = cos_dist / quant_budget_cos_dist   (∞ if budget is 0 and cos_dist > 0)
 */

export function f32CaseRatio(cosDist, quantBudgetCosDist) {
  if (quantBudgetCosDist > 0) return cosDist / quantBudgetCosDist;
  return cosDist === 0 ? 0 : Infinity;
}

/**
 * @param {{
 *   cos_dist: number,
 *   quant_budget_cos_dist: number,
 *   gate_cos_dist: number,
 *   ratio_max: number,
 * }} args
 */
export function f32GatePass({
  cos_dist,
  quant_budget_cos_dist,
  gate_cos_dist,
  ratio_max,
}) {
  if (!(ratio_max > 0)) {
    throw new Error("fail-closed: ratio_max must be positive");
  }
  if (!(gate_cos_dist > 0)) {
    throw new Error("fail-closed: gate_cos_dist must be positive");
  }
  const ratio = f32CaseRatio(cos_dist, quant_budget_cos_dist);
  const within_absolute = cos_dist <= gate_cos_dist;
  const within_ratio = ratio <= ratio_max;
  return {
    ratio,
    within_absolute,
    within_ratio,
    pass: within_absolute && within_ratio,
  };
}
