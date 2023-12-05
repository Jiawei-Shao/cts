/**
* AUTO-GENERATED - DO NOT EDIT. Source: https://github.com/gpuweb/cts
**/import { FP } from '../../../../util/floating_point.js';import { sparseScalarF32Range, sparseVectorF32Range } from '../../../../util/math.js';import { makeCaseCache } from '../case_cache.js';

const additionVectorScalarInterval = (v, s) => {
  return FP.f32.toVector(v.map((e) => FP.f32.additionInterval(e, s)));
};

const additionScalarVectorInterval = (s, v) => {
  return FP.f32.toVector(v.map((e) => FP.f32.additionInterval(s, e)));
};

const scalar_cases = [true, false].
map((nonConst) => ({
  [`scalar_${nonConst ? 'non_const' : 'const'}`]: () => {
    return FP.f32.generateScalarPairToIntervalCases(
      sparseScalarF32Range(),
      sparseScalarF32Range(),
      nonConst ? 'unfiltered' : 'finite',
      FP.f32.additionInterval
    );
  }
})).
reduce((a, b) => ({ ...a, ...b }), {});

const vector_scalar_cases = [2, 3, 4].
flatMap((dim) =>
[true, false].map((nonConst) => ({
  [`vec${dim}_scalar_${nonConst ? 'non_const' : 'const'}`]: () => {
    return FP.f32.generateVectorScalarToVectorCases(
      sparseVectorF32Range(dim),
      sparseScalarF32Range(),
      nonConst ? 'unfiltered' : 'finite',
      additionVectorScalarInterval
    );
  }
}))
).
reduce((a, b) => ({ ...a, ...b }), {});

const scalar_vector_cases = [2, 3, 4].
flatMap((dim) =>
[true, false].map((nonConst) => ({
  [`scalar_vec${dim}_${nonConst ? 'non_const' : 'const'}`]: () => {
    return FP.f32.generateScalarVectorToVectorCases(
      sparseScalarF32Range(),
      sparseVectorF32Range(dim),
      nonConst ? 'unfiltered' : 'finite',
      additionScalarVectorInterval
    );
  }
}))
).
reduce((a, b) => ({ ...a, ...b }), {});

export const d = makeCaseCache('binary/f32_addition', {
  ...scalar_cases,
  ...vector_scalar_cases,
  ...scalar_vector_cases
});
//# sourceMappingURL=f32_addition.cache.js.map