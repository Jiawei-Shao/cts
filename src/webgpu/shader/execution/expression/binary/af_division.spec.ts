export const description = `
Execution Tests for non-matrix AbstractFloat division expression
`;

import { makeTestGroup } from '../../../../../common/framework/test_group.js';
import { GPUTest } from '../../../../gpu_test.js';
import { TypeAbstractFloat, TypeVec } from '../../../../util/conversion.js';
import { onlyConstInputSource, run } from '../expression.js';

import { d } from './af_division.cache.js';
import { abstractBinary } from './binary.js';

export const g = makeTestGroup(GPUTest);

g.test('scalar')
  .specURL('https://www.w3.org/TR/WGSL/#floating-point-evaluation')
  .desc(
    `
Expression: x / y, where x and y are scalars
Accuracy: 2.5 ULP for |y| in the range [2^-126, 2^126]
`
  )
  .params(u => u.combine('inputSource', onlyConstInputSource))
  .fn(async t => {
    const cases = await d.get('scalar');
    await run(
      t,
      abstractBinary('/'),
      [TypeAbstractFloat, TypeAbstractFloat],
      TypeAbstractFloat,
      t.params,
      cases
    );
  });

g.test('vector')
  .specURL('https://www.w3.org/TR/WGSL/#floating-point-evaluation')
  .desc(
    `
Expression: x / y, where x and y are vectors
Accuracy: 2.5 ULP for |y| in the range [2^-126, 2^126]
`
  )
  .params(u =>
    u.combine('inputSource', onlyConstInputSource).combine('vectorize', [2, 3, 4] as const)
  )
  .fn(async t => {
    const cases = await d.get('scalar'); // Using vectorize to generate vector cases based on scalar cases
    await run(
      t,
      abstractBinary('/'),
      [TypeAbstractFloat, TypeAbstractFloat],
      TypeAbstractFloat,
      t.params,
      cases
    );
  });

g.test('vector_scalar')
  .specURL('https://www.w3.org/TR/WGSL/#floating-point-evaluation')
  .desc(
    `
Expression: x / y, where x is a vector and y is a scalar
Accuracy: Correctly rounded
`
  )
  .params(u => u.combine('inputSource', onlyConstInputSource).combine('dim', [2, 3, 4] as const))
  .fn(async t => {
    const dim = t.params.dim;
    const cases = await d.get(`vec${dim}_scalar`);
    await run(
      t,
      abstractBinary('/'),
      [TypeVec(dim, TypeAbstractFloat), TypeAbstractFloat],
      TypeVec(dim, TypeAbstractFloat),
      t.params,
      cases
    );
  });

g.test('scalar_vector')
  .specURL('https://www.w3.org/TR/WGSL/#floating-point-evaluation')
  .desc(
    `
Expression: x / y, where x is a scalar and y is a vector
Accuracy: Correctly rounded
`
  )
  .params(u => u.combine('inputSource', onlyConstInputSource).combine('dim', [2, 3, 4] as const))
  .fn(async t => {
    const dim = t.params.dim;
    const cases = await d.get(`scalar_vec${dim}`);
    await run(
      t,
      abstractBinary('/'),
      [TypeAbstractFloat, TypeVec(dim, TypeAbstractFloat)],
      TypeVec(dim, TypeAbstractFloat),
      t.params,
      cases
    );
  });
