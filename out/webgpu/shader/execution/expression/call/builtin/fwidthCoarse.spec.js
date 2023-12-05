/**
* AUTO-GENERATED - DO NOT EDIT. Source: https://github.com/gpuweb/cts
**/export const description = `
Execution tests for the 'fwidthCoarse' builtin function

T is f32 or vecN<f32>
fn fwidthCoarse(e:T) ->T
Returns abs(dpdxCoarse(e)) + abs(dpdyCoarse(e)).
`;import { makeTestGroup } from '../../../../../../common/framework/test_group.js';
import { GPUTest } from '../../../../../gpu_test.js';
import { allInputSources } from '../../expression.js';

export const g = makeTestGroup(GPUTest);

g.test('f32').
specURL('https://www.w3.org/TR/WGSL/#derivative-builtin-functions').
desc(`f32 tests`).
params((u) =>
u.combine('inputSource', allInputSources).combine('vectorize', [undefined, 2, 3, 4])
).
unimplemented();
//# sourceMappingURL=fwidthCoarse.spec.js.map