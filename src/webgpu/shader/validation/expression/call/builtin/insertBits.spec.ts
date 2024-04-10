const builtin = 'insertBits';
export const description = `
Validation tests for the ${builtin}() builtin.
`;

import { makeTestGroup } from '../../../../../../common/framework/test_group.js';
import { keysOf, objectsToRecord } from '../../../../../../common/util/data_tables.js';
import {
  Type,
  kConcreteIntegerScalarsAndVectors,
  kFloatScalarsAndVectors,
  u32,
} from '../../../../../util/conversion.js';
import { ShaderValidationTest } from '../../../shader_validation_test.js';

import {
  fullRangeForType,
  kConstantAndOverrideStages,
  stageSupportsType,
  validateConstOrOverrideBuiltinEval,
} from './const_override_validation.js';

export const g = makeTestGroup(ShaderValidationTest);

const kValuesTypes = objectsToRecord(kConcreteIntegerScalarsAndVectors);

g.test('values')
  .desc(
    `
Validates that constant evaluation and override evaluation of ${builtin}() never errors on valid inputs
`
  )
  .params(u =>
    u
      .combine('stage', kConstantAndOverrideStages)
      .combine('type', keysOf(kValuesTypes))
      .filter(u => stageSupportsType(u.stage, kValuesTypes[u.type]))
      .beginSubcases()
      .expand('value', u => fullRangeForType(kValuesTypes[u.type], 5))
      .expand('newbits', u => fullRangeForType(kValuesTypes[u.type], 5))
      .combineWithParams([
        { offset: 0, count: 0 },
        { offset: 0, count: 31 },
        { offset: 0, count: 32 },
        { offset: 4, count: 0 },
        { offset: 4, count: 27 },
        { offset: 4, count: 28 },
        { offset: 16, count: 0 },
        { offset: 16, count: 15 },
        { offset: 16, count: 16 },
        { offset: 32, count: 0 },
      ] as const)
  )
  .fn(t => {
    const expectedResult = true; // insertBits() should never error
    validateConstOrOverrideBuiltinEval(
      t,
      builtin,
      expectedResult,
      [
        kValuesTypes[t.params.type].create(t.params.value),
        kValuesTypes[t.params.type].create(t.params.newbits),
        u32(t.params.offset),
        u32(t.params.count),
      ],
      t.params.stage
    );
  });

g.test('mismatched')
  .desc(
    `
Validates that even with valid types, if arg0 and arg1 do not match types ${builtin}() errors
`
  )
  .params(u => u.combine('arg0', keysOf(kValuesTypes)).combine('arg1', keysOf(kValuesTypes)))
  .fn(t => {
    const arg0 = kValuesTypes[t.params.arg0];
    const arg1 = kValuesTypes[t.params.arg1];
    validateConstOrOverrideBuiltinEval(
      t,
      builtin,
      /* expectedResult */ t.params.arg0 === t.params.arg1,
      [arg0.create(0), arg1.create(1), u32(0), u32(32)],
      'constant'
    );
  });

g.test('count_offset')
  .desc(
    `
Validates that count and offset must be smaller than the size of the primitive.
`
  )
  .params(u =>
    u
      .combine('stage', kConstantAndOverrideStages)
      .beginSubcases()
      .combineWithParams([
        // offset + count < 32
        { offset: 0, count: 31 },
        { offset: 1, count: 30 },
        { offset: 31, count: 0 },
        { offset: 30, count: 1 },
        // offset + count == 32
        { offset: 0, count: 32 },
        { offset: 1, count: 31 },
        { offset: 16, count: 16 },
        { offset: 31, count: 1 },
        { offset: 32, count: 0 },
        // offset + count > 32
        { offset: 2, count: 31 },
        { offset: 31, count: 2 },
        // offset > 32
        { offset: 33, count: 0 },
        { offset: 33, count: 1 },
        // count > 32
        { offset: 0, count: 33 },
        { offset: 1, count: 33 },
      ] as const)
  )
  .fn(t => {
    validateConstOrOverrideBuiltinEval(
      t,
      builtin,
      /* expectedResult */ t.params.offset + t.params.count <= 32,
      [u32(0), u32(1), u32(t.params.offset), u32(t.params.count)],
      t.params.stage
    );
  });

interface Argument {
  /** Argument as a string. */
  readonly arg: string;
  /** Is this a valid argument type. Note that all args must be valid for the call to be valid. */
  readonly pass: boolean;
  /** Additional setup code necessary for this arg in the function scope. */
  readonly preamble?: string;
}

function typesToArguments(types: readonly Type[], pass: boolean): Record<string, Argument> {
  return types.reduce(
    (res, type) => ({
      ...res,
      [type.toString()]: { arg: type.create(0).wgsl(), pass },
    }),
    {}
  );
}

// u32 is included here to confirm that validation is failing due to a type issue and not something else.
const kInputArgTypes: { readonly [name: string]: Argument } = {
  ...typesToArguments([Type.u32], true),
  ...typesToArguments([...kFloatScalarsAndVectors, Type.bool, Type.mat2x2f], false),
  alias: { arg: 'u32_alias(1)', pass: true },
  vec_bool: { arg: 'vec2<bool>(false,true)', pass: false },
  atomic: { arg: 'a', pass: false },
  array: {
    preamble: 'var arry: array<u32, 5>;',
    arg: 'arry',
    pass: false,
  },
  array_runtime: { arg: 'k.arry', pass: false },
  struct: {
    preamble: 'var x: A;',
    arg: 'x',
    pass: false,
  },
  enumerant: { arg: 'read_write', pass: false },
  ptr: {
    preamble: `var<function> f = 1u;
               let p: ptr<function, u32> = &f;`,
    arg: 'p',
    pass: false,
  },
  ptr_deref: {
    preamble: `var<function> f = 1u;
               let p: ptr<function, u32> = &f;`,
    arg: '*p',
    pass: true,
  },
  sampler: { arg: 's', pass: false },
  texture: { arg: 't', pass: false },
};

g.test('typed_arguments')
  .desc(
    `
Test compilation validation of ${builtin} with variously typed arguments
  - The input types are matching to reduce testing permutations. Mismatching input types are
    validated in 'mismatched' test above.
  - For failing input types, just use the same type for offset and count to reduce combinations.
`
  )
  .params(u =>
    u
      .combine('input', keysOf(kInputArgTypes))
      .beginSubcases()
      .combine('offset', keysOf(kInputArgTypes))
      .expand('count', u => (kInputArgTypes[u.input].pass ? keysOf(kInputArgTypes) : [u.offset]))
  )
  .fn(t => {
    const input = kInputArgTypes[t.params.input];
    const offset = kInputArgTypes[t.params.offset];
    const count = kInputArgTypes[t.params.count];
    t.expectCompileResult(
      input.pass && offset.pass && count.pass,
      `alias u32_alias = u32;

      @group(0) @binding(0) var s: sampler;
      @group(0) @binding(1) var t: texture_2d<f32>;

      var<workgroup> a: atomic<u32>;

      struct A {
        i: u32,
      }
      struct B {
        arry: array<u32>,
      }
      @group(0) @binding(3) var<storage> k: B;


      @vertex
      fn main() -> @builtin(position) vec4<f32> {
        ${input.preamble ? input.preamble : ''}
        ${offset.preamble && offset !== input ? offset.preamble : ''}
        ${count.preamble && count !== input && count !== offset ? count.preamble : ''}
        _ = ${builtin}(${input.arg},${input.arg},${offset.arg},${count.arg});
        return vec4<f32>(.4, .2, .3, .1);
      }`
    );
  });

g.test('must_use')
  .desc(`Result of ${builtin} must be used`)
  .params(u => u.combine('use', [true, false]))
  .fn(t => {
    const use_it = t.params.use ? '_ = ' : '';
    t.expectCompileResult(t.params.use, `fn f() { ${use_it}${builtin}(0u,1u,0u,1u); }`);
  });
