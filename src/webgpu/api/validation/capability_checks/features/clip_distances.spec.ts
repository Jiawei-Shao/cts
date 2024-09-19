import { range } from '../../../../../common/util/util.js';
import { align } from '../../../../util/math.js';
import { kMaximumLimitBaseParams, makeLimitTestGroup } from '../limits/limit_utils.js';

function getPipelineDescriptorWithClipDistances(
  device: GPUDevice,
  interStageShaderVariables: number,
  pointList: boolean,
  clipDistances: number
): GPURenderPipelineDescriptor {
  const vertexOutputVariables =
    interStageShaderVariables - (pointList ? 1 : 0) - align(clipDistances, 4) / 4;
  const maxVertexOutputVariables =
    device.limits.maxInterStageShaderVariables - (pointList ? 1 : 0) - align(clipDistances, 4) / 4;

  const varyings = `
      ${range(vertexOutputVariables, i => `@location(${i}) v4_${i}: vec4f,`).join('\n')}
  `;

  const code = `
    // test value                        : ${interStageShaderVariables}
    // maxInterStageShaderVariables     : ${device.limits.maxInterStageShaderVariables}
    // num variables in vertex shader : ${vertexOutputVariables}${
      pointList ? ' + point-list' : ''
    }${
      clipDistances > 0
        ? ` + ${align(clipDistances, 4) / 4} (clip_distances[${clipDistances}])`
        : ''
    }
    // maxInterStageVariables:           : ${maxVertexOutputVariables}
    // num used inter stage variables    : ${vertexOutputVariables}

    enable clip_distances;

    struct VSOut {
      @builtin(position) p: vec4f,
      ${varyings}
      ${
        clipDistances > 0
          ? `@builtin(clip_distances) clipDistances: array<f32, ${clipDistances}>,`
          : ''
      }
    }
    struct FSIn {
      ${varyings}
    }
    struct FSOut {
      @location(0) color: vec4f,
    }
    @vertex fn vs() -> VSOut {
      var o: VSOut;
      o.p = vec4f(0);
      return o;
    }
    @fragment fn fs(i: FSIn) -> FSOut {
      var o: FSOut;
      o.color = vec4f(0);
      return o;
    }
  `;
  const module = device.createShaderModule({ code });
  const pipelineDescriptor: GPURenderPipelineDescriptor = {
    layout: 'auto',
    primitive: {
      topology: pointList ? 'point-list' : 'triangle-list',
    },
    vertex: {
      module,
    },
    fragment: {
      module,
      targets: [
        {
          format: 'rgba8unorm',
        },
      ],
    },
  };
  return pipelineDescriptor;
}

const limit = 'maxInterStageShaderVariables';
export const { g, description } = makeLimitTestGroup(limit);

g.test('createRenderPipeline,at_over')
  .desc(`Test using at and over ${limit} limit with clip_distances in createRenderPipeline(Async)`)
  .params(
    kMaximumLimitBaseParams
      .combine('async', [false, true])
      .combine('pointList', [false, true])
      .combine('clipDistances', [1, 2, 3, 4, 5, 6, 7, 8])
  )
  .beforeAllSubcases(t => {
    t.selectDeviceOrSkipTestCase('clip-distances');
  })
  .fn(async t => {
    const { limitTest, testValueName, async, pointList, clipDistances } = t.params;
    await t.testDeviceWithRequestedMaximumLimits(
      limitTest,
      testValueName,
      async ({ device, testValue, shouldError }) => {
        const pipelineDescriptor = getPipelineDescriptorWithClipDistances(
          device,
          testValue,
          pointList,
          clipDistances
        );

        await t.testCreateRenderPipeline(pipelineDescriptor, async, shouldError);
      },
      undefined,
      ['clip-distances']
    );
  });
