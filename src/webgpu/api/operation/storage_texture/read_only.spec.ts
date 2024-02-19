export const description = `
Tests for the behavior of read-only storage textures.

TODO:
- Test mipmap level > 0
- Test bgra8unorm with 'bgra8unorm-storage'
- Test resource usage transitions with read-only storage textures
`;

import { makeTestGroup } from '../../../../common/framework/test_group.js';
import { unreachable, assert } from '../../../../common/util/util.js';
import { Float16Array } from '../../../../external/petamoriken/float16/float16.js';
import { kTextureDimensions } from '../../../capability_info.js';
import {
  ColorTextureFormat,
  kColorTextureFormats,
  kTextureFormatInfo,
} from '../../../format_info.js';
import { GPUTest } from '../../../gpu_test.js';
import { kShaderStages, ShaderStage } from '../../../shader/validation/decl/util.js';

function ComponentCount(format: ColorTextureFormat): number {
  switch (format) {
    case 'r32float':
    case 'r32sint':
    case 'r32uint':
      return 1;
    case 'rg32float':
    case 'rg32sint':
    case 'rg32uint':
      return 2;
    case 'rgba32float':
    case 'rgba32sint':
    case 'rgba32uint':
    case 'rgba8sint':
    case 'rgba8uint':
    case 'rgba8snorm':
    case 'rgba8unorm':
    case 'rgba16float':
    case 'rgba16sint':
    case 'rgba16uint':
      return 4;
    default:
      unreachable();
      return 0;
  }
}

class F extends GPUTest {
  InitTextureAndGetExpectedOutputBufferData(
    storageTexture: GPUTexture,
    format: ColorTextureFormat
  ): ArrayBuffer {
    const bytesPerBlock = kTextureFormatInfo[format].bytesPerBlock;
    assert(bytesPerBlock !== undefined);

    const width = storageTexture.width;
    const height = storageTexture.height;
    const depthOrArrayLayers = storageTexture.depthOrArrayLayers;

    const texelData = new ArrayBuffer(bytesPerBlock * width * height * depthOrArrayLayers);
    const texelTypedDataView = this.GetTypedArrayBufferViewForTexelData(texelData, format);
    const componentCount = ComponentCount(format);
    const outputBufferData = new ArrayBuffer(4 * 4 * width * height * depthOrArrayLayers);
    const outputBufferTypedData = this.GetTypedArrayBufferForOutputBufferData(
      outputBufferData,
      format
    );

    const SetData = (
      texelValue: number,
      outputValue: number,
      texelDataIndex: number,
      component: number
    ) => {
      const texelComponentIndex = texelDataIndex * componentCount + component;
      texelTypedDataView[texelComponentIndex] = texelValue;
      const outputTexelComponentIndex = texelDataIndex * 4 + component;
      outputBufferTypedData[outputTexelComponentIndex] = outputValue;
    };
    for (let z = 0; z < depthOrArrayLayers; ++z) {
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          const texelDataIndex = z * width * height + y * width + x;
          outputBufferTypedData[4 * texelDataIndex] = 0;
          outputBufferTypedData[4 * texelDataIndex + 1] = 0;
          outputBufferTypedData[4 * texelDataIndex + 2] = 0;
          outputBufferTypedData[4 * texelDataIndex + 3] = 1;
          for (let component = 0; component < componentCount; ++component) {
            switch (format) {
              case 'r32uint':
              case 'rg32uint':
              case 'rgba16uint':
              case 'rgba32uint': {
                const texelValue = 4 * texelDataIndex + component + 1;
                SetData(texelValue, texelValue, texelDataIndex, component);
                break;
              }
              case 'rgba8uint': {
                const texelValue = (4 * texelDataIndex + component + 1) % 256;
                SetData(texelValue, texelValue, texelDataIndex, component);
                break;
              }
              case 'rgba8unorm': {
                const texelValue = (4 * texelDataIndex + component + 1) % 256;
                const outputValue = texelValue / 255.0;
                SetData(texelValue, outputValue, texelDataIndex, component);
                break;
              }
              case 'r32sint':
              case 'rg32sint':
              case 'rgba16sint':
              case 'rgba32sint': {
                const texelValue =
                  (texelDataIndex & 1 ? 1 : -1) * (4 * texelDataIndex + component + 1);
                SetData(texelValue, texelValue, texelDataIndex, component);
                break;
              }
              case 'rgba8sint': {
                const texelValue = ((4 * texelDataIndex + component + 1) % 256) - 128;
                SetData(texelValue, texelValue, texelDataIndex, component);
                break;
              }
              case 'rgba8snorm': {
                const texelValue = ((4 * texelDataIndex + component + 1) % 256) - 128;
                const outputValue = Math.max(texelValue / 127.0, -1.0);
                SetData(texelValue, outputValue, texelDataIndex, component);
                break;
              }
              case 'r32float':
              case 'rg32float':
              case 'rgba32float': {
                const texelValue = (4 * texelDataIndex + component + 1) / 10.0;
                SetData(texelValue, texelValue, texelDataIndex, component);
                break;
              }
              case 'rgba16float': {
                const texelValue = (4 * texelDataIndex + component + 1) / 10.0;
                const f16Array = new Float16Array(1);
                f16Array[0] = texelValue;
                SetData(texelValue, f16Array[0], texelDataIndex, component);
                break;
              }
              default:
                unreachable();
                break;
            }
          }
        }
      }
    }
    this.queue.writeTexture(
      {
        texture: storageTexture,
      },
      texelData,
      {
        bytesPerRow: bytesPerBlock * width,
        rowsPerImage: height,
      },
      [width, height, depthOrArrayLayers]
    );

    return outputBufferData;
  }

  GetTypedArrayBufferForOutputBufferData(arrayBuffer: ArrayBuffer, format: ColorTextureFormat) {
    switch (kTextureFormatInfo[format].color.type) {
      case 'uint':
        return new Uint32Array(arrayBuffer);
      case 'sint':
        return new Int32Array(arrayBuffer);
      case 'float':
      case 'unfilterable-float':
        return new Float32Array(arrayBuffer);
    }
  }

  GetTypedArrayBufferViewForTexelData(arrayBuffer: ArrayBuffer, format: ColorTextureFormat) {
    switch (format) {
      case 'r32uint':
      case 'rg32uint':
      case 'rgba32uint':
        return new Uint32Array(arrayBuffer);
      case 'rgba8uint':
      case 'rgba8unorm':
        return new Uint8Array(arrayBuffer);
      case 'rgba16uint':
        return new Uint16Array(arrayBuffer);
      case 'r32sint':
      case 'rg32sint':
      case 'rgba32sint':
        return new Int32Array(arrayBuffer);
      case 'rgba8sint':
      case 'rgba8snorm':
        return new Int8Array(arrayBuffer);
      case 'rgba16sint':
        return new Int16Array(arrayBuffer);
      case 'r32float':
      case 'rg32float':
      case 'rgba32float':
        return new Float32Array(arrayBuffer);
      case 'rgba16float':
        return new Float16Array(arrayBuffer);
      default:
        unreachable();
        return new Uint8Array(arrayBuffer);
    }
  }

  GetOutputBufferWGSLType(format: ColorTextureFormat) {
    switch (kTextureFormatInfo[format].color.type) {
      case 'uint':
        return 'vec4u';
      case 'sint':
        return 'vec4i';
      case 'float':
      case 'unfilterable-float':
        return 'vec4f';
      default:
        unreachable();
        return '';
    }
  }

  DoTransform(
    storageTexture: GPUTexture,
    shaderStage: ShaderStage,
    format: ColorTextureFormat,
    outputBuffer: GPUBuffer
  ) {
    let declaration = '';
    switch (storageTexture.dimension) {
      case '1d':
        declaration = 'texture_storage_1d';
        break;
      case '2d':
        declaration =
          storageTexture.depthOrArrayLayers > 1 ? 'texture_storage_2d_array' : 'texture_storage_2d';
        break;
      case '3d':
        declaration = 'texture_storage_3d';
        break;
    }
    const textureDeclaration = `
    @group(0) @binding(0) var readOnlyTexture: ${declaration}<${format}, read>;
    `;
    const bindingResourceDeclaration = `
    ${textureDeclaration}
    @group(0) @binding(1)
    var<storage,read_write> outputBuffer : array<${this.GetOutputBufferWGSLType(format)}>;
    `;

    const bindGroupEntries = [
      {
        binding: 0,
        resource: storageTexture.createView(),
      },
      {
        binding: 1,
        resource: {
          buffer: outputBuffer,
        },
      },
    ];

    const commandEncoder = this.device.createCommandEncoder();

    switch (shaderStage) {
      case 'compute': {
        let textureLoadCoord = '';
        switch (storageTexture.dimension) {
          case '1d':
            textureLoadCoord = 'invocationID.x';
            break;
          case '2d':
            textureLoadCoord =
              storageTexture.depthOrArrayLayers > 1
                ? `vec2u(invocationID.x, invocationID.y), invocationID.z`
                : `vec2u(invocationID.x, invocationID.y)`;
            break;
          case '3d':
            textureLoadCoord = 'invocationID';
            break;
        }

        const computeShader = `
      ${bindingResourceDeclaration}
      @compute
      @workgroup_size(
        ${storageTexture.width}, ${storageTexture.height}, ${storageTexture.depthOrArrayLayers})
      fn main(
        @builtin(local_invocation_id) invocationID: vec3u,
        @builtin(local_invocation_index) invocationIndex: u32) {
        let initialValue = textureLoad(readOnlyTexture, ${textureLoadCoord});
        outputBuffer[invocationIndex] = initialValue;
      }`;
        const computePipeline = this.device.createComputePipeline({
          compute: {
            module: this.device.createShaderModule({
              code: computeShader,
            }),
          },
          layout: 'auto',
        });
        const bindGroup = this.device.createBindGroup({
          layout: computePipeline.getBindGroupLayout(0),
          entries: bindGroupEntries,
        });

        const computePassEncoder = commandEncoder.beginComputePass();
        computePassEncoder.setPipeline(computePipeline);
        computePassEncoder.setBindGroup(0, bindGroup);
        computePassEncoder.dispatchWorkgroups(1);
        computePassEncoder.end();
        break;
      }
      case 'fragment': {
        let textureLoadCoord = '';
        switch (storageTexture.dimension) {
          case '1d':
            textureLoadCoord = 'textureCoord.x';
            break;
          case '2d':
            textureLoadCoord =
              storageTexture.depthOrArrayLayers > 1 ? 'textureCoord, z' : 'textureCoord';
            break;
          case '3d':
            textureLoadCoord = 'vec3u(textureCoord, z)';
            break;
        }

        const fragmentShader = `
        ${bindingResourceDeclaration}
        @fragment
        fn main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
          let textureCoord = vec2u(fragCoord.xy);
          let storageTextureTexelCountPerImage = ${storageTexture.width * storageTexture.height}u;
          for (var z = 0u; z < ${storageTexture.depthOrArrayLayers}; z++) {
            let initialValue = textureLoad(readOnlyTexture, ${textureLoadCoord});
            let outputIndex =
              storageTextureTexelCountPerImage * z + textureCoord.y * ${storageTexture.width} +
              textureCoord.x;
            outputBuffer[outputIndex] = initialValue;
          }
          return vec4f(0.0, 1.0, 0.0, 1.0);
        }`;
        const vertexShader = `
            @vertex
            fn main(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
                var pos = array(
                  vec2f(-1.0, -1.0),
                  vec2f(-1.0,  1.0),
                  vec2f( 1.0, -1.0),
                  vec2f(-1.0,  1.0),
                  vec2f( 1.0, -1.0),
                  vec2f( 1.0,  1.0));
                return vec4f(pos[vertexIndex], 0.0, 1.0);
            }
          `;
        const renderPipeline = this.device.createRenderPipeline({
          layout: 'auto',
          vertex: {
            module: this.device.createShaderModule({
              code: vertexShader,
            }),
          },
          fragment: {
            module: this.device.createShaderModule({
              code: fragmentShader,
            }),
            targets: [
              {
                format: 'rgba8unorm',
              },
            ],
          },
          primitive: {
            topology: 'triangle-list',
          },
        });

        const bindGroup = this.device.createBindGroup({
          layout: renderPipeline.getBindGroupLayout(0),
          entries: bindGroupEntries,
        });

        const placeholderColorTexture = this.device.createTexture({
          size: [storageTexture.width, storageTexture.height, 1],
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
          format: 'rgba8unorm',
        });
        this.trackForCleanup(placeholderColorTexture);

        const renderPassEncoder = commandEncoder.beginRenderPass({
          colorAttachments: [
            {
              view: placeholderColorTexture.createView(),
              loadOp: 'clear',
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              storeOp: 'store',
            },
          ],
        });
        renderPassEncoder.setPipeline(renderPipeline);
        renderPassEncoder.setBindGroup(0, bindGroup);
        renderPassEncoder.draw(6);
        renderPassEncoder.end();
        break;
      }
      case 'vertex': {
        // Draw one vertex (as one point in the point list) at (coordX + 0.5, coordY + 0.5) in the
        // storageTexture.width * storageTexture.height grid (in frame buffer coordinates), and each
        // vertex carries all the texels at (coordX, coordY) in every array layer of the read-only
        // storage texture in its vertex attributes.
        let vertexOutputs = '';
        for (let layer = 0; layer < storageTexture.depthOrArrayLayers; ++layer) {
          vertexOutputs = vertexOutputs.concat(
            `
            @location(${layer + 1}) @interpolate(flat)
            vertex_out${layer}: ${this.GetOutputBufferWGSLType(format)},`
          );
        }

        let loadFromTextureWGSL = '';
        switch (storageTexture.dimension) {
          case '1d':
            loadFromTextureWGSL = `
              output.vertex_out0 = textureLoad(readOnlyTexture, coordX);`;
            break;
          case '2d':
            if (storageTexture.depthOrArrayLayers === 1) {
              loadFromTextureWGSL = `
                output.vertex_out0 = textureLoad(readOnlyTexture, vec2u(coordX, coordY));`;
            } else {
              for (let z = 0; z < storageTexture.depthOrArrayLayers; ++z) {
                loadFromTextureWGSL = loadFromTextureWGSL.concat(
                  `output.vertex_out${z} =
                    textureLoad(readOnlyTexture, vec2u(coordX, coordY), ${z});`
                );
              }
            }
            break;
          case '3d':
            for (let z = 0; z < storageTexture.depthOrArrayLayers; ++z) {
              loadFromTextureWGSL = loadFromTextureWGSL.concat(
                `output.vertex_out${z} = textureLoad(readOnlyTexture, vec3u(coordX, coordY, ${z}));`
              );
            }
            break;
        }

        let outputToBufferWGSL = '';
        for (let layer = 0; layer < storageTexture.depthOrArrayLayers; ++layer) {
          outputToBufferWGSL = outputToBufferWGSL.concat(
            `
            let outputIndex${layer} =
              storageTextureTexelCountPerImage * ${layer}u +
              fragmentInput.tex_coord.y * ${storageTexture.width}u + fragmentInput.tex_coord.x;
            outputBuffer[outputIndex${layer}] = fragmentInput.vertex_out${layer};`
          );
        }

        const shader = `
        ${bindingResourceDeclaration}
        struct VertexOutput {
          @builtin(position) my_pos: vec4f,
          @location(0) @interpolate(flat) tex_coord: vec2u,
          ${vertexOutputs}
        }
        @vertex
        fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
            var output : VertexOutput;
            let coordX = vertexIndex % ${storageTexture.width}u;
            let coordY = vertexIndex / ${storageTexture.width}u;

            let outputPosX =
              -1.0 + 1.0 / f32(${storageTexture.width}) +
              2.0 / f32(${storageTexture.width}) * f32(coordX);
            let outputPosY =
              -1.0 + 1.0 / f32(${storageTexture.height}) +
              2.0 / f32(${storageTexture.height}) * f32(coordY);
            output.my_pos = vec4f(outputPosX, outputPosY, 0.0, 1.0);

            output.tex_coord = vec2u(coordX, coordY);

            ${loadFromTextureWGSL}

            return output;
        }
        @fragment
        fn fs_main(fragmentInput : VertexOutput) -> @location(0) vec4f {
          let storageTextureTexelCountPerImage = ${storageTexture.width * storageTexture.height}u;
          ${outputToBufferWGSL}
          return vec4f(0.0, 1.0, 0.0, 1.0);
        }
        `;

        const renderPipeline = this.device.createRenderPipeline({
          layout: 'auto',
          vertex: {
            module: this.device.createShaderModule({
              code: shader,
            }),
          },
          fragment: {
            module: this.device.createShaderModule({
              code: shader,
            }),
            targets: [
              {
                format: 'rgba8unorm',
              },
            ],
          },
          primitive: {
            topology: 'point-list',
          },
        });

        const bindGroup = this.device.createBindGroup({
          layout: renderPipeline.getBindGroupLayout(0),
          entries: bindGroupEntries,
        });

        const placeholderColorTexture = this.device.createTexture({
          size: [storageTexture.width, storageTexture.height, 1],
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
          format: 'rgba8unorm',
        });
        this.trackForCleanup(placeholderColorTexture);

        const renderPassEncoder = commandEncoder.beginRenderPass({
          colorAttachments: [
            {
              view: placeholderColorTexture.createView(),
              loadOp: 'clear',
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              storeOp: 'store',
            },
          ],
        });
        renderPassEncoder.setPipeline(renderPipeline);
        renderPassEncoder.setBindGroup(0, bindGroup);
        renderPassEncoder.draw(storageTexture.width * storageTexture.height);
        renderPassEncoder.end();

        break;
      }
    }

    this.queue.submit([commandEncoder.finish()]);
  }
}

export const g = makeTestGroup(F);

g.test('basic')
  .desc(
    `The basic functionality tests for read-only storage textures. In the test we read data from
    the read-only storage texture, write the data into an output storage buffer, and check if the
    data in the output storage buffer is exactly what we expect.`
  )
  .params(u =>
    u
      .combine('format', kColorTextureFormats)
      .filter(p => kTextureFormatInfo[p.format].color?.storage === true)
      .combine('shaderStage', kShaderStages)
      .combine('textureDimension', kTextureDimensions)
      .combine('depthOrArrayLayers', [1, 2] as const)
      .unless(p => p.textureDimension === '1d' && p.depthOrArrayLayers > 1)
  )
  .fn(t => {
    const { format, shaderStage, textureDimension, depthOrArrayLayers } = t.params;

    const kWidth = 8;
    const height = textureDimension === '1d' ? 1 : 8;
    const textureSize = [kWidth, height, depthOrArrayLayers] as const;
    const storageTexture = t.device.createTexture({
      format,
      size: textureSize,
      dimension: textureDimension,
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
    });
    t.trackForCleanup(storageTexture);

    const expectedData = t.InitTextureAndGetExpectedOutputBufferData(storageTexture, format);

    const outputBuffer = t.device.createBuffer({
      size: 4 * 4 * kWidth * height * depthOrArrayLayers,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    });
    t.trackForCleanup(outputBuffer);

    t.DoTransform(storageTexture, shaderStage, format, outputBuffer);

    switch (kTextureFormatInfo[format].color.type) {
      case 'uint':
        t.expectGPUBufferValuesEqual(outputBuffer, new Uint32Array(expectedData));
        break;
      case 'sint':
        t.expectGPUBufferValuesEqual(outputBuffer, new Int32Array(expectedData));
        break;
      case 'float':
      case 'unfilterable-float':
        t.expectGPUBufferValuesEqual(outputBuffer, new Float32Array(expectedData));
        break;
      default:
        unreachable();
        break;
    }
  });
