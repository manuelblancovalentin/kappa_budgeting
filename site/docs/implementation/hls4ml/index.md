---
title: "hls4ml Implementation Roadmap"
sidebar_label: "🏁 Roadmap"
status:
  - preliminary
  - inprogress
tags:
  - hls4ml
  - implementation
  - training
last_modified: 2026-05-21
author: mbvalentin
---
# hls4ml Implementation Roadmap
<PageMeta />
---

<TBox type="summary" title="How to use this page">

This is the ordered implementation map for `hls4ml-trainable`. Each row links back to the task board so the roadmap stays connected to project status. Add phase-specific implementation notes under this subsection as work starts.

</TBox>

## Phase Notes

Use these pages as the implementation log for each block of work:

- [Phase 0: Architecture map](./phase-0-architecture-map.md)
- [Phase 1: Trainable configuration](./phase-1-trainable-config.md)
- [Phase 2: Graph and flow integration](./phase-2-graph-flow.md)
- [Phase 3: Templates and kernels](./phase-3-templates-kernels.md)
- [Phase 4: Controllers and synchronization](./phase-4-controllers-sync.md)
- [Phase 5: ENABOL bridge and validation](./phase-5-enabol-bridge-validation.md)

## Milestone Shape

The first target is not the full trainable hls4ml backend. The first target is the smallest upstreamable path:

```text
Keras Dense model
  -> hls4ml ModelGraph
  -> trainable HLSConfig
  -> Dense forward
  -> loss endpoint
  -> Dense backpass
  -> raw SGD update
  -> global throttle alpha
  -> two-phase update
  -> CSIM agreement with ENABOL software traces
```

Only after that path is stable should we generalize to two dense layers, activations, convolution, BatchNorm, and more aggressive pipelining.

## Roadmap

| Phase | Step | Task | Main files / objects | Exit criteria |
|---|---|---|---|---|
| 0. Orientation | Freeze the first trainable architecture map. | [HLS4ML-001](/docs/status/tasks?query=HLS4ML-001) | `site/docs/hls4ml/*`, `site/src/data/tasks.js` | The config, flow, writer, endpoint, and controller boundaries are documented enough to start code changes. |
| 1. Config | Define the normalized `Model.Training` schema. | [HLS4ML-001](/docs/status/tasks?query=HLS4ML-001) | `hls4ml/model/graph.py`, ENABOL bridge code | `Trainable`, losses, optimizer, controller, batch size, and trainable precision fields have stable config names. |
| 1. Config | Add parser/accessor methods for trainable config. | [HLS4ML-001](/docs/status/tasks?query=HLS4ML-001) | `HLSConfig` | Code can ask `is_trainable()`, `get_training_config()`, `get_loss_config()`, `get_optimizer_config()`, and `get_controller_config()` without reading raw dict paths everywhere. |
| 1. Config | Add early validation for impossible training configurations. | [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002) | `HLSConfig`, trainable validation pass | Missing losses, unsupported layers, invalid controller names, and ambiguous loss/output counts fail before writing C++. |
| 2. Flow | ✅ Register a Vivado trainable flow. | [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002) | `hls4ml/backends/vivado/vivado_backend.py`, `hls4ml/model/flow` | `vivado:trainable` runs after normal forward template resolution and before writer emission. |
| 2. Flow | ✅ Add trainable validation pass. | [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002) | `hls4ml/backends/vivado/passes/trainable.py` | `vivado:validate_trainable_config` checks the first Dense trainable subset without generating C++. |
| 2. Flow | ✅ Resolve reverse traversal order. | [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002) | `hls4ml/backends/vivado/passes/trainable.py`, `ModelGraph` | Sequential Dense graphs produce `trainable_backward_order`; unsupported branches are rejected. |
| 3. Layer metadata | Add trainable layer attributes. | [HLS4ML-003](/docs/status/tasks?query=HLS4ML-003) | `model/attributes.py`, backend layer wrappers or trainable passes | Layers can hold `trainable`, `grad_in_t`, `grad_out_t`, update types, backward function code, backward config code, and headers. |
| 3. Layer metadata | Normalize ENABOL semantic precision into hls4ml trainable fields. | [HLS4ML-003](/docs/status/tasks?query=HLS4ML-003) / [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | `enabol/precision.py`, ENABOL hls4ml bridge | `PrecisionDict` fields like `gradient`, `update`, and `accumulator` map to explicit hls4ml typedefs. |
| 4. Static assets | Port deprecated trainable source material into the new layout. | [HLS4ML-004](/docs/status/tasks?query=HLS4ML-004) | `templates/vivado/trainable/{common,losses,backprop,optimizers,controllers}` | Static headers copy into generated projects from one trainable umbrella folder without relying on old writer-specific insertion logic. |
| 4. Static assets | Trim old headers to generic template-driven kernels. | [HLS4ML-004](/docs/status/tasks?query=HLS4ML-004) | `trainable/**/*.h` | Dense/loss headers depend on generated config structs, not hardcoded ENABOL assumptions. |
| 5. Loss endpoint | ✅ Generate endpoint metadata for each model output. | [HLS4ML-006](/docs/status/tasks?query=HLS4ML-006) | `hls4ml/backends/vivado/passes/trainable.py` | The first one-output `half_mse` path has ground-truth name, scalar loss name/type, loss input, gradient seed name/type, output layer, and gradient scale metadata. |
| 5. Loss endpoint | Generate loss typedefs and config structs. | [HLS4ML-006](/docs/status/tasks?query=HLS4ML-006) | loss template pass, `parameters.h` emission | `mse` or `half_mse` config compiles and creates `loss_*_t` and `loss_*_grads_t`. |
| 5. Loss endpoint | Decide and document `half_mse` versus full `mse` scaling. | [HLS4ML-006](/docs/status/tasks?query=HLS4ML-006) | loss headers, ENABOL comparison traces | Hardware gradients match the software reference scale. |
| 6. Dense backpass | Implement Dense backward config template. | [HLS4ML-005](/docs/status/tasks?query=HLS4ML-005) | `backends/vivado/passes/*`, `nnet_dense_backprop.h` | Generated Dense config includes data, gradient, update, accumulator, and raw-update typedefs. |
| 6. Dense backpass | Implement Dense backward function template. | [HLS4ML-005](/docs/status/tasks?query=HLS4ML-005) | Dense backward template pass | Writer can emit a Dense backpass call from layer attributes without parsing `config_cpp`. |
| 6. Dense backpass | Verify Dense gradient math against ENABOL software. | [HLS4ML-013](/docs/status/tasks?query=HLS4ML-013) | CSIM traces, ENABOL trace dump | `dL/dW`, `dL/db`, loss, and forward outputs match within fixed-point expectations. |
| 7. Raw update | Implement two-phase raw update storage for Dense. | [HLS4ML-007](/docs/status/tasks?query=HLS4ML-007) | Dense backpass kernel, update buffers | Dense can compute/store raw update separately from applying it. |
| 7. Raw update | Add explicit update-application phase. | [HLS4ML-007](/docs/status/tasks?query=HLS4ML-007) | `trainable/optimizers` and `trainable/controllers` headers | Stored updates can be applied with a supplied scalar alpha. |
| 8. Controllers | Implement `CTRL-NONE`. | [HLS4ML-008](/docs/status/tasks?query=HLS4ML-008) | controller helper header | Alpha is always 1 and the path behaves like ordinary trainable SGD. |
| 8. Controllers | Implement `CTRL-GT-ORDER-0`. | [HLS4ML-008](/docs/status/tasks?query=HLS4ML-008) | controller helper header | Algebraic safe alpha is computed from learning rate and curvature signal. |
| 8. Controllers | Implement `CTRL-GT-ORDER-1`. | [HLS4ML-008](/docs/status/tasks?query=HLS4ML-008) | controller helper header | Persistent alpha state updates synthesizably. |
| 8. Controllers | Implement `CTRL-GT-ORDER-2`. | [HLS4ML-008](/docs/status/tasks?query=HLS4ML-008) | controller helper header | Alpha and velocity state update synthesizably. |
| 8. Controllers | Implement `CTRL-GT-ORDER-2-QA`. | [HLS4ML-008](/docs/status/tasks?query=HLS4ML-008) | controller helper header | Dynamic lower alpha bound from update quantum is represented explicitly. |
| 9. Global sync | Add global norm/reduction accumulators. | [HLS4ML-009](/docs/status/tasks?query=HLS4ML-009) | trainable reduction helpers | Layer-local gradient/update/parameter-change contributions reduce into global metrics. |
| 9. Global sync | Add alpha synchronization point. | [HLS4ML-009](/docs/status/tasks?query=HLS4ML-009) | controller + update phase | The design makes clear which alpha is used to apply which stored update. |
| 9. Global sync | Add fixed-point precision policy for controller metrics. | [HLS4ML-009](/docs/status/tasks?query=HLS4ML-009) | `HLSConfig`, controller config | Norms, curvature, EMA, alpha, and velocity have explicit types and rails. |
| 10. Writer | Add trainable top-level IO hooks. | [HLS4ML-011](/docs/status/tasks?query=HLS4ML-011) | `writer/vivado_writer.py`, firmware templates | Ground truth, loss outputs, train/apply-update flags, and controller inputs/outputs are emitted only when trainable. |
| 10. Writer | Emit trainable config/function attributes. | [HLS4ML-011](/docs/status/tasks?query=HLS4ML-011) | `write_project_cpp()`, `write_parameters()` | Writer emits trainable loss/backward config structs, including batch shift and static learning-rate constants, without semantic inference in kernels. |
| 10. Writer | Update bridge/testbench path for inference-only calls. | [HLS4ML-011](/docs/status/tasks?query=HLS4ML-011) | `myproject_test.cpp`, bridge writer | Existing prediction flow still works with dummy trainable arguments or `train=false`. |
| 11. ENABOL bridge | Build config producer in current ENABOL. | [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | `enabol`, new compile/export module | Current ENABOL can call hls4ml with trainable config without patching generated C++. |
| 11. ENABOL bridge | Write dataset/testbench export for trainable CSIM. | [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) / [HLS4ML-020](/docs/status/tasks?query=HLS4ML-020) / [HLS4ML-037](/docs/status/tasks?query=HLS4ML-037) | `enabol/dataset.py`, `vivado_writer.py` | CSIM receives inputs, ground truth, epoch/shuffle settings, compact console progress, and `tb_data/training/*.dat` traces for first firmware validation. |
| 12. Validation | Run one-layer Dense trainable CSIM. | [HLS4ML-013](/docs/status/tasks?query=HLS4ML-013) | generated Vivado project, CSIM logs | Forward output, loss, gradients, alpha, and updated weights match ENABOL software traces. |
| 12. Validation | Debug mismatch reports into targeted tests. | [HLS4ML-013](/docs/status/tasks?query=HLS4ML-013) | hls4ml tests, ENABOL fixtures | Every observed mismatch maps to a small reproducible fixture. |
| 13. Generalization | Extend to Dense -> activation -> Dense. | [HLS4ML-014](/docs/status/tasks?query=HLS4ML-014) | activation backpass, reverse traversal | Backward traversal through a hidden activation works before convolution/BatchNorm work resumes. |
| 13. Generalization | Revisit Conv2D and BatchNorm tasks after Dense path is stable. | [ENB-014](/docs/status/tasks?query=ENB-014) / [ENB-015](/docs/status/tasks?query=ENB-015) | conv and BN backpasses | Existing conv/BN work is reattached to the new hls4ml-native flow instead of the old writer-heavy path. |
| 14. Optimization | Evaluate delayed-alpha pipelined update schedule. | [HLS4ML-010](/docs/status/tasks?query=HLS4ML-010) | schedule design, HLS experiments | We know whether one-step stale alpha is acceptable and what storage policy avoids weight/backpass inconsistency. |
| 14. Optimization | Decide on update-buffer versus weight double-buffer strategy. | [HLS4ML-010](/docs/status/tasks?query=HLS4ML-010) | hardware schedule design | The design either delays per-layer update until backpass consumes old weights or double-buffers weights safely. |

## Recommended Order

The practical order is:

```text
HLS4ML-001
  -> HLS4ML-002
  -> HLS4ML-003
  -> HLS4ML-004
  -> HLS4ML-006
  -> HLS4ML-005
  -> HLS4ML-007
  -> HLS4ML-008
  -> HLS4ML-009
  -> HLS4ML-011
  -> HLS4ML-012
  -> HLS4ML-013
  -> HLS4ML-014
  -> HLS4ML-010
```

`HLS4ML-010` is intentionally late. The delayed-alpha schedule is promising, but it should not be allowed to obscure the first correctness target. The basic two-phase path gives us a reference implementation. The pipelined schedule can then be judged against it.

## First Correctness Target

The first correct trainable build should be boring:

```text
one Dense layer
one MSE or half-MSE loss
SGD
CTRL-NONE and CTRL-GT-ORDER-0
io_parallel
single batch size policy
no branches
no convolution
no BatchNorm
no delayed-alpha optimization
```

Once this is correct in CSIM, every extra feature has a known-good baseline to compare against.
