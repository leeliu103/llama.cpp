# Qwen3-235B TP3 decode extension: 22.34 tokens/s

Branch `feat/qwen3-tp3-decode-megakernels` preserves the source used for a measured
**22.344669 tokens/s TG128** run on three Strix Halo hosts. The selected decode
plan combines a QKV-prefix kernel with two split-MoE kernels per layer. It uses
the execution-extension framework, leaves prefill on stock llama.cpp, and
retains the existing RCCL Direct A2A USB4 transport.

This is a research checkpoint with a measured short-decode result. Full
560-chunk decode perplexity and the complete long-decode benchmark sweep were
not completed. Source development and this documentation were meaningfully
AI-assisted by OpenAI Codex under the user's direction.

## Source identity

The CSE TP3 baseline was reconstructed from upstream commit
`09cedfd6996d2ff90b4e617f56230212005c0166` plus the shipped CSE patch through
`40d1867bd`, from `AMD-GTE-CSE/llama.cpp` branch
`feat/batched-seq-v2-rpc-rccl-tp`. Its build label was 9790. The extension was
adapted from
[execution-extension-experiment](https://github.com/leeliu103/llama.cpp/tree/execution-extension-experiment)
and developed on that reconstructed baseline.

The measured extension source snapshot is `20261028-082059-519127`. The branch
retains the complete runtime source from that snapshot. It already contains
the separately selectable plan12 custom O-projection implementation; the
22.34 tokens/s run selected **plan11**, whose O projection remains stock.
Later P2P, whole-sequence graph, RPC UID-cache lifetime, client-I/O timing and
weight-repacking experiments are not included in this timed checkpoint.

The publication adds this documentation and curated evidence. An inherited
local notes file is omitted; it is not a build or runtime input. Raw local
patch archives and unrelated workspace artifacts are not published. See
[provenance.json](evidence/provenance.json) for the reconstruction and
publication record, and [source-manifest.json](evidence/source-manifest.json)
for the measured source inventory.

The published documentation and evidence replace machine-specific addresses,
host names, network-interface names and filesystem paths with rank labels and
configuration placeholders. Recorded timing samples, execution counters,
source identities and executable/library hashes are preserved. In
`provenance.json`, `original_evidence_sha256` identifies the original private
evidence files and `evidence_sha256` identifies the sanitized published copies.
Unsanitized copies containing machine-specific details are not included in
this branch.

The reconstructed baseline is commit `63226c73`; the exact retained extension
source is commit `d56382b2`. All 3024 retained source files were checked against
the measured snapshot inventory. Documentation and evidence are added after
those source commits.

Verify all 3024 retained source hashes, check the sanitized evidence hashes,
and recompute the recorded timing statistics without building or executing
the model:

```bash
python3 -B docs/tp3-qwen3-decode/verify_source.py
```

## Recorded measurement

| Item | Value |
|---|---|
| Run | `20261028-084704-569075/extension-tg128-run1` |
| Test timestamp | `2026-10-28T12:51:16Z` |
| Model | Qwen3-235B-A22B Q4_K_M, five GGUF shards |
| Workload | `-p 0 -n 128 -d 0 -b 512 -ub 512 -r 3` |
| Mean | 22.344669 tokens/s |
| Within-session sample SD | 0.178042 tokens/s |
| Three samples, printed precision | 22.1519, 22.3790, 22.5030 tokens/s |
| Raw sample durations | 5778273035, 5719637261, 5688125546 ns |
| Selected variant | `split-moe-qkv-prefix-q8-128` |
| Eligible/custom tokens | 385 / 385, including one warmup token |
| Custom regions | 217140 total; 188 per rank per output token |
| Fallback/failed tokens | 0 / 0 |

The three samples belong to **one fresh worker session**, not three independent
worker-session repeats. They support this recorded TG128 result, not a
guaranteed throughput across contexts or hosts. Earlier plan11 measurements
and stock controls used other source snapshots and measurement times.

Primary evidence:

- [result.json](evidence/result.json): timing, workload fields and execution counters.
- [invocation.json](evidence/invocation.json): recorded arguments and client/worker environments with machine details replaced by placeholders.
- [benchmark.log](evidence/benchmark.log): recorded benchmark output with machine details sanitized.
- [build-source-check.json](evidence/build-source-check.json): source stability during the measured build.
- [build-binary-manifest.json](evidence/build-binary-manifest.json): original executable/library hashes.

The original canonical runtime-manifest SHA256 was
`1c3e07dcb32b5ad489988dca9eb066d88b59a75a04d6c46e1834b61a3f9155af`.
This is a historical identifier: sanitizing paths changes the published
manifest's serialized contents. Its executable/library hash entries remain
unchanged. A new build need not have identical binary hashes; these identify
the executables used for the recorded result.

## Implementation and accuracy scope

Plan11 uses native Q4_K/Q6_K model weights and approximate Q8 activation groups
of 128 elements. The attention partition retains the original rotating
16/16/32 query heads and 1/1/2 KV heads across ranks. Flash attention, attention
output projection, KV-cache layout and the 188 Direct A2A collectives per token
remain stock. Stock HIP graph capture and stock operator fusions were enabled.

An earlier plan11 snapshot, `20261028-070726-571346`, passed a 24-token stock
prefill followed by 16 custom decode tokens. All logits were finite and top-1
agreement against the retained stock control was 16/16. Its eight-chunk decode
PPL was **4.0334**, versus **4.0301** for the older stock control
`20261027-214612-251628`: +0.0033 PPL, about +0.0819%. These checks used
`-c 512 -b 1 -ub 1 -ngl 99 -fa on --chunks 8` and the same WikiText-2 test data.
They are earlier-snapshot development evidence, not paired full-accuracy
validation of this exact timed build.

The WikiText-2 `wiki.test.raw` SHA256 was
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
The dataset and model are not included in this branch.

The accuracy-oriented `full-moe` variant remains in the source. Its retained
small handoff matched stock logits exactly with the experimental opt-in and
`GGML_CUDA_DISABLE_STOCK_FUSION=1`. That observation does not establish full
PPL equivalence for every variant or build.

A later attempt to run stock decode PPL for 560 chunks was stopped after
135 chunks because RPC graph metadata accumulated by about 333 MiB per chunk.
A subsequent UID-cache lifetime repair passed short loaded checks, but that
repair is **outside this timed source snapshot**. This checkpoint has not
been validated for long PPL runs; do not interpret partial results as a
completed 560-chunk evaluation.

## Existing cluster prerequisites

The measurement used rank0 as orchestrator and ranks1/2 as peer workers.
No container image, driver installation or host reconfiguration was part of
the experiment. Private process/mount namespaces protected host files and
held peer artifacts in RAM.

Supply these values for the environment where you intend to reproduce the run:

| Variable | Meaning |
|---|---|
| `MODEL` | Path to the first of the five existing Q4_K_M GGUF shards |
| `TP3_KIT` | Existing CSE kit directory on the process's host |
| `TP3_TOOLCHAIN` | Workspace-local directory containing the compatible ROCm overlay and CMake/environment adapters |
| `TP3_RPC_RANK0` | IP address of the rank0 RPC endpoint, without a port |
| `TP3_RPC_RANK1` | IP address of the rank1 RPC endpoint, without a port |
| `TP3_RPC_RANK2` | IP address of the rank2 RPC endpoint, without a port |
| `TP3_SOCKET_IFNAME` | Existing network interface used for RCCL socket bootstrap on that host |

Each host already had the CSE kit, including RCCL 2.30.7 with Direct A2A and
the OdinLink USB4 plugin. All three direct USB4 links were
20.0 Gb/s x2. The recorded kernel was `6.17.0-40-generic`, with
`amd_iommu=on iommu=pt` and GTT `pages_limit=31457280`. These are reproduction
prerequisites, not instructions to change host settings.

Read the five existing shards in place. The peer runner also requires the
existing RPC weight caches; it mounts them read-only instead of creating or
updating cache files.

## Build with a compatible development toolchain

These commands are instructions for a future build. Publishing the branch
does not build it or start any workers. Keep the checkout and all new build
outputs in your private workspace, and use a separate build directory.

The measured build used a workspace-only ROCm 10 overlay with the preinstalled
`_rocm_sdk_core` compiler and runtimes, borrowed ROCm development headers,
and the CSE kit's custom RCCL runtime. Host C/C++ compilers were GCC/G++
14.2.0; the HIP compiler was AMD Clang 23 with LLVM
commit `8f497e0992fb7513f7f78a6f6b6f1056c375e961`. The overlay uses HIP headers
7.15.26333 and borrowed RCCL headers 2.30.4 with the kit's 2.30.7 runtime.

The toolchain overlay and its adapters are not included here. A compatible
HIP development toolchain targeting `gfx1151`, the required BLAS/rocWMMA
headers and libraries, and the custom CSE RCCL/OdinLink runtime must already
be available. The ordinary stock RCCL package does not reproduce this
transport setup. Preserve compiler/runtime provenance when comparing builds.
The following configuration assumes an equivalent overlay layout under
`TP3_TOOLCHAIN`; its environment and CMake adapters must reference your own
SDK and library locations.

From the root of this branch's checkout:

```bash
TP3_SOURCE="$PWD"
TP3_BUILD="${TP3_BUILD:-$TP3_SOURCE/build-tp3-qwen3}"
: "${TP3_TOOLCHAIN:?Set TP3_TOOLCHAIN to the compatible workspace toolchain directory}"
source "$TP3_TOOLCHAIN/rocm10-env.sh"

cmake -S "$TP3_SOURCE" -B "$TP3_BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TP3_TOOLCHAIN/rocm10-toolchain.cmake" \
  -DCMAKE_HIP_FLAGS="--rocm-path=$TP3_TOOLCHAIN/rocm10-overlay --hip-path=$TP3_TOOLCHAIN/rocm10-overlay -include __clang_hip_runtime_wrapper.h" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_HIP=ON -DGGML_HIP_RCCL=ON -DGGML_RPC=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DAMDGPU_TARGETS=gfx1151 \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_NUMBER=9790 \
  -DLLAMA_BUILD_COMMIT=40d1867bd-reconstructed-extension

cmake --build "$TP3_BUILD" \
  --target rpc-server llama-bench llama-perplexity -j 8
```

The explicit HIP wrapper include was required by this SDK compiler. The
`LLAMA_BUILD_COMMIT` string above retains the historical benchmark label; it
is not the Git commit ID of the published branch. Use the source/provenance
files and the branch's Git commit when identifying a new build.

## Runtime environment and TG128 command

Use the same rebuilt libraries on all three workers and the client. For each
process, set `TP3_BIN` to that process's staged build `bin` directory. The
peer build directory exists only inside its private RAM namespace. Set
`TP3_KIT` and `TP3_SOCKET_IFNAME` for each process's host.

```bash
: "${TP3_KIT:?Set TP3_KIT to the existing CSE kit directory}"
: "${TP3_SOCKET_IFNAME:?Set TP3_SOCKET_IFNAME to the existing bootstrap interface}"
: "${TP3_BIN:?Set TP3_BIN to the staged build bin directory for this process}"
export LD_LIBRARY_PATH="$TP3_BIN:$TP3_KIT/odinlink/plugin:$TP3_KIT/rccl:$TP3_KIT/rocm/lib:$TP3_KIT/bin"
export GGML_RPC_RCCL=1
export NCCL_NET_PLUGIN=odl_tb5
export NCCL_NET=OdinLink_TB5
export RCCL_NET_PLUGIN=ODL_TB5
export RCCL_PLUGIN_DIR="$TP3_KIT/odinlink/plugin"
export ODL_TB5_VIRTUAL_SINGLE_DEV=1
export NCCL_SOCKET_IFNAME="$TP3_SOCKET_IFNAME"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DMABUF_ENABLE=0
export RCCL_DIRECT_A2A_ENABLE=1
export RCCL_DIRECT_A2A_MAX_BYTES=65536
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=2
export NCCL_DEBUG=WARN
export AMD_COMGR_CACHE=0
export AMD_COMGR_SAVE_LLVM_TEMPS=0
export PYTHONDONTWRITEBYTECODE=1
export LLAMA_EXECUTION_EXTENSION=1
export LLAMA_EXECUTION_EXTENSION_STATS=1
export LLAMA_EXECUTION_EXTENSION_VARIANT=split-moe-qkv-prefix-q8-128
export GGML_CUDA_ENABLE_EXPERIMENTAL_FULL_MOE=1
```

Start from a clean process environment for NCCL/RCCL/ODL/GGML/extension
overrides. The measured runner filters inherited overrides before setting the
recorded values. In particular, `GGML_CUDA_DISABLE_GRAPHS`,
`GGML_CUDA_DISABLE_STOCK_FUSION` and `GGML_CUDA_DISABLE_MOE_ROUTING_FUSION`
were unset. Profiling, sequence timing and graph-dump logging were disabled.

For the local benchmark and worker, the protected runner sets `TMPDIR` and
`XDG_CACHE_HOME` to newly created directories under its local run directory.
Peer `TMPDIR` points into private RAM, while peer `XDG_CACHE_HOME` remains
the preexisting cache root with its RPC cache rebound read-only. Pointing
the peer cache at empty scratch would change model loading behavior.

After fresh, matching, task-owned workers have been staged and started by the
protected runner, use the recorded workload arguments with your own addresses
and paths:

```bash
: "${MODEL:?Set MODEL to the first existing GGUF shard}"
: "${TP3_RPC_RANK0:?Set TP3_RPC_RANK0 to the rank0 RPC address without a port}"
: "${TP3_RPC_RANK1:?Set TP3_RPC_RANK1 to the rank1 RPC address without a port}"
: "${TP3_RPC_RANK2:?Set TP3_RPC_RANK2 to the rank2 RPC address without a port}"
RPCS="${TP3_RPC_RANK0}:50052,${TP3_RPC_RANK1}:50052,${TP3_RPC_RANK2}:50052"
"$TP3_BIN/llama-bench" \
  -m "$MODEL" -sm tensor --rpc "$RPCS" -dev RPC0/RPC1/RPC2 \
  -ngl 99 -fa on -p 0 -n 128 -d 0 -b 512 -ub 512 -r 3 -o jsonl
```

For reference, the recorded worker argument vectors, with address placeholders,
were:

```text
rank0: rpc-server -H ${TP3_RPC_RANK0} -p 50052 -t 8
rank1: rpc-server -H ${TP3_RPC_RANK1} -p 50052 -t 8 -c
rank2: rpc-server -H ${TP3_RPC_RANK2} -p 50052 -t 8 -c
```

These argument vectors do not reproduce the protective namespaces by
themselves. Do not launch them against an occupied cluster or reuse another
user's worker session. The original runner starts fresh workers for each
independent run, refuses occupied hosts/ports, and cleans up only its own
processes.

## Original orchestration dependencies

The protected research runner is not included as a standalone script in this
branch. It depends on staging, supervision, profiling and evidence modules,
plus an external runner, `restricted_exec.py`, Python `paramiko`, bubblewrap,
Landlock and seccomp. Its original source/build paths were fixed to the
development workspace. Running an unchanged local copy does not necessarily
select this published snapshot; its build directories may contain later
experimental binaries.

The historical high-level selection was:

```text
--cases extension --tests tg128 --repetitions 3
--extension-variant split-moe-qkv-prefix-q8-128
--enable-experimental-full-moe --timeout 1800
```

To use that orchestration for a new reproduction, explicitly point its build
and artifact inputs at a separately built copy of this branch while retaining
its private staging, busy-host checks and owned-worker cleanup. No worker
autostart script is provided by this publication.

Accept a timing result only when startup selects
`qwen3moe.decode.split-moe-qkv-prefix-q8-128.v1`, every rank reports 188
first-sequence regions, all eligible decode tokens execute custom work, and
fallback/failed counters remain zero. Keep debug/profile runs separate from
throughput measurements and retain all samples with their worker-session
identities.
