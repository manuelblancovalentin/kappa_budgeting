---
sidebar_position: 1
status:
  - draft
tags:
  - todo
  - placeholder
  - overview
last_modified: 2026-05-15
---
# 🔎 Project Overview
<PageMeta />
---

This page explains the project to someone joining it for the first time.

<TBox type="summary" title="Project snapshot">
<ENABOL /> is currently focused on controlled ablation experiments for closed-loop online training stability. The immediate work is to move from one-layer floating-point sanity checks toward quantized and multi-layer tests while keeping the documentation and experiment registry handoff-ready.
</TBox>

## One-sentence summary

<ENABOL /> is a hardware-aware online-training framework for studying how fixed-point neural networks can keep learning stable under drift, quantization, saturation, and limited on-edge precision.

## Project goal

Build a small, reproducible ablation harness that separates two ideas:

- static $\kappa$-budgeting as a representational gain rail,
- dynamic global throttling as a closed-loop learning-stability controller.

The goal is to understand when each mechanism helps, when it disrupts learning, and how to document the resulting methodology clearly enough for another collaborator to continue the work.

## Project roadmap

<ProjectTimeline
  title="ENABOL Ablation Roadmap"
  months={['May 2026', 'Jun', 'Jul', 'Aug', 'Sep']}
  stages={[
    {
      title: 'Framing',
      status: 'done',
      start: 1,
      end: 1,
      details: ['Closed-loop stability formulation', 'Documentation skeleton'],
    },
    {
      title: 'Float LIN1',
      status: 'done',
      start: 1,
      end: 2,
      details: ['EXP-000A', 'Affine drift', 'Global throttle sanity'],
    },
    {
      title: 'QFX LIN1',
      status: 'active',
      start: 2,
      end: 3,
      details: ['EXP-000B', 'Rails and saturation', 'Quantized diagnostics'],
    },
    {
      title: 'Multi-layer',
      status: 'planned',
      start: 3,
      end: 4,
      details: ['Two Dense layers', 'Inter-layer gradients', 'Activation rails'],
    },
    {
      title: 'Benchmarks',
      status: 'planned',
      start: 4,
      end: 5,
      details: ['Dataset/model registry expansion', 'Compare controllers'],
    },
    {
      title: 'Handoff',
      status: 'planned',
      start: 5,
      end: 5,
      details: ['Final notes', 'Decision log', 'Reusable notebooks'],
    },
  ]}
/>


## Why this matters

Online training on edge hardware is not only a forward-pass quantization problem. Once learning is enabled, the weights become part of a closed-loop dynamical system. This project studies how to keep that loop stable while preserving the descent direction enough for learning to continue.

## Current owner

| Role | Person |
|---|---|
| Primary owner | NAME |
| Faculty advisor | NAME |
| Active contributors | NAME, NAME |
| Former contributors | NAME, NAME |

## Where to start

1. Read the [current status](./current-status.md).
2. Follow the [setup guide](./onboarding/setup.md).
3. Check the latest [lab log](/blog).
4. Review [known issues](./handoff/known-issues.md).
5. Pick an item from [next steps](./handoff/next-steps.md).

## Repository links

- GitHub repository:
- Main branch:
- Important experimental branches:
- Related papers / proposals:
- Shared drive / data location: