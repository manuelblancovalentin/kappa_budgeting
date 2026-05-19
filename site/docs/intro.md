---
sidebar_position: 1
status:
  - draft
tags:
  - todo
  - placeholder
  - overview
last_modified: 2026-05-15
author: mbvalentin
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

## How to read the docs
* [🔎 Project Overview](./intro.md) (this page) is the global entry point to understand this project.
* [📝 Current Status](./current-status.md) summarizes the current state of the project and next steps.
* [👥 People](./people.md) describes the current and past contributors to the project.
* [🚀 Onboarding](./onboarding/setup.md) is a step-by-step guide to set up the code and run the first experiments.
* [🏷️ Docs ](./index.md) contains the technical documentation of the formulation, algorithms, and hardware implementation details.
* [💡 Decisions ](./decisions/index.md) is the decision log of important design choices and their rationale.
* [📅 Meetings ](./meetings/index.md) contains notes and recordings of project meetings.
* [🚚 Handoff ](./handoff/index.md) contains the documentation and materials to hand off the project to another collaborator.
* [💬 Lab log ](/blog) is the public-facing log of experiments, results, and insights.

## Where to start

I, <Person id="mbvalentin" />, personally recommend starting with:

1. Read the [📝 Current Status](./current-status.md).
2. Follow the [🚀 Onboarding](./onboarding/setup.md) guide.
3. Read the documentation in [🏷️ Docs ](./index.md) as needed to understand the experiments and algorithms.
4. Check the latest [💬 Lab log ](/blog) entry.
5. Pick an item from [active tasks](./status/tasks.md).

## How to contribute to the docs
The documentation page is based on [🦖 Docusaurus](https://docusaurus.io/), a React-based static site generator. The docs are written in Markdown with some custom components for styling and interactivity. 

<TBox type="todo" title="TODO">
- [ ] I, <Person id="mbvalentin" />, intend to do a tutorial walkthrough on how to modify the docs and add new stuff. In the meantime I suggest you use Codex with care, or simply log stuff in the blog or in a google docs/plain markdown text locally and I will add entries to the docs as we go.
</TBox>

## Repository links

- 👾 GitHub repository (for now): [github.com/manuelblancovalentin/kappa-budgeting](https://github.com/manuelblancovalentin/kappa-budgeting)
- Main branch: `main`
- Shared drive / data location: `kona`