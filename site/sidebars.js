// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    'formulation/index',
    {
      type: 'category',
      label: '⟦♾️⟧ Formulation',
      items: [
        'formulation/overview',
        'formulation/lipschitzness-stability',
        'formulation/kappa-budgeting',
        'formulation/global-throttle',
        'formulation/how-this-fits',
        'formulation/quantization',
      ],
    },
    {
      type: 'category',
      label: '🔩 Implementation',
      items: [
        'implementation/code-map',
        'implementation/dataset',
        'implementation/dtypes',
        'implementation/history',
        'implementation/nn',
        'implementation/precision',
        'implementation/utils',
        'implementation/quantization',
        'implementation/architecture',
        'implementation/training-loop',
      ],
    },
    {
      type: 'category',
      label: '📚 Datasets',
      items: [
        'datasets/datasets-index',
        'datasets/affine-linear-000',
      ],
    },
    {
      type: 'category',
      label: '📦 Models',
      items: [
        'models/models-index',
        'models/dense1-linear-nobias-000',
      ],
    },
    {
      type: 'category',
      label: '🥊 Ablation Tests',
      items: [
        'ablations/index',
        'ablations/st-000-high-learning-rate-sanity',
        'ablations/st-001-distribution-shift',
        'ablations/st-002-transient-spike',
        'ablations/st-003-rail-saturation',
        'ablations/st-004-update-dead-zone',
        'ablations/st-005-optimizer-state-precision',
        'ablations/st-006-naive-safety-mechanisms',
        'ablations/st-007-kappa-projection-mismatch',
      ],
    },
    {
      type: 'category',
      label: '🕹️ Controllers',
      items: [
        'controllers/index',
        'controllers/ctrl-none',
        'controllers/ctrl-gt-order-0',
        'controllers/ctrl-gt-order-1',
        'controllers/ctrl-gt-order-2',
        'controllers/ctrl-gt-order-2-qa',
      ],
    },
    {
      type: 'category',
      label: '🔬 Experiments',
      items: [
        'experiments/index',
        'experiments/exp-000a-global-throttle-float-lin1',
        'experiments/exp-000b-global-throttle-qfx-lin1',
        // 'experiments/experiment-template',
      ],
    },
  ],
  statusSidebar: [
    'status/current-status',
    'status/tasks',
  ],
  onBoardingSidebar: [
    'onboarding/index',
    'onboarding/setup',
    'onboarding/repo-map',
    'onboarding/docusaurus',
  ],
  HandOffSidebar: [
    'handoff/index',
    'handoff/handoff-checklist',
    'handoff/known-issues',
  ],
  MeetingsSidebar: [
    'meetings/index',
    'meetings/2026-05-20-meeting-seda-alan-manu',
  ],
};

export default sidebars;
