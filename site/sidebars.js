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
    'intro',
    'current-status',
    'people',
    {
      type: 'category',
      label: '🚀 Onboarding',
      items: [
        'onboarding/setup',
        'onboarding/repo-map',
        'onboarding/common-tasks',
      ],
    },
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
      label: '🔬 Experiments',
      items: [
        'experiments/index',
        'experiments/exp-000a-global-throttle-float-lin1',
        'experiments/exp-000b-global-throttle-qfx-lin1',
        // 'experiments/experiment-template',
      ],
    },
    {
      type: 'category',
      label: '💡 Decisions',
      items: [
        'decisions/index',
        'decisions/adr-template',
      ],
    },
    {
      type: 'category',
      label: '📅 Meetings',
      items: [
        'meetings/index',
      ],
    },
    {
      type: 'category',
      label: '🚚 Handoff',
      items: [
        'handoff/handoff-checklist',
        'handoff/known-issues',
        'handoff/next-steps',
      ],
    },
  ],
};

export default sidebars;
