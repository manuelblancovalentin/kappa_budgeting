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
        'formulation/kappa-budgeting',
        'formulation/lipschitzness-stability',
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
        'implementation/precision',
        'implementation/nn',
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
        'experiments/global-throttle-sanity',
        'experiments/global-throttle-sanity-quantization',
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
