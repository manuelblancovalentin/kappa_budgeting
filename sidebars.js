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
      label: 'Onboarding',
      items: [
        'onboarding/setup',
        'onboarding/repo-map',
        'onboarding/common-tasks',
      ],
    },
    {
      type: 'category',
      label: 'Background',
      items: [
        'background/overview',
        'background/reading-list',
      ],
    },
    {
      type: 'category',
      label: 'Implementation',
      items: [
        'implementation/architecture',
        'implementation/code-map',
      ],
    },
    {
      type: 'category',
      label: 'Experiments',
      items: [
        'experiments/index',
        'experiments/experiment-template',
      ],
    },
    {
      type: 'category',
      label: 'Decisions',
      items: [
        'decisions/index',
        'decisions/adr-template',
      ],
    },
    {
      type: 'category',
      label: 'Meetings',
      items: [
        'meetings/index',
      ],
    },
    {
      type: 'category',
      label: 'Handoff',
      items: [
        'handoff/handoff-checklist',
        'handoff/known-issues',
        'handoff/next-steps',
      ],
    },
  ],
};

export default sidebars;
