// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Project Logbook',
  tagline: 'A living record of project status, experiments, decisions, and handoff notes.',
  favicon: 'img/favicon-emoji.svg',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://manuelblancovalentin.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/kappa_budgeting/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'manuelblancovalentin', // Usually your GitHub org/user name.
  projectName: 'kappa_budgeting', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'ENABOL Logbook',
        logo: {
          alt: 'Project Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'intro',
            position: 'left',
            label: '🔎 Overview',
          },
          {
            type: 'dropdown',
            position: 'left',
            label: '📝 Status',
            to: '/docs/current-status',
            items: [
              {type: 'doc', docId: 'current-status', label: 'Current Snapshot'},
              {type: 'doc', docId: 'status/tasks', label: 'Tasks'},
            ],
          },
          {
            type: 'doc',
            docId: 'people',
            position: 'left',
            label: '👥 People',
          },
          {
            type: 'dropdown',
            position: 'left',
            label: '🚀 Onboarding',
            to: '/docs/onboarding/',
            items: [
              {type: 'doc', docId: 'onboarding/setup', label: 'Setup'},
              {type: 'doc', docId: 'onboarding/repo-map', label: 'Repo Map'},
              {type: 'doc', docId: 'onboarding/common-tasks', label: 'Common Tasks'},
            ],
          },
          {
            type: 'dropdown',
            position: 'left',
            label: '🏷️ Docs',
            to: '/docs/',
            items: [
              {type: 'doc', docId: 'formulation/overview', label: 'Formulation'},
              {type: 'doc', docId: 'implementation/code-map', label: 'Implementation'},
              {type: 'doc', docId: 'datasets/datasets-index', label: 'Datasets'},
              {type: 'doc', docId: 'models/models-index', label: 'Models'},
              {type: 'doc', docId: 'ablations/index', label: 'Ablation Tests'},
              {type: 'doc', docId: 'controllers/index', label: 'Controllers'},
              {type: 'doc', docId: 'experiments/index', label: 'Experiments'},
            ],
          },
          {
            type: 'dropdown',
            position: 'left',
            label: '💡 Decisions',
            to: '/docs/decisions/',
            items: [
              {type: 'doc', docId: 'decisions/adr-template', label: 'ADR Template'},
            ],
          },
          {
            type: 'doc',
            docId: 'meetings/index',
            position: 'left',
            label: '📅 Meetings',
          },
          {
            type: 'dropdown',
            position: 'left',
            label: '🚚 Handoff',
            to: '/docs/handoff/',
            items: [
              {type: 'doc', docId: 'handoff/handoff-checklist', label: 'Checklist'},
              {type: 'doc', docId: 'handoff/known-issues', label: 'Known Issues'},
            ],
          },
          {to: '/blog', label: '💬 Lab Log', position: 'left'},
          {
            href: 'https://github.com/manuelblancovalentin/kappa_budgeting',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Project',
            items: [
              {label: 'Overview', to: '/docs/intro'},
              {label: 'Current Status', to: '/docs/current-status'},
              {label: 'People', to: '/docs/people'},
              {label: 'Lab Log', to: '/blog'},
            ],
          },
          {
            title: 'Work',
            items: [
              {label: 'Experiments', to: '/docs/experiments/'},
              {label: 'Ablation Tests', to: '/docs/ablations/'},
              {label: 'Controllers', to: '/docs/controllers/'},
              {label: 'Datasets', to: '/docs/datasets/'},
            ],
          },
          {
            title: 'Coordination',
            items: [
              {label: 'Decisions', to: '/docs/decisions/'},
              {label: 'Meetings', to: '/docs/meetings/'},
              {label: 'Handoff', to: '/docs/handoff/'},
              {label: 'Onboarding', to: '/docs/onboarding/'},
            ],
          },
          {
            title: 'Links',
            items: [
              {
                label: 'GitHub Repository',
                href: 'https://github.com/manuelblancovalentin/kappa_budgeting',
              },
              {
                label: 'Memik Lab',
                href: 'https://github.com/orgs/Memik-Lab',
              },
            ],
          },
        ],
        copyright:
          `Copyright © ${new Date().getFullYear()} <span class="brand-name">🚂ENABOL</span>. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      mermaid: {
        theme: {
          light: 'neutral',
          dark: 'dark',
        },
        options: {
          forceLegacyMathML: true,
        },
      },
    }),

};

export default config;
