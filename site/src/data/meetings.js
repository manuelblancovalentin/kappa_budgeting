const meetings = [
  {
    id: 'MTG-2026-05-18-STATUS',
    title: 'Project status and docs structure',
    date: '2026-05-18',
    status: 'completed',
    type: 'working-session',
    attendees: ['mbvalentin'],
    summary: 'Organized the project docs around registries for datasets, models, ablation tests, controllers, experiments, tasks, and handoff material.',
    links: {
      docs: [
        {label: 'Current Status', href: '/docs/current-status'},
        {label: 'Tasks', href: '/docs/status/tasks'},
      ],
    },
  },
  {
    id: 'MTG-2026-05-25-ABLATION-PLAN',
    title: 'Ablation ladder planning',
    date: '2026-05-25',
    status: 'planned',
    type: 'research-planning',
    attendees: ['mbvalentin'],
    summary: 'Pick the next fixed-point stress test and decide which controller variant should be implemented first.',
    links: {
      docs: [
        {label: 'Ablation Tests', href: '/docs/ablations/'},
        {label: 'Controllers', href: '/docs/controllers/'},
      ],
    },
  },
];

export default meetings;
