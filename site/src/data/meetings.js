const meetings = [
  {
    id: 'MTG-2026-05-20-STATUS',
    title: 'Alan, Seda, Manu sync and next steps',
    date: '2026-05-20',
    status: 'completed',
    type: 'working-session',
    attendees: ['mbvalentin', 'alan-guo', 'seda-ogrenci'],
    summary: 'Presented documentation. Discussed batch norm and global-throttling planning.',
    notesDoc: {
      label: 'Click here for full meeting notes',
      href: '/docs/meetings/2026-05-20-meeting-seda-alan-manu',
    },
    notes: `## Notes
      - Andrew/Darell tested batchnorm. It improves accuracy in software but still suffers from drift (good for ENABOL).
        * They will now freeze dense and try to train conv layers only. 
      - Alan found that kappa throttling is uneven across layers, this distorts learning. 
        * He will work on the backpass for BN (even if not training it, we need propagated gradients).
        * He also needs to consolidate the logging to always keep the pre-drift baseline, post-drift initial + recovered (3) points.
      - We basically need 3 tests in the mid-term (without dense training):
        * Only training conv layers (no batchnorm training).
        * Only training batchnorm layers (no conv training).
        * Training both conv and batchnorm layers together. 
      - Manu presented the global throttling. Seems effective.
        * Next steps are to do all the rest of ablation tests with throttling.
        * Pull hls4ml and run first csim with synthesizable controller for global throttling.
      `,
    links: {
      docs: [
        {label: 'Current Status', href: '/docs/status/current-status'},
        {label: 'Tasks', href: '/docs/status/tasks'},
      ],
    },
  },
  // {
  //   id: 'MTG-2026-05-25-ABLATION-PLAN',
  //   title: 'Ablation ladder planning',
  //   date: '2026-05-25',
  //   status: 'planned',
  //   type: 'research-planning',
  //   attendees: ['mbvalentin'],
  //   summary: 'Pick the next fixed-point stress test and decide which controller variant should be implemented first.',
  //   links: {
  //     docs: [
  //       {label: 'Ablation Tests', href: '/docs/ablations/'},
  //       {label: 'Controllers', href: '/docs/controllers/'},
  //     ],
  //   },
  // },
];

export default meetings;
