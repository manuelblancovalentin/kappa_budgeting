import React from 'react';
import clsx from 'clsx';

const DEFAULT_MONTHS = ['May', 'Jun', 'Jul', 'Aug', 'Sep'];

const DEFAULT_STAGES = [
  {
    title: 'Project framing',
    status: 'done',
    start: 1,
    end: 1,
    details: ['Define closed-loop stability framing.', 'Set documentation structure.'],
  },
  {
    title: 'Float sanity',
    status: 'done',
    start: 1,
    end: 2,
    details: ['Validate one-layer affine drift experiment.', 'Compare baseline and global throttle.'],
  },
  {
    title: 'Quantized sanity',
    status: 'active',
    start: 2,
    end: 3,
    details: ['Add fake fixed-point rails.', 'Track saturation and update geometry.'],
  },
  {
    title: 'Layer coupling',
    status: 'planned',
    start: 3,
    end: 4,
    details: ['Move to two-layer dense models.', 'Inspect inter-layer gradient coupling.'],
  },
  {
    title: 'Handoff',
    status: 'planned',
    start: 4,
    end: 5,
    details: ['Summarize decisions.', 'Package reproducible notebooks and results.'],
  },
];

function normalizeStatus(status) {
  return String(status || 'planned').toLowerCase();
}

function stageProgress(status) {
  const value = normalizeStatus(status);
  if (['done', 'complete', 'completed', 'valid'].includes(value)) {
    return 1;
  }
  if (['active', 'running', 'inprogress', 'in-progress'].includes(value)) {
    return 0.5;
  }
  return 0;
}

function totalProgress(stages) {
  if (!stages.length) {
    return 0;
  }
  const score = stages.reduce((acc, stage) => acc + stageProgress(stage.status), 0);
  return Math.round((100 * score) / stages.length);
}

export default function ProjectTimeline({
  title = 'Project Roadmap',
  months = DEFAULT_MONTHS,
  stages = DEFAULT_STAGES,
  defaultOpen = false,
}) {
  const progress = totalProgress(stages);
  const monthCount = Math.max(months.length, 1);

  return (
    <details className="project-timeline" open={defaultOpen}>
      <summary className="project-timeline__summary">
        <div className="project-timeline__summary-head">
          <span className="project-timeline__title">{title}</span>
          <span className="project-timeline__percent">{progress}%</span>
        </div>
        <div className="project-timeline__bar" aria-label={`${progress}% complete`}>
          {stages.map((stage, index) => (
            <span
              key={`${stage.title}-${index}`}
              className={clsx(
                'project-timeline__segment',
                `project-timeline__segment--${normalizeStatus(stage.status)}`,
              )}
              style={{flexGrow: Math.max((stage.end ?? index + 1) - (stage.start ?? index), 1)}}
              title={`${stage.title}: ${stage.status ?? 'planned'}`}
            >
              <span>{stage.title}</span>
            </span>
          ))}
        </div>
      </summary>

      <div className="project-timeline__expanded">
        <div
          className="project-timeline__months"
          style={{gridTemplateColumns: `minmax(9rem, 1.1fr) repeat(${monthCount}, minmax(5rem, 1fr))`}}
        >
          <span />
          {months.map((month) => (
            <span key={month}>{month}</span>
          ))}
        </div>

        <div className="project-timeline__rows">
          {stages.map((stage, index) => {
            const start = Math.max(Number(stage.start ?? index + 1), 1);
            const end = Math.min(Number(stage.end ?? start), monthCount);
            const status = normalizeStatus(stage.status);

            return (
              <div
                className="project-timeline__row"
                key={`${stage.title}-${index}`}
                style={{gridTemplateColumns: `minmax(9rem, 1.1fr) repeat(${monthCount}, minmax(5rem, 1fr))`}}
              >
                <div className="project-timeline__row-label">
                  <span>{stage.title}</span>
                  <small>{stage.status ?? 'planned'}</small>
                </div>
                <div
                  className={clsx('project-timeline__task', `project-timeline__task--${status}`)}
                  style={{gridColumn: `${start + 1} / ${end + 2}`}}
                >
                  {(stage.details ?? []).map((detail) => (
                    <span key={detail}>{detail}</span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </details>
  );
}
