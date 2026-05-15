import React from 'react';

const STATUS_LABELS = {
  valid: 'Valid',
  done: 'Done',
  passed: 'Passed',
  running: 'Running',
  preliminary: 'Preliminary',
  planned: 'Planned',
  failing: 'Failing',
  failed: 'Failed',
  invalid: 'Invalid',
  'not-run': 'Not run yet',
  draft: 'Draft',
  todo: 'TODO',
  placeholder: 'Placeholder',
  revise: 'Needs revision',
  inprogress: 'In progress',
};

function normalizeStatus(status) {
  return String(status)
    .trim()
    .toLowerCase()
    .replace(/[_\s]+/g, '-')
    .replace(/^not-run-yet$/, 'not-run');
}

function splitStatuses(statuses, status) {
  const source = statuses ?? status ?? '';

  if (Array.isArray(source)) {
    return source.map(normalizeStatus).filter(Boolean);
  }

  return String(source)
    .split(/[;,|]/)
    .map(normalizeStatus)
    .filter(Boolean);
}

function titleCaseStatus(status) {
  return status
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export function Badge({status, children}) {
  const normalized = normalizeStatus(status);
  const label = children ?? STATUS_LABELS[normalized] ?? titleCaseStatus(normalized);

  return (
    <span className={`status-badge status-badge--${normalized}`}>
      {label}
    </span>
  );
}

export default function StatusBadges({statuses, status, label = 'Status:'}) {
  const values = splitStatuses(statuses, status);

  return (
    <div className="status-line">
      {label !== false && label !== null && (
        <span className="status-line__label">{label}</span>
      )}
      {values.map((value) => (
        <Badge key={value} status={value} />
      ))}
    </div>
  );
}
