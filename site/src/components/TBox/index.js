import React from 'react';

const DEFAULT_TITLES = {
  summary: 'Summary',
  warning: 'Warning',
  todo: 'TODO',
};

function normalizeType(type) {
  return String(type ?? 'summary')
    .trim()
    .toLowerCase()
    .replace(/[_\s]+/g, '-');
}

export default function TBox({type = 'summary', title, children}) {
  const normalized = normalizeType(type);
  const resolvedTitle = title ?? DEFAULT_TITLES[normalized] ?? 'Note';

  return (
    <section className={`tbox tbox--${normalized}`}>
      <div className="tbox__title">{resolvedTitle}</div>
      <div className="tbox__body">{children}</div>
    </section>
  );
}
