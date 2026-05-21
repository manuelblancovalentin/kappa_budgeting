import React from 'react';
import Link from '@docusaurus/Link';
import katex from 'katex';

const CONTROLLER_LABELS = {
  none: 'NONE',
  'gt-order-0': 'GT $\\mathcal{O}(0): \\alpha$',
  'gt-order-1': 'GT $\\mathcal{O}(1): \\dot \\alpha $',
  'gt-order-2': 'GT $\\mathcal{O}(2): \\ddot \\alpha $',
  'gt-order-2-qa': 'GT $\\mathcal{O}(2): \\dot \\alpha$ QA',
};

const CONTROLLER_LINKS = {
  none: '/docs/controllers/ctrl-none',
  'gt-order-0': '/docs/controllers/ctrl-gt-order-0',
  'gt-order-1': '/docs/controllers/ctrl-gt-order-1',
  'gt-order-2': '/docs/controllers/ctrl-gt-order-2',
  'gt-order-2-qa': '/docs/controllers/ctrl-gt-order-2-qa',
};

function normalizeController(controller) {
  return String(controller)
    .trim()
    .replace(/[_\s]+/g, '-')
}


function titleCaseController(controller) {
  return controller
}

function MathLabel({label}) {
  if (typeof label !== 'string') return label;

  return label.split(/(\$[^$]+\$)/g).map((part, index) => {
    if (!part.startsWith('$') || !part.endsWith('$')) return part;

    const latex = part.slice(1, -1);
    const html = katex.renderToString(latex, {
      throwOnError: false,
      strict: false,
      displayMode: false,
    });

    return (
      <span
        key={`${latex}-${index}`}
        className="controller-badge__math"
        dangerouslySetInnerHTML={{__html: html}}
      />
    );
  });
}

export function ControllerBadge({controller, children}) {
  const normalized = normalizeController(controller);
  const label = children ?? CONTROLLER_LABELS[normalized] ?? titleCaseController(normalized);
  const link = CONTROLLER_LINKS[normalized];

  const badge = (
    <span className={`controller-badge controller-badge--${normalized}`}>
      <MathLabel label={label} />
    </span>
  );

  if (!link) return badge;

  return (
    <Link className="controller-badge-link" to={link}>
      {badge}
    </Link>
  );
}
