import React from 'react';
import clsx from 'clsx';
import katex from 'katex';

function normalizeCssLength(value, fallback) {
  if (typeof value === 'number') {
    return `${value}px`;
  }
  return value || fallback;
}

function normalizeScale(scale) {
  if (typeof scale === 'number') {
    return String(scale);
  }
  return scale || '1.08';
}

function renderLatex(value) {
  return {
    __html: katex.renderToString(value, {
      displayMode: false,
      throwOnError: false,
      strict: false,
    }),
  };
}

function DiagramText({value, math = true}) {
  if (!value) {
    return null;
  }
  if (!math) {
    return <>{value}</>;
  }
  return <span dangerouslySetInnerHTML={renderLatex(value)} />;
}

function SeqNode({item}) {
  const kind = item.kind || 'box';
  return (
    <div className={clsx('seq-diagram__node', `seq-diagram__node--${kind}`)}>
      <div className="seq-diagram__label">
        <DiagramText value={item.label} math={item.math !== false} />
      </div>
      {item.sublabel && (
        <div className="seq-diagram__sublabel">
          <DiagramText value={item.sublabel} math={item.sublabelMath !== false} />
        </div>
      )}
    </div>
  );
}

function SeqArrow({arrow}) {
  const label = typeof arrow === 'string' ? arrow : arrow?.label;
  const math = typeof arrow === 'string' ? true : arrow?.math !== false;
  return (
    <div className="seq-diagram__arrow" aria-hidden={!label}>
      {label && (
        <div className="seq-diagram__arrow-label">
          <DiagramText value={label} math={math} />
        </div>
      )}
      <div className="seq-diagram__arrow-line" />
    </div>
  );
}

export default function SeqDiagram({
  items = [],
  arrows = [],
  maxWidth = '100%',
  gap = '5.4em',
  arrowPadding = '0.7em',
  scale = 1.08,
  className,
}) {
  const arrowAt = (index) => arrows[index] || {};

  return (
    <div
      className={clsx('seq-diagram', className)}
      style={{
        '--seq-diagram-max-width': normalizeCssLength(maxWidth, '100%'),
        '--seq-diagram-gap': normalizeCssLength(gap, '5.4em'),
        '--seq-diagram-arrow-padding': normalizeCssLength(arrowPadding, '0.7em'),
        '--seq-diagram-scale': normalizeScale(scale),
      }}
    >
      <div className="seq-diagram__inner">
        {items.map((item, index) => (
          <React.Fragment key={item.id || `${item.label}-${index}`}>
            <SeqNode item={item} />
            {index < items.length - 1 && <SeqArrow arrow={arrowAt(index)} />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
