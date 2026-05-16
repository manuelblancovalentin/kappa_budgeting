import React from 'react';
import clsx from 'clsx';
import useBaseUrl from '@docusaurus/useBaseUrl';

function normalizeMaxWidth(maxWidth) {
  if (typeof maxWidth === 'number') {
    return `${maxWidth}px`;
  }
  return maxWidth || '80%';
}

export default function Figure({
  src,
  alt = '',
  maxWidth = '80%',
  caption,
  label,
  children,
  className,
}) {
  const resolvedSrc = typeof src === 'string' ? useBaseUrl(src) : src;
  const hasCaption = Boolean(children || caption || label);

  return (
    <figure
      className={clsx('doc-figure', className)}
      style={{'--doc-figure-max-width': normalizeMaxWidth(maxWidth)}}
    >
      <img className="doc-figure__image" src={resolvedSrc} alt={alt} />
      {hasCaption && (
        <figcaption className="doc-figure__caption">
          {children || (
            <>
              {label && <strong>{label}</strong>}
              {label && caption && ': '}
              {caption}
            </>
          )}
        </figcaption>
      )}
    </figure>
  );
}
