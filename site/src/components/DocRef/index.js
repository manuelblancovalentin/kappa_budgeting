import React from 'react';

export default function DocRef({target, id, label, children, className}) {
  const anchor = target || id;
  if (!anchor) {
    return <>{children || label}</>;
  }

  const href = anchor.startsWith('#') ? anchor : '#' + anchor;

  return (
    <a href={href} className={className} data-noBrokenLinkCheck>
      {children || label || href}
    </a>
  );
}
