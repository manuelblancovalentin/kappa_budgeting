import React from 'react';

function textFromChildren(node) {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return '';
  }

  if (typeof node === 'string' || typeof node === 'number') {
    return String(node);
  }

  if (Array.isArray(node)) {
    return node.map(textFromChildren).join('');
  }

  if (React.isValidElement(node)) {
    return textFromChildren(node.props.children);
  }

  return String(node);
}

function cleanTerminalText(text) {
  return String(text)
    .replace(/^\n/, '')
    .replace(/\n\s*$/, '');
}

export default function Terminal({children, content, title = 'terminal'}) {
  const body = cleanTerminalText(content ?? textFromChildren(children));

  return (
    <div className="terminal-box" role="region" aria-label={title}>
      <div className="terminal-box__bar">
        <span className="terminal-box__dot terminal-box__dot--red" />
        <span className="terminal-box__dot terminal-box__dot--yellow" />
        <span className="terminal-box__dot terminal-box__dot--green" />
        <span className="terminal-box__title">{title}</span>
      </div>
      <pre className="terminal-box__body"><code>{body}</code></pre>
    </div>
  );
}
