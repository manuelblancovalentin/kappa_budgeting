import React from 'react';
import clsx from 'clsx';

const KEYWORDS = new Set([
  'def',
  'return',
  'if',
  'else',
  'elif',
  'for',
  'while',
  'in',
  'with',
  'as',
  'break',
  'continue',
]);

function trimOuterBlankLines(lines) {
  let start = 0;
  let end = lines.length;
  while (start < end && lines[start].trim() === '') {
    start += 1;
  }
  while (end > start && lines[end - 1].trim() === '') {
    end -= 1;
  }
  return lines.slice(start, end);
}

function splitComment(line) {
  const index = line.indexOf('#');
  if (index < 0) {
    return {code: line, comment: ''};
  }
  return {
    code: line.slice(0, index).replace(/\s+$/, ''),
    comment: line.slice(index),
  };
}

function renderCodeTokens(code) {
  const tokens = code.split(/(\b[A-Za-z_]\w*\b)/g);
  return tokens.map((token, index) => {
    if (KEYWORDS.has(token)) {
      return <span key={`${token}-${index}`} className="pseudo-kw">{token}</span>;
    }
    return <React.Fragment key={`${token}-${index}`}>{token}</React.Fragment>;
  });
}

function sourceIndentLevel(rawLine) {
  const leading = rawLine.match(/^\s*/)?.[0].length || 0;
  return Math.min(3, Math.floor(leading / 4));
}

function inferPythonIndentLevels(lines) {
  let level = 0;
  return lines.map((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return 0;
    }

    if (/^(end\b|else\b|elif\b)/.test(trimmed)) {
      level = Math.max(0, level - 1);
    }

    const current = level;
    if (trimmed.endsWith(':') && !/^(return\b)/.test(trimmed)) {
      level = Math.min(3, level + 1);
    }
    return current;
  });
}

function AlgorithmLine({rawLine, indentLevel}) {
  const text = rawLine.trimStart();
  const {code, comment} = splitComment(text);

  return (
    <li className={!text ? 'algorithm-line-empty' : undefined}>
      {text && (
        <span className={clsx('algorithm-line', indentLevel > 0 && `pseudo-indent-${indentLevel}`)}>
          <span className="algorithm-code">{renderCodeTokens(code)}</span>
          {comment && <span className="algorithm-comment">{comment}</span>}
        </span>
      )}
    </li>
  );
}

export default function Algorithm({
  id,
  title = 'Algorithm',
  content = '',
  caption,
  className,
  inferIndent = true,
}) {
  const lines = trimOuterBlankLines(String(content).replace(/\t/g, '    ').split('\n'));
  const hasSourceIndent = lines.some((line) => line.trim() && sourceIndentLevel(line) > 0);
  const inferredLevels = inferIndent && !hasSourceIndent ? inferPythonIndentLevels(lines) : null;

  return (
    <>
      {id && <span id={id} className="doc-anchor" aria-hidden="true" />}
      <div className={clsx('pseudo algorithm', className)}>
        <div className="pseudo-title">{title}</div>
        <div className="pseudo-code algorithm-code-block">
          <ol>
            {lines.map((line, index) => (
              <AlgorithmLine
                key={`${index}-${line}`}
                rawLine={line}
                indentLevel={inferredLevels ? inferredLevels[index] : sourceIndentLevel(line)}
              />
            ))}
          </ol>
        </div>
        {caption && <div className="pseudo-caption">{caption}</div>}
      </div>
    </>
  );
}
