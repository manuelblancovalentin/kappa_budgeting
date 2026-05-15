import React from 'react';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import StatusBadges, {Badge} from '@site/src/components/StatusBadges';

function asList(value) {
  if (!value) {
    return [];
  }

  if (Array.isArray(value)) {
    return value.filter(Boolean);
  }

  return String(value)
    .split(/[;,|]/)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function formatDate(value) {
  if (!value) {
    return null;
  }

  if (value instanceof Date) {
    return value.toISOString().slice(0, 10);
  }

  return String(value);
}

export default function PageMeta({showStatus = true, showTags = true, showLastModified = true}) {
  const {frontMatter} = useDoc();
  const status = asList(frontMatter.status);
  const tags = asList(frontMatter.tags);
  const lastModified = formatDate(frontMatter.last_modified ?? frontMatter.lastModified);

  if (!status.length && !tags.length && !lastModified) {
    return null;
  }

  return (
    <div className="page-meta">
      {showStatus && status.length > 0 && (
        <StatusBadges statuses={status} label="Status:" />
      )}
      {showTags && tags.length > 0 && (
        <div className="page-meta__row">
          <span className="page-meta__label">Tags:</span>
          <span className="page-meta__badges">
            {tags.map((tag) => (
              <Badge key={tag} status={tag} />
            ))}
          </span>
        </div>
      )}
      {showLastModified && lastModified && (
        <div className="page-meta__row page-meta__row--date">
          <span className="page-meta__label">Last modified:</span>
          <time dateTime={lastModified}>{lastModified}</time>
        </div>
      )}
    </div>
  );
}
