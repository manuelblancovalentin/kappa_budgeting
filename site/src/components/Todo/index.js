import React from 'react';
import {Badge} from '@site/src/components/StatusBadges';

export default function Todo({id, owner = 'open', children}) {
  return (
    <aside className="todo-callout">
      <div className="todo-callout__header">
        <Badge status="todo" />
        {id && <code className="todo-callout__id">{id}</code>}
        {owner && <span className="todo-callout__owner">Owner: {owner}</span>}
      </div>
      <div className="todo-callout__body">{children}</div>
    </aside>
  );
}
