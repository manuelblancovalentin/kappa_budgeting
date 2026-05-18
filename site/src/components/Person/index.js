import React from 'react';
import clsx from 'clsx';
import {getPerson} from '@site/src/data/people';

export default function Person({id, handle, label, showRole = false, className}) {
  const person = getPerson(id || handle);
  const display = person ? (label || person.name) : (label || id || handle || 'Unknown person');
  const title = person ? [person.name, person.role].filter(Boolean).join(' - ') : undefined;
  const body = (
    <>
      <span className="person-chip__handle">@</span>
      <span className="person-chip__name">{display}</span>
      {showRole && person?.role && <span className="person-chip__role">{person.role}</span>}
    </>
  );

  if (person?.github) {
    return (
      <a className={clsx('person-chip', 'person-chip--link', className)} href={person.github} target="_blank" rel="noreferrer" title={title}>
        {body}
      </a>
    );
  }

  return (
    <span className={clsx('person-chip', className)} title={title}>
      {body}
    </span>
  );
}
