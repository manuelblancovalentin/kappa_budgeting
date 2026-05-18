import React from 'react';
import {lab, listPeople} from '@site/src/data/people';
import Person from '@site/src/components/Person';

export default function PeopleTable() {
  const people = listPeople();

  return (
    <div className="people-roster">
      <div className="people-roster__lab">
        <span className="people-roster__label">Lab:</span>
        <a href={lab.github} target="_blank" rel="noreferrer">{lab.name}</a>
      </div>
      <table>
        <thead>
          <tr>
            <th>Person</th>
            <th>Role</th>
            <th>GitHub</th>
          </tr>
        </thead>
        <tbody>
          {people.map((person) => (
            <tr key={person.id}>
              <td><Person id={person.id} /></td>
              <td>{person.role}</td>
              <td>
                {person.github ? (
                  <a href={person.github} target="_blank" rel="noreferrer">
                    @{person.handle}
                  </a>
                ) : (
                  <span className="people-roster__muted">Not registered</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
