import React, {useMemo, useState} from 'react';
import tasks from '@site/src/data/tasks';
import Person from '@site/src/components/Person';
import {Badge} from '@site/src/components/StatusBadges';
import {getPerson} from '@site/src/data/people';

const STATUS_ORDER = {
  blocked: 0,
  inprogress: 1,
  todo: 2,
  completed: 3,
  done: 3,
  dropped: 4,
};

const ALL = '__all__';
const UNASSIGNED = '__unassigned__';
const UNDEFINED_DATE = 'UNDEFINED';

function normalizeStatus(status) {
  if (status === 'done') return 'completed';
  return status;
}

function normalizeList(value) {
  if (!value) return [];
  return Array.isArray(value) ? value.map(normalizeStatus) : [normalizeStatus(value)];
}

function unique(values) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

function displayDate(value) {
  return value || UNDEFINED_DATE;
}

function taskText(task) {
  return [
    task.id,
    task.title,
    task.status,
    task.priority,
    task.type,
    task.area,
    task.notes,
    task.created,
    task.due_date,
    task.start_date,
    task.end_date,
    ...(task.owners || []),
  ].filter(Boolean).join(' ').toLowerCase();
}

function matches(task, filters) {
  const statuses = normalizeList(filters.statuses ?? filters.status);
  const owners = normalizeList(filters.owners ?? filters.owner);
  const types = normalizeList(filters.types ?? filters.type);
  const areas = normalizeList(filters.areas ?? filters.area);
  const priorities = normalizeList(filters.priorities ?? filters.priority);
  const query = String(filters.query || '').trim().toLowerCase();

  if (statuses.length && !statuses.includes(normalizeStatus(task.status))) return false;
  if (owners.length) {
    const taskOwners = task.owners || [];
    const wantsUnassigned = owners.includes(UNASSIGNED);
    const wantsNamedOwner = owners.some((owner) => owner !== UNASSIGNED && taskOwners.includes(owner));
    if (!(wantsUnassigned && taskOwners.length === 0) && !wantsNamedOwner) return false;
  }
  if (types.length && !types.includes(task.type)) return false;
  if (areas.length && !areas.includes(task.area)) return false;
  if (priorities.length && !priorities.includes(task.priority)) return false;
  if (query && !taskText(task).includes(query)) return false;
  return true;
}

function sortTasks(a, b) {
  const statusDelta = (STATUS_ORDER[normalizeStatus(a.status)] ?? 99) - (STATUS_ORDER[normalizeStatus(b.status)] ?? 99);
  if (statusDelta) return statusDelta;
  const priorityOrder = {high: 0, medium: 1, low: 2};
  const priorityDelta = (priorityOrder[a.priority] ?? 99) - (priorityOrder[b.priority] ?? 99);
  if (priorityDelta) return priorityDelta;
  return String(b.created || '').localeCompare(String(a.created || ''));
}

function TaskLinks({links}) {
  const groups = Object.entries(links || {}).filter(([, values]) => values?.length);
  if (!groups.length) return <span className="task-empty">None</span>;

  return (
    <span className="task-links">
      {groups.flatMap(([group, values]) =>
        values.map((link) => (
          <a key={`${group}-${link.href}-${link.label}`} href={link.href}>
            {link.label}
          </a>
        )),
      )}
    </span>
  );
}

function Owners({owners = []}) {
  if (!owners.length) return <span className="task-empty">Unassigned</span>;
  return (
    <span className="task-owners">
      {owners.map((owner) => (
        <Person key={owner} id={owner} />
      ))}
    </span>
  );
}

function TaskDates({task}) {
  return (
    <dl className="task-dates">
      <div><dt>Due</dt><dd>{displayDate(task.due_date)}</dd></div>
      <div><dt>Start</dt><dd>{displayDate(task.start_date)}</dd></div>
      <div><dt>End</dt><dd>{displayDate(task.end_date)}</dd></div>
    </dl>
  );
}

function SelectFilter({label, value, values, onChange, renderLabel = (item) => item}) {
  return (
    <label className="task-filter">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value={ALL}>All</option>
        {values.map((item) => (
          <option key={item} value={item}>
            {renderLabel(item)}
          </option>
        ))}
      </select>
    </label>
  );
}

function TaskFilters({filters, setFilters, counts, options}) {
  const update = (key, value) => {
    setFilters((current) => ({...current, [key]: value}));
  };

  const clear = () => {
    setFilters({
      status: ALL,
      owner: ALL,
      type: ALL,
      area: ALL,
      priority: ALL,
      query: '',
    });
  };

  return (
    <div className="task-filter-panel">
      <div className="task-filter-panel__top">
        <div>
          <strong>Filter tasks</strong>
          <span>{counts.visible} of {counts.total} shown</span>
        </div>
        <button type="button" onClick={clear}>Reset</button>
      </div>
      <div className="task-filter-grid">
        <label className="task-filter task-filter--search">
          <span>Search</span>
          <input
            type="search"
            value={filters.query}
            placeholder="title, ID, notes, area, date..."
            onChange={(event) => update('query', event.target.value)}
          />
        </label>
        <SelectFilter label="Status" value={filters.status} values={options.statuses} onChange={(value) => update('status', value)} />
        <SelectFilter label="Owner" value={filters.owner} values={options.owners} onChange={(value) => update('owner', value)} renderLabel={(id) => id === UNASSIGNED ? 'Unassigned' : getPerson(id)?.shortName || getPerson(id)?.name || id} />
        <SelectFilter label="Type" value={filters.type} values={options.types} onChange={(value) => update('type', value)} />
        <SelectFilter label="Area" value={filters.area} values={options.areas} onChange={(value) => update('area', value)} />
        <SelectFilter label="Priority" value={filters.priority} values={options.priorities} onChange={(value) => update('priority', value)} />
      </div>
      <div className="task-filter-panel__quick">
        <button type="button" onClick={() => update('status', 'todo')}>Unassigned</button>
        <button type="button" onClick={() => update('status', 'inprogress')}>In progress</button>
        <button type="button" onClick={() => update('status', 'completed')}>Completed</button>
        <button type="button" onClick={() => update('owner', UNASSIGNED)}>No owner</button>
        <button type="button" onClick={() => update('owner', 'mbvalentin')}>Manuel</button>
      </div>
    </div>
  );
}

function TaskTable({items}) {
  return (
    <div className="task-table-wrap">
      <table className="task-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Task</th>
            <th>Status</th>
            <th>Priority</th>
            <th>Type</th>
            <th>Area</th>
            <th>Owner</th>
            <th>Due</th>
            <th>Start</th>
            <th>End</th>
            <th>Links</th>
          </tr>
        </thead>
        <tbody>
          {items.map((task) => (
            <tr key={task.id}>
              <td><code>{task.id}</code></td>
              <td>
                <strong>{task.title}</strong>
                {task.notes && <div className="task-note">{task.notes}</div>}
              </td>
              <td><Badge status={task.status} /></td>
              <td><Badge status={`priority-${task.priority}`} /></td>
              <td><span className="task-chip">{task.type}</span></td>
              <td><span className="task-chip task-chip--muted">{task.area}</span></td>
              <td><Owners owners={task.owners} /></td>
              <td>{displayDate(task.due_date)}</td>
              <td>{displayDate(task.start_date)}</td>
              <td>{displayDate(task.end_date)}</td>
              <td><TaskLinks links={task.links} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TaskLabels({items}) {
  return (
    <div className="task-label-grid">
      {items.map((task) => (
        <article key={task.id} className={`task-label task-label--${normalizeStatus(task.status)}`}>
          <div className="task-label__top">
            <code>{task.id}</code>
            <Badge status={task.status} />
          </div>
          <h3>{task.title}</h3>
          {task.notes && <p>{task.notes}</p>}
          <div className="task-label__meta">
            <Badge status={`priority-${task.priority}`} />
            <span className="task-chip">{task.type}</span>
            <span className="task-chip task-chip--muted">{task.area}</span>
          </div>
          <TaskDates task={task} />
          <div className="task-label__owners">
            <Owners owners={task.owners} />
          </div>
          <div className="task-label__links">
            <TaskLinks links={task.links} />
          </div>
        </article>
      ))}
    </div>
  );
}

export default function TaskBoard({view = 'table', limit, interactive = false, ...staticFilters}) {
  const options = useMemo(() => ({
    statuses: unique(tasks.map((task) => normalizeStatus(task.status))),
    owners: [UNASSIGNED, ...unique(tasks.flatMap((task) => task.owners || []))],
    types: unique(tasks.map((task) => task.type)),
    areas: unique(tasks.map((task) => task.area)),
    priorities: unique(tasks.map((task) => task.priority)),
  }), []);

  const [uiFilters, setUiFilters] = useState({
    status: ALL,
    owner: ALL,
    type: ALL,
    area: ALL,
    priority: ALL,
    query: '',
  });

  const activeFilters = interactive
    ? {
        ...staticFilters,
        status: uiFilters.status === ALL ? staticFilters.status : uiFilters.status,
        owner: uiFilters.owner === ALL ? staticFilters.owner : uiFilters.owner,
        type: uiFilters.type === ALL ? staticFilters.type : uiFilters.type,
        area: uiFilters.area === ALL ? staticFilters.area : uiFilters.area,
        priority: uiFilters.priority === ALL ? staticFilters.priority : uiFilters.priority,
        query: uiFilters.query,
      }
    : staticFilters;

  const filtered = tasks
    .filter((task) => matches(task, activeFilters))
    .sort(sortTasks)
    .slice(0, limit || undefined);

  const board = filtered.length ? (
    view === 'labels' || view === 'cards'
      ? <TaskLabels items={filtered} />
      : <TaskTable items={filtered} />
  ) : (
    <p className="task-empty">No tasks match this filter.</p>
  );

  if (!interactive) {
    return board;
  }

  return (
    <div className="task-board">
      <TaskFilters
        filters={uiFilters}
        setFilters={setUiFilters}
        options={options}
        counts={{visible: filtered.length, total: tasks.length}}
      />
      {board}
    </div>
  );
}
