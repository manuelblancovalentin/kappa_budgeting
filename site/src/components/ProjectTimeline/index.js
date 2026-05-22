import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import tasks, {taskStages} from '@site/src/data/tasks';
import {getPerson} from '@site/src/data/people';

const DEFAULT_WINDOW_BEFORE_DAYS = 3;
const DEFAULT_WINDOW_AFTER_DAYS = 3;
const DEFAULT_VISIBLE_TAGS = 3;
const DAY_MS = 24 * 60 * 60 * 1000;

function normalizeStatus(status) {
  return String(status || 'planned').toLowerCase();
}

function statusBucket(status) {
  const value = normalizeStatus(status);
  if (['done', 'complete', 'completed', 'valid'].includes(value)) return 'completed';
  if (['active', 'running', 'inprogress', 'in-progress', 'blocked', 'failing'].includes(value)) return 'inprogress';
  return 'planned';
}

function taskStage(task) {
  return task.stage || 'publication';
}

function taskTags(task) {
  return task.summaryTags || task.tags || [];
}

function ownerLabels(task) {
  const owners = task.owners || [];
  if (!owners.length) return ['Unassigned'];
  return owners.map((owner) => getPerson(owner)?.shortName || getPerson(owner)?.name || owner);
}

function parseDate(value) {
  if (!value) return null;
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

function isoDate(date) {
  return date.toISOString().slice(0, 10);
}

function addDays(date, days) {
  return new Date(date.getTime() + days * DAY_MS);
}

function startOfUtcDay(date) {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
}

function endOfMonth(year, monthIndex) {
  return new Date(Date.UTC(year, monthIndex + 1, 0));
}

function startOfMonth(date) {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), 1));
}

function monthNumber(label) {
  const date = new Date(`${label} 1, 2000 00:00:00 UTC`);
  return Number.isNaN(date.getTime()) ? null : date.getUTCMonth();
}

function monthLabels(start, end) {
  const labels = [];
  let cursor = startOfMonth(start);
  while (cursor <= end) {
    const monthStart = new Date(cursor);
    const monthEnd = endOfMonth(cursor.getUTCFullYear(), cursor.getUTCMonth());
    labels.push({
      label: monthStart.toLocaleDateString('en-US', {month: 'short', year: 'numeric', timeZone: 'UTC'}),
      start: monthStart,
      end: monthEnd,
    });
    cursor = new Date(Date.UTC(cursor.getUTCFullYear(), cursor.getUTCMonth() + 1, 1));
  }
  return labels;
}

function visibleRangeFromMonths(months, today) {
  if (!months?.length) return null;
  let activeYear = today.getUTCFullYear();
  const parsed = months.map((label) => {
    const parts = String(label).trim().split(/\s+/);
    const month = monthNumber(parts[0]);
    const explicitYear = parts.find((part) => /^\d{4}$/.test(part));
    if (explicitYear) activeYear = Number(explicitYear);
    return {label, month, year: activeYear};
  }).filter((item) => item.month !== null);

  if (!parsed.length) return null;

  return {
    labels: parsed.map((item) => ({
      label: item.label,
      start: new Date(Date.UTC(item.year, item.month, 1)),
      end: endOfMonth(item.year, item.month),
    })),
    start: new Date(Date.UTC(parsed[0].year, parsed[0].month, 1)),
    end: endOfMonth(parsed[parsed.length - 1].year, parsed[parsed.length - 1].month),
  };
}

function allTaskDates(items) {
  return items
    .flatMap((task) => [task.start_date, task.end_date, task.created, task.due_date])
    .map(parseDate)
    .filter(Boolean);
}

function inferredDateWindow(items, today) {
  const dates = allTaskDates(items);
  const earliest = dates.length ? new Date(Math.min(...dates.map((date) => date.getTime()))) : today;
  const latest = dates.length ? new Date(Math.max(...dates.map((date) => date.getTime()))) : today;
  return {
    start: addDays(earliest, -DEFAULT_WINDOW_BEFORE_DAYS),
    end: addDays(new Date(Math.max(latest.getTime(), today.getTime())), DEFAULT_WINDOW_AFTER_DAYS),
  };
}

function visibleRange({items, months, startDate, endDate, today}) {
  const explicitStart = parseDate(startDate);
  const explicitEnd = parseDate(endDate);
  const monthRange = visibleRangeFromMonths(months, today);

  if (explicitStart || explicitEnd) {
    const taskDates = allTaskDates(items);
    const inferredStart = taskDates.length ? new Date(Math.min(...taskDates.map((date) => date.getTime()))) : today;
    const inferredEnd = taskDates.length ? new Date(Math.max(...taskDates.map((date) => date.getTime()))) : today;
    const start = explicitStart || addDays(inferredStart, -DEFAULT_WINDOW_BEFORE_DAYS);
    const end = explicitEnd || addDays(new Date(Math.max(inferredEnd.getTime(), today.getTime())), DEFAULT_WINDOW_AFTER_DAYS);
    return {start, end, labels: monthLabels(start, end)};
  }

  if (monthRange) return monthRange;

  const {start, end} = inferredDateWindow(items, today);
  return {start, end, labels: monthLabels(start, end)};
}

function clampDate(date, start, end) {
  if (date < start) return start;
  if (date > end) return end;
  return date;
}

function percentForDate(date, range) {
  const total = Math.max(range.end.getTime() - range.start.getTime(), DAY_MS);
  return (100 * (date.getTime() - range.start.getTime())) / total;
}

function normalizedBarEnd(start, end) {
  return end.getTime() <= start.getTime() ? addDays(start, 0.35) : end;
}

function barStyle(start, end, range) {
  const left = percentForDate(clampDate(start, range.start, range.end), range);
  const right = percentForDate(clampDate(normalizedBarEnd(start, end), range.start, range.end), range);
  const width = Math.max(right - left, 0.8);
  return {
    left: `${Math.max(0, Math.min(left, 100))}%`,
    width: `${width}%`,
    '--bar-width-pct': width,
  };
}

function barWidthPct(start, end, range) {
  const left = percentForDate(clampDate(start, range.start, range.end), range);
  const right = percentForDate(clampDate(normalizedBarEnd(start, end), range.start, range.end), range);
  return Math.max(right - left, 0.8);
}

function itemDates(item, range, today) {
  const status = statusBucket(item.status);
  const start = parseDate(item.start_date) || parseDate(item.created) || (status === 'planned' ? today : range.start);

  if (status === 'completed') {
    return {
      start,
      end: parseDate(item.end_date) || parseDate(item.due_date) || start,
    };
  }

  if (status === 'inprogress') {
    return {
      start,
      end: today,
    };
  }

  return {
    start: today,
    end: parseDate(item.due_date) || range.end,
  };
}

function stageStatus(items) {
  if (!items.length) return 'planned';
  if (items.some((task) => normalizeStatus(task.status) === 'blocked')) return 'blocked';
  if (items.every((task) => ['completed', 'done'].includes(normalizeStatus(task.status)))) return 'completed';
  if (items.some((task) => normalizeStatus(task.status) === 'inprogress')) return 'inprogress';
  if (items.some((task) => normalizeStatus(task.status) === 'completed')) return 'inprogress';
  return 'planned';
}

function mostFrequentTags(items, limit = 6) {
  const counts = new Map();
  for (const task of items) {
    for (const tag of taskTags(task)) {
      counts.set(tag, (counts.get(tag) || 0) + 1);
    }
  }
  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([tag]) => tag);
}

function deriveStages(items, range, today) {
  const timelineTasks = items.filter((task) => task.timeline !== false);
  const byStage = new Map();
  for (const task of timelineTasks) {
    const stage = taskStage(task);
    if (!byStage.has(stage)) byStage.set(stage, []);
    byStage.get(stage).push(task);
  }

  return Object.entries(taskStages)
    .sort(([, a], [, b]) => (a.order ?? 99) - (b.order ?? 99))
    .map(([id, meta]) => {
      const stageTasks = byStage.get(id) || [];
      const status = stageStatus(stageTasks);
      const bucket = statusBucket(status);
      const taskRanges = stageTasks.map((task) => itemDates(task, range, today));
      const completedCount = stageTasks.filter((task) => statusBucket(task.status) === 'completed').length;

      let start = taskRanges.length
        ? new Date(Math.min(...taskRanges.map((item) => item.start.getTime())))
        : today;
      let end = taskRanges.length
        ? new Date(Math.max(...taskRanges.map((item) => item.end.getTime())))
        : range.end;

      if (bucket === 'inprogress') end = today;
      if (bucket === 'planned') {
        start = today;
        end = range.end;
      }

      return {
        id,
        title: meta.label || id,
        status,
        start,
        end,
        details: mostFrequentTags(stageTasks),
        tasks: stageTasks,
        taskCount: stageTasks.length,
        completedCount,
      };
    });
}

function totalProgress(stages) {
  if (!stages.length) return 0;
  const completed = stages.reduce((acc, stage) => acc + (stage.completedCount || 0), 0);
  const total = stages.reduce((acc, stage) => acc + (stage.taskCount || 0), 0);
  return total ? Math.round((100 * completed) / total) : 0;
}

function summarySegments(stages) {
  const total = Math.max(stages.length, 1);
  return ['completed', 'inprogress', 'planned'].map((bucket) => {
    const count = stages.filter((stage) => statusBucket(stage.status) === bucket).length;
    return {
      bucket,
      count,
      percent: Math.round((100 * count) / total),
    };
  }).filter((segment) => segment.count > 0);
}

function TimelineAxis({range}) {
  return (
    <div className="project-timeline__axis">
      <span />
      <div className="project-timeline__axis-track">
        {range.labels.map((month) => (
          <span
            key={month.label}
            style={barStyle(month.start, month.end, range)}
          >
            {month.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function TodayMarker({range, today, className}) {
  if (today < range.start || today > range.end) return null;
  return (
    <span
      className={clsx('project-timeline__today', className)}
      style={{'--today-left-pct': percentForDate(today, range) / 100}}
      title={`Today: ${isoDate(today)}`}
    />
  );
}

function TimelineBar({item, range, className, children, onClick}) {
  const status = statusBucket(item.status);
  const Element = onClick ? 'button' : 'div';
  const width = barWidthPct(item.start, item.end, range);

  return (
    <Element
      type={onClick ? 'button' : undefined}
      className={clsx(
        'project-timeline__gantt-bar',
        `project-timeline__gantt-bar--${status}`,
        width < 4 && 'project-timeline__gantt-bar--pin',
        width < 12 && 'project-timeline__gantt-bar--compact',
        width < 24 && 'project-timeline__gantt-bar--short',
        className,
      )}
      style={barStyle(item.start, item.end, range)}
      title={`${item.title}: ${isoDate(item.start)} to ${isoDate(item.end)}`}
      onClick={onClick}
    >
      <span className="project-timeline__bar-content">
        {children}
      </span>
      {onClick && (
        <span className="project-timeline__bar-action" aria-hidden="true">
          ↵
        </span>
      )}
    </Element>
  );
}

function TimelineControls({range, today, startValue, endValue, onRangeChange, onPreset}) {
  return (
    <div className="project-timeline__controls">
      <div className="project-timeline__preset-group" aria-label="Timeline range presets">
        <button type="button" onClick={() => onPreset('now')}>Now</button>
        <button type="button" onClick={() => onPreset('30d')}>30d</button>
        <button type="button" onClick={() => onPreset('90d')}>90d</button>
        <button type="button" onClick={() => onPreset('all')}>All</button>
      </div>
      <div className="project-timeline__date-fields">
        <label>
          <span>Start</span>
          <input
            type="date"
            value={startValue}
            onChange={(event) => onRangeChange(event.target.value, endValue)}
          />
        </label>
        <label>
          <span>End</span>
          <input
            type="date"
            value={endValue}
            onChange={(event) => onRangeChange(startValue, event.target.value)}
          />
        </label>
      </div>
      <span className="project-timeline__range-note">
        {isoDate(range.start)} to {isoDate(range.end)}
      </span>
    </div>
  );
}

function TaskRows({stage, range, today}) {
  return (
    <div className="project-timeline__subrows">
      {stage.tasks.map((task) => {
        const dates = itemDates(task, range, today);
        const item = {...task, title: task.title, start: dates.start, end: dates.end};
        return (
          <div className="project-timeline__subrow" key={task.id}>
            <Link className="project-timeline__subrow-label" to={`/docs/status/tasks?query=${encodeURIComponent(task.id)}`}>
              <code>{task.id}</code>
              <span>{task.title}</span>
            </Link>
            <div className="project-timeline__track">
              <TimelineBar item={item} range={range}>
                {ownerLabels(task).map((owner) => (
                  <span key={owner}>{owner}</span>
                ))}
                {taskTags(task).slice(0, 2).map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </TimelineBar>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function ProjectTimeline({
  title = 'Project Roadmap',
  months,
  startDate,
  endDate,
  stages,
  defaultOpen = false,
}) {
  const [openStages, setOpenStages] = useState(() => new Set());
  const rawToday = startOfUtcDay(new Date());
  const initialRange = visibleRange({items: tasks, months, startDate, endDate, today: rawToday});
  const [rangeInput, setRangeInput] = useState(() => ({
    startDate: isoDate(initialRange.start),
    endDate: isoDate(initialRange.end),
  }));
  const range = visibleRange({
    items: tasks,
    startDate: rangeInput.startDate,
    endDate: rangeInput.endDate,
    today: rawToday,
  });
  const today = clampDate(rawToday, range.start, range.end);
  const resolvedStages = stages || deriveStages(tasks, range, today);
  const progress = totalProgress(resolvedStages);
  const segments = summarySegments(resolvedStages);

  const toggleStage = (id) => {
    setOpenStages((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const setRange = (nextStart, nextEnd) => {
    const parsedStart = parseDate(nextStart);
    const parsedEnd = parseDate(nextEnd);
    if (parsedStart && parsedEnd && parsedStart > parsedEnd) {
      setRangeInput({startDate: nextEnd, endDate: nextStart});
      return;
    }
    setRangeInput({startDate: nextStart, endDate: nextEnd});
  };

  const setPreset = (preset) => {
    if (preset === 'now') {
      const defaultRange = inferredDateWindow(tasks, rawToday);
      setRangeInput({
        startDate: isoDate(defaultRange.start),
        endDate: isoDate(defaultRange.end),
      });
      return;
    }
    if (preset === '30d') {
      setRangeInput({
        startDate: isoDate(addDays(rawToday, -7)),
        endDate: isoDate(addDays(rawToday, 30)),
      });
      return;
    }
    if (preset === '90d') {
      setRangeInput({
        startDate: isoDate(addDays(rawToday, -14)),
        endDate: isoDate(addDays(rawToday, 90)),
      });
      return;
    }
    const fullRange = inferredDateWindow(tasks, rawToday);
    setRangeInput({
      startDate: isoDate(fullRange.start),
      endDate: isoDate(fullRange.end),
    });
  };

  return (
    <details className="project-timeline" open={defaultOpen}>
      <summary className="project-timeline__summary">
        <div className="project-timeline__summary-head">
          <span className="project-timeline__title">{title}</span>
          <span className="project-timeline__percent">{progress}% task completion</span>
        </div>
        <div className="project-timeline__bar" aria-label={`${progress}% complete`}>
          {segments.map((segment) => (
            <span
              key={segment.bucket}
              className={clsx('project-timeline__segment', `project-timeline__segment--${segment.bucket}`)}
              style={{flexGrow: segment.count}}
              title={`${segment.bucket}: ${segment.count} stages`}
            >
              {segment.percent}%
            </span>
          ))}
        </div>
      </summary>

      <div className="project-timeline__expanded">
        <TimelineControls
          range={range}
          today={rawToday}
          startValue={rangeInput.startDate}
          endValue={rangeInput.endDate}
          onRangeChange={setRange}
          onPreset={setPreset}
        />
        <TimelineAxis range={range} />

        <div className="project-timeline__rows">
          <TodayMarker range={range} today={rawToday} className="project-timeline__today--global" />
          {resolvedStages.map((stage) => {
            const isOpen = openStages.has(stage.id);
            const taskText = stage.taskCount ? `${stage.completedCount}/${stage.taskCount} tasks` : 'No tasks';
            const visibleDetails = stage.details.slice(0, DEFAULT_VISIBLE_TAGS);
            const hiddenDetails = Math.max(stage.details.length - visibleDetails.length, 0);

            return (
              <div className={clsx('project-timeline__row-group', isOpen && 'project-timeline__row-group--open')} key={stage.id}>
                <div className="project-timeline__row">
                  <Link className="project-timeline__row-label" to={`/docs/status/tasks?stage=${stage.id}`}>
                    <span>{stage.title}</span>
                    <small>{taskText}</small>
                  </Link>
                  <div className="project-timeline__track">
                    <TimelineBar item={stage} range={range} onClick={() => toggleStage(stage.id)}>
                      {visibleDetails.map((detail) => (
                        <span key={detail}>{detail}</span>
                      ))}
                      {hiddenDetails > 0 && <span>+{hiddenDetails}</span>}
                    </TimelineBar>
                  </div>
                </div>
                {isOpen && <TaskRows stage={stage} range={range} today={today} />}
              </div>
            );
          })}
        </div>
      </div>
    </details>
  );
}
