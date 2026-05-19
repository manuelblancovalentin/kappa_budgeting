import React, {useMemo, useState} from 'react';
import meetings from '@site/src/data/meetings';
import Person from '@site/src/components/Person';
import {Badge} from '@site/src/components/StatusBadges';

const MONTH_FORMAT = new Intl.DateTimeFormat('en-US', {month: 'long', year: 'numeric'});
const DAY_FORMAT = new Intl.DateTimeFormat('en-US', {weekday: 'short'});
const ISO_DAY_MS = 24 * 60 * 60 * 1000;

function parseDate(value) {
  const [year, month, day] = String(value).split('-').map(Number);
  return new Date(Date.UTC(year, month - 1, day));
}

function monthKeyFromDate(date) {
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}`;
}

function monthLabel(monthKey) {
  const [year, month] = monthKey.split('-').map(Number);
  return MONTH_FORMAT.format(new Date(Date.UTC(year, month - 1, 1)));
}

function shiftMonth(monthKey, offset) {
  const [year, month] = monthKey.split('-').map(Number);
  return monthKeyFromDate(new Date(Date.UTC(year, month - 1 + offset, 1)));
}

function unique(values) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

function statusOrder(status) {
  return {planned: 0, running: 1, completed: 2, canceled: 3}[status] ?? 9;
}

function compareMeetings(a, b) {
  const dateDelta = String(a.date).localeCompare(String(b.date));
  if (dateDelta) return dateDelta;
  const statusDelta = statusOrder(a.status) - statusOrder(b.status);
  if (statusDelta) return statusDelta;
  return String(a.id).localeCompare(String(b.id));
}

function meetingsByDate(items) {
  return items.reduce((acc, meeting) => {
    if (!acc[meeting.date]) acc[meeting.date] = [];
    acc[meeting.date].push(meeting);
    return acc;
  }, {});
}

function buildCalendarDays(monthKey) {
  const [year, month] = monthKey.split('-').map(Number);
  const first = new Date(Date.UTC(year, month - 1, 1));
  const last = new Date(Date.UTC(year, month, 0));
  const startOffset = first.getUTCDay();
  const totalCells = Math.ceil((startOffset + last.getUTCDate()) / 7) * 7;
  const start = new Date(first.getTime() - startOffset * ISO_DAY_MS);

  return Array.from({length: totalCells}, (_, index) => {
    const date = new Date(start.getTime() + index * ISO_DAY_MS);
    const iso = date.toISOString().slice(0, 10);
    return {
      iso,
      day: date.getUTCDate(),
      inMonth: monthKeyFromDate(date) === monthKey,
    };
  });
}

function MeetingLinks({links}) {
  const groups = Object.entries(links || {}).filter(([, values]) => values?.length);
  if (!groups.length) return null;

  return (
    <span className="meeting-links">
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

function Attendees({attendees = []}) {
  if (!attendees.length) return <span className="meeting-empty">No attendees recorded</span>;
  return (
    <span className="meeting-attendees">
      {attendees.map((person) => (
        <Person key={person} id={person} />
      ))}
    </span>
  );
}

function MeetingCard({meeting}) {
  return (
    <article className={`meeting-card meeting-card--${meeting.status}`}>
      <div className="meeting-card__top">
        <code>{meeting.id}</code>
        <Badge status={meeting.status} />
      </div>
      <h3>{meeting.title}</h3>
      <div className="meeting-card__meta">
        <span>{meeting.date}</span>
        <span>{meeting.type}</span>
      </div>
      {meeting.summary && <p>{meeting.summary}</p>}
      <Attendees attendees={meeting.attendees} />
      <MeetingLinks links={meeting.links} />
    </article>
  );
}

function CalendarCell({day, dayMeetings}) {
  return (
    <div className={`meeting-calendar__day ${day.inMonth ? '' : 'meeting-calendar__day--muted'}`}>
      <div className="meeting-calendar__date">{day.day}</div>
      <div className="meeting-calendar__items">
        {dayMeetings.map((meeting) => (
          <a
            key={meeting.id}
            href={`#${meeting.id.toLowerCase()}`}
            className={`meeting-calendar__event meeting-calendar__event--${meeting.status}`}
            title={meeting.title}
          >
            {meeting.title}
          </a>
        ))}
      </div>
    </div>
  );
}

export default function MeetingCalendar({month, status, limit}) {
  const options = useMemo(() => {
    const monthKeys = unique(meetings.map((meeting) => monthKeyFromDate(parseDate(meeting.date))));
    const todayKey = monthKeyFromDate(new Date());
    const initial = month || (monthKeys.includes(todayKey) ? todayKey : monthKeys[0] || todayKey);
    return {monthKeys, initial};
  }, [month]);

  const [activeMonth, setActiveMonth] = useState(options.initial);
  const filtered = meetings
    .filter((meeting) => !status || meeting.status === status)
    .sort(compareMeetings)
    .slice(0, limit || undefined);
  const byDate = meetingsByDate(filtered);
  const days = buildCalendarDays(activeMonth);
  const visibleMeetings = filtered.filter((meeting) => monthKeyFromDate(parseDate(meeting.date)) === activeMonth);
  const weekdayLabels = Array.from({length: 7}, (_, day) => DAY_FORMAT.format(new Date(Date.UTC(2026, 0, 4 + day))));

  return (
    <div className="meeting-board">
      <div className="meeting-calendar">
        <div className="meeting-calendar__toolbar">
          <button type="button" onClick={() => setActiveMonth(shiftMonth(activeMonth, -1))}>Previous</button>
          <strong>{monthLabel(activeMonth)}</strong>
          <button type="button" onClick={() => setActiveMonth(shiftMonth(activeMonth, 1))}>Next</button>
        </div>
        <div className="meeting-calendar__weekdays">
          {weekdayLabels.map((label) => <span key={label}>{label}</span>)}
        </div>
        <div className="meeting-calendar__grid">
          {days.map((day) => (
            <CalendarCell key={day.iso} day={day} dayMeetings={byDate[day.iso] || []} />
          ))}
        </div>
      </div>

      <div className="meeting-list">
        <h2>Meetings in {monthLabel(activeMonth)}</h2>
        {visibleMeetings.length ? visibleMeetings.map((meeting) => (
          <div key={meeting.id} id={meeting.id.toLowerCase()}>
            <MeetingCard meeting={meeting} />
          </div>
        )) : <p className="meeting-empty">No meetings recorded for this month.</p>}
      </div>
    </div>
  );
}
