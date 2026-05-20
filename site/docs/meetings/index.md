---
title: "Meetings"
sidebar_label: "Meetings"
status:
  - preliminary
tags:
  - meeting
  - project-log
last_modified: 2026-05-20
author: mbvalentin
---
# 📅 Meetings
<PageMeta />
---

<TBox type="summary" title="Meeting registry">

This page is the project meeting calendar. Meetings are stored in `site/src/data/meetings.js` and rendered here. Each entry can record a date, status, meeting type, attendees, links, a short markdown notes preview, and a pointer to a full markdown note page. Attendees should use registered people IDs so the chips link back to the project roster.

</TBox>

<MeetingCalendar />

## Registry Format

A meeting entry is intentionally small:

<Algorithm
  title="Meeting record shape"
  content={`meeting:
  id: MTG-YYYY-MM-DD-SLUG
  title: short readable title
  date: YYYY-MM-DD
  status: planned | completed | canceled
  type: working-session | research-planning | project-update | handoff
  attendees: [mbvalentin, alan-guo]
  summary: one short paragraph
  notes: markdown preview shown in the meeting card
  notesDoc:
    label: 'Click here for full meeting notes'
    href: /docs/meetings/YYYY-MM-DD-meeting-slug
  links:
    docs:
      - label: Related page
        href: /docs/...
`}
/>

<TBox type="note" title="Source of truth">

The calendar registry stays as the structured index. Longer notes should live as dedicated markdown pages under `site/docs/meetings/`, with the registry entry linking to them through `notesDoc`.

</TBox>
