---
title: "Meetings"
sidebar_label: "Meetings"
status:
  - preliminary
tags:
  - meeting
  - project-log
last_modified: 2026-05-18
author: mbvalentin
---
# 📅 Meetings
<PageMeta />
---

<TBox type="summary" title="Meeting registry">

This page is the project meeting calendar. For now, meetings are stored in `site/src/data/meetings.js` and rendered here. Each entry can record a date, status, meeting type, attendees, links, and short notes. Attendees should use registered people IDs so the chips link back to the project roster.

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
  links:
    docs:
      - label: Related page
        href: /docs/...
`}
/>

<TBox type="todo" title="Future upgrade">

If meeting notes become longer than a short summary, we can keep this calendar registry but point each entry to a dedicated markdown page under `site/docs/meetings/`. The calendar would remain the index, while each meeting page would hold agenda, action items, and raw notes.

</TBox>
