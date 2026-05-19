---
title: "Tasks"
sidebar_label: "✅ Tasks"
status:
  - preliminary
  - inprogress
tags:
  - status
  - task-log
last_modified: 2026-05-18
author: mbvalentin
---
# ✅ Tasks
<PageMeta />
---

<TBox type="summary" title="Project-wide task log">

This page tracks project work that may or may not be connected to a specific experiment. Tasks can describe code, documentation, operations, slides, meetings, server/device setup, or research planning.

</TBox>

The task log is generated from the shared task registry in `site/src/data/tasks.js`. Experiment links are optional.

<TBox type="summary" title="Task lifecycle convention">

A task with status `todo` is unassigned. Once someone starts working on it, set `status: 'inprogress'`, add that person to `owners`, and fill `start_date`. Once finished, set `status: 'completed'` and fill `end_date`. `due_date` may be left empty, which renders as `UNDEFINED`.

</TBox>

## Active Tasks

<TaskBoard interactive view="labels" />

## Static Views

### Unassigned Backlog

<TaskBoard status="todo" view="labels" />

### In Progress

<TaskBoard status="inprogress" view="labels" />

### Blocked

<TaskBoard status="blocked" view="labels" />

### Completed Tasks

<TaskBoard status="completed" />

## Useful Filters

```mdx
<TaskBoard status="todo" />
<TaskBoard owner="__unassigned__" />
<TaskBoard owner="mbvalentin" />
<TaskBoard type="ops" />
<TaskBoard area="controllers" />
<TaskBoard statuses={["todo", "inprogress"]} view="labels" />
```
