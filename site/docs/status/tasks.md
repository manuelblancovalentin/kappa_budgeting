---
title: "Tasks"
sidebar_label: "✅ Tasks"
status:
  - preliminary
  - inprogress
tags:
  - status
  - task-log
last_modified: 2026-05-21
author: mbvalentin
---
# ✅ Tasks
<PageMeta />
---

<TBox type="summary" title="Project-wide task log">

This page tracks project work that may or may not be connected to a specific experiment. Tasks can describe code, documentation, operations, slides, meetings, server/device setup, or research planning.

</TBox>

The task log is generated from the shared task registry in `site/src/data/tasks.js`. The same registry now drives the project roadmap on the overview page.

<TBox type="summary" title="Task lifecycle convention">

A task with status `todo` is unassigned. Once someone starts working on it, set `status: 'inprogress'`, add that person to `owners`, and fill `start_date`. Once finished, set `status: 'completed'` and fill `end_date`. `due_date` may be left empty, which renders as `UNDEFINED`.

Each task is classified by three axes:

- `stage`: lifecycle position, such as research, implementation, validation, integration, deployment, or publication.
- `target`: the artifact or system being changed, such as `enabol`, `software`, `hls4ml`, `cpp_backend`, `csim`, `fpga`, or `paper`.
- `action`: the kind of work, such as design, development, documentation, experiment, testing, integration, analysis, or writing.

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
<TaskBoard stage="validation" />
<TaskBoard target="enabol" />
<TaskBoard action="development" />
<TaskBoard tag="global-throttle" />
<TaskBoard statuses={["todo", "inprogress"]} view="labels" />
```
