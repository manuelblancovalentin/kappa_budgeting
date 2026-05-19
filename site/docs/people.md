---
status:
  - valid
tags:
  - project
  - people
last_modified: 2026-05-18
author: mbvalentin
---

# 👥 People

<PageMeta />
---

This page is the project roster. Use the IDs below in page frontmatter and inline mentions. To know how to contribute to the people registry, read the [Docusaurus contribution guide](./onboarding/docusaurus.md#-people).

<PeopleTable />

## Usage

For page authorship use the `id` in `src/data/people.js`:

```yaml
author: mbvalentin
```

For multiple authors:

```yaml
authors:
  - mbvalentin
  - alan-guo
```

For inline mentions:

```mdx
<Person id="alan-guo" />
```
