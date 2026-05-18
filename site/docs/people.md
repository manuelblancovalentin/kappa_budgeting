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

This page is the project roster. Use the IDs below in page frontmatter and inline mentions.

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


## Adding New People
Just add a new entry to `src/data/people.js` with the following format:

```javascript
'entryname': {
    id: 'unique-id',
    handle: 'username',
    name: 'Full Name',
    shortName: 'Short Name',
    role: 'Role in Project',
    github: 'https://github.com/username',
  },
```