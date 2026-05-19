---
title: "🦖 Docusaurus contribution guide"
status:
  - draft
tags:
  - todo
  - placeholder
  - onboarding
last_modified: 2026-05-15
---
<PageMeta />
---

## [👥 People](../people.md)

### How to add a new person to the registry
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



---

## [💬 Blog posts](/blog)

### How to write a blog post
Go to `site/blog` and create a new markdown file with the following front matter:

```yaml
---
slug: project-kickoff
title: Project kickoff
authors: mbvalentin
tags: [kickoff, planning]
---
```

Then write any content below in normal markdown. You can use any of the custom components available in the docs, such as `<TBox>`, `<Person>`, `<ProjectTimeline>`, etc. The name of the file doesn't matter too much, but for good practice we could keep it in the format: `YYYY-MM-DD-title.md`, so that the posts are ordered by date and have a nice slug. We can add the author at the end if we want too. 

Note that if there are many authors, you can simply just do:
```yaml
authors: [mbvalentin, johndoe]
```

### How to add a new author to the blog
Go to `site/blog` and you should find a file called `authors.yml`. If it doesn't exist, create it. Then add a new entry for the author, like so (example dummy name):

```yaml
johndoe:
  name: John Doe
  title: PhD Candidate at NU
  url: https://github.com/johndoe
  #image_url: https://avatars.githubusercontent.com/u/10078326?v=4
  page: true
  socials:
    github: johndoe
    linkedin: johndoe
    twitter: johndoe
```

You can skip fields by commenting them. The `page: true` field means that the author will have a dedicated page with their profile and all their posts. If you set it to false, the author will still be credited in their posts but won't have a profile page. 

Now, in any blog post just use the key name (`johndoe` in the example) in the `authors` field of the front matter. 

---

## [🏷️ Docs](./index.md)
### How to add a new doc page
Go to `site/docs` and create a new markdown file with the following front matter:
```yaml
---
title: "Page Title"
tags:
  - tag1
  - tag2
last_modified: 2026-05-15
author: john-doe
---
```

Now you need to add a link to this page in the sidebar. Go to `sidebars.js` and find the appropriate section where you want to add the link. Then add a new entry with the following format:

```javascript
{
  type: 'doc',
  docId: 'path/to/your/doc', // this should match the file path of your doc without the .md extension
  label: 'Page Title', // this is the name that will appear in the sidebar
},
```

In case you need multiple pages under the same section, it's better to create a folder for that section and put all related markdown files inside (e.g., see `site/docs/ablations`). In this case, you can add a category in the sidebar like this:

```javascript
{
  type: 'category',
  label: 'Section Name',
  items: [
    {
      type: 'doc',
      docId: 'path/to/your/doc1',
      label: 'Page Title 1',
    },
    {
      type: 'doc',
      docId: 'path/to/your/doc2',
      label: 'Page Title 2',
    },
  ],
},
```


---

## [✅ Tasks](../status/tasks.md)
Tasks are a bit trickier, but don't be scared. This is because there are some react (js) components involved to make the tables and interactivity work. But it's not that bad once you understand the structure.

### How to add a new task
First, go to `site/src/data/tasks.js` and add a new entry to the `tasks` object with the following format:

```javascript
{
  id: 'unique-task-id',
  title: 'Task Title',
  description: 'Task description goes here.',
  status: 'todo', // can be 'todo', 'inprogress', 'completed', or 'blocked'
  type: 'research', // can be 'research', 'code', 'ops', etc.
  area: 'controllers', // can be 'controllers', 'formulation', etc.
  owners: ['mbvalentin'], // array of person ids from the people registry
  start_date: '2026-05-15',
  due_date: '2026-06-01',
  end_date: null, // set this when the task is completed
},
```

Then, you can view this task in the [Tasks page](../status/tasks.md) and filter it by status, type, area, or owner. You can also update the status and other fields as the task progresses.

Some generic notes on tasks so we keep this sane (please):
* Each task should ideally be small and actionable, something that can be done in a few hours to a couple of days. If a task is too big, consider breaking it down into smaller subtasks.
* Try to keep the task descriptions clear and concise, so that anyone can understand what needs to be done without needing additional context.
* Update the task status and dates as you work on it, so that the task log remains accurate and useful for everyone.
* Once a task is assigned to someone, remember to add the date they started working on it.
* Once a task is completed, update the status to 'completed' and add the end date. This helps us track how long tasks are taking and plan better in the future.
* Changing the ownership of an already completed task is unmoral and ethical 🙂. Don't.

### How does the task table work?
The task table is rendered using a React component that reads the `tasks` data and displays it in a nice format. The table allows filtering by status, type, area, and owner. It also allows updating the status of each task directly from the table. The component is designed to be reusable, so you can use it in other pages if needed. The filtering and interactivity are handled using React state and event handlers. If you want to modify the table or add new features, you can check the source code in `site/src/components/TaskBoard/index.js`, but to be honest **you probably don't need to**. Just adding tasks to the `tasks.js` file and updating their status as you work on them should be enough for most use cases.


---

## Dynamical JS objects in the docs
In some cases, we want to render dynamic/nice content that uses custom CSS. For instance, this is the case of the tags (like `completed`, `active`, `in-progress`, `placeholder`, etc.). I'm going to try to add here an overview of the documentation of the custom components I've added to the docs, just in case you want to add more. 

### `StatusBadges`
This is a simple component that renders a badge with a specific color based on the status of a task or project, e.g. <StatusBadges status="completed" />. The possible statuses are 'todo', 'inprogress', 'completed', and 'blocked', each with its own color. The definition of this component can be found in `site/src/components/StatusBadges/index.js`. 

If you want to add more statuses, there are two things you need to do:
1. Add the new status label in the `javascript` code above,
2. Add the corresponding CSS class in `site/src/css/custom.css` with the desired styling.

For example, say we wanted to add a new status called 'review' with a purple color. We would first expand the `STATUS_LABELS` object in `StatusBadges/index.js` to include the new status:

```javascript
const STATUS_LABELS = {
  ...
  review: 'Review',
};
```

and then we would add a new CSS class in `custom.css` to define the styling for the 'review' status (note that the class name should begin with `.status-badge.` followed by the status name, this is very important):

```css
.status-badge.review {
  border-color: purple;
  background-color: purple;
  color: white;
}
```

---

### Algorithm
You can use in any page a pseudo-algorithm block by using the object `Algorithm` which is defined in `site/src/components/Algorithm/index.js`. This component allows you to write algorithms in a nice format using markdown. For example:

```html
<Algorithm
  id="sample-algorithm"
  title="Sample algorithm"
  caption="This is some description of the algorithm and its inputs/outputs."
  content={
  `input x, y; output z
  for i in range(10):
      z = x + y
  end for
  if z > 10 then
      return z
  else
      return 0
  end if
`}
/>
```

which should render:
<Algorithm
  id="sample-algorithm"
  title="Sample algorithm"
  caption="This is some description of the algorithm and its inputs/outputs."
  content={
  `input x, y; output z
  for i in range(10):
      z = x + y
  end for
  if z > 10 then
      return z
  else
      return 0
  end if
`}
/>

<TBox type="warning" title="Note on algorithm formatting">
 Very important: Note that the content starts and ends with the character `. This is intentional. This means that the content is considered a "raw" string.
</ TBox>

--- 

### Brand Name
This is just the name of ENABOL in a fancy way: <ENABOL />. You can use it in any page by importing the component from `site/src/components/BrandName/index.js` and then using it as `<ENABOL />` whenever you need to display the brand name. It's the html equivalent of what we use in Overleaf when we define `\newcommand{\enabol}{...}`. 

---

### Figure 
This is a simple component that allows you to display figures with captions in a nice format. You can use it by importing the `Figure` component from `site/src/components/Figure/index.js` and then using it as follows:

```html
<Figure
  src="/img/figure1.png"
  alt="Figure description"
  caption="This is the caption of the figure."
/>
```

Even better, you can add a reference to the image, such that later to can add a link in the text pointing to it. For instance: 

```html
<FigureRef target="fig-example-unique-id">Figure 2</FigureRef> shows something.

<Figure
  id="fig-example-unique-id"
  src="/img/figure1.png"
  alt="Figure description"
  caption="This is the caption of the figure."
  maxWidth="80%"
  label="Figure 2"
/>
```


---

### Page Meta
This is just something you should add at the top of every new markdown file you create for documentation. It renders the last modified date, author, and tags of the page in a nice format at the top. You can use it by importing the `PageMeta` component from `site/src/components/PageMeta/index.js` and then using it as `<PageMeta />` at the top of your markdown file, right after the front matter. For example:

```markdown h_lines={9}
---
title: "Page Title"
tags:
  - tag1
  - tag2
last_modified: 2026-05-15
author: john-doe
---
<PageMeta />  
```

---

### TBox
This is a simple component that allows you to create colored boxes with different types (e.g., summary, warning, todo, etc.) to highlight important information in the docs. You can use it by importing the `TBox` component from `site/src/components/TBox/index.js` and then using it as follows:
```html
<TBox type="summary" title="Summary">
This is a summary of the main points of the section.
</TBox>
```

which should render:
<TBox type="summary" title="Summary">
This is a summary of the main points of the section.
</TBox>

Below are all the available types for the `TBox` component, along with their corresponding colors:
<TBox type="info" title="Info">
This is an informational box.
</TBox>

<TBox type="warning" title="Warning">
This is a warning box.
</TBox>

<TBox type="todo" title="TODO">
This is a TODO box.
</TBox>

#### Extending TBox
If you want to add more types of boxes, you can do so by modifying the `TBox` component in `site/src/components/TBox/index.js`. You would need to add a new case in the switch statement that defines the styling for each type, and then add the corresponding CSS class in `site/src/css/custom.css` to define the colors and styles for the new type. For example, if you wanted to add a new type called 'success' with a green color, you would first extend the `DEFAULT_TITLES` object in `TBox/index.js` to include the new type:

```javascript
const DEFAULT_TITLES = {
  ...
  success: 'Success',
};
```
and then you would add a new CSS class in `custom.css` to define the styling for the 'success' type:

```css
.tbox.success {
  border-color: green;
  background-color: #e6ffe6;
  color: green;
}
```

---

### Terminal
This is a simple component that allows you to display terminal-like output in the docs. You can use it by importing the `Terminal` component from `site/src/components/Terminal/index.js` and then using it as follows:

```html
<Terminal
  content={
  `> python train.py --config=config.yaml
  Training started...
  Epoch 1/10: loss=0.5, acc=0.8
  Epoch 2/10: loss=0.4, acc=0.85
  ...
  Training completed!`
}
/>
```

which should render:
<Terminal
  content={
  `> python train.py --config=config.yaml
  Training started...
  Epoch 1/10: loss=0.5, acc=0.8
  Epoch 2/10: loss=0.4, acc=0.85
  ...
  Training completed!`
}
/>

<TBox type="info" title="Note">
Note this terminal is literally just for visualization purposes. Just to show something in a nice way.
</TBox>

---

