const people = {
  mbvalentin: {
    id: 'mbvalentin',
    handle: 'manuelblancovalentin',
    name: 'Manu B. Valentin',
    shortName: 'Manu',
    role: 'Project lead',
    github: 'https://github.com/manuelblancovalentin',
  },
  'alan-guo': {
    id: 'alan-guo',
    handle: 'hoeseng',
    name: 'Alan Guo',
    shortName: 'Alan',
    role: 'Collaborator',
    github: 'https://github.com/HoesenG',
  },
  'ryan-forelli': {
    id: 'ryan-forelli',
    handle: 'rfforelli',
    name: 'Ryan Forelli',
    shortName: 'Ryan',
    role: 'Collaborator',
    github: 'https://github.com/rfforelli',
  },
  'ethan-gindlesperger': {
    id: 'ethan-gindlesperger',
    handle: 'egindy',
    name: 'Ethan Gindlesperger',
    shortName: 'Gindy',
    role: 'Collaborator',
    github: 'https://github.com/egindy',
  },
  'seda-ogrenci': {
    id: 'seda-ogrenci',
    handle: 'sedaogrenci',
    name: 'Seda Ogrenci',
    shortName: 'Seda',
    role: 'PI / advisor',
    github: 'https://github.com/sedaogrenci',
  },
};

export const lab = {
  name: 'Memik Lab',
  github: 'https://github.com/orgs/Memik-Lab',
  people: 'https://github.com/orgs/Memik-Lab/people/sedaogrenci',
};

export function getPerson(id) {
  if (!id) {
    return null;
  }

  const key = String(id).replace(/^@/, '');
  return people[key] || Object.values(people).find((person) => person.handle === key) || null;
}

export function getPeople(ids) {
  const list = Array.isArray(ids) ? ids : [ids];
  return list.map(getPerson).filter(Boolean);
}

export function listPeople() {
  return Object.values(people);
}

export default people;
