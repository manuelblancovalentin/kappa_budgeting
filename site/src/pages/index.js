import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const cards = [
  {
    title: '📝 Current Status',
    description:
      'The canonical snapshot of what works, what is blocked, and what should happen next.',
    link: '/docs/current-status',
    cta: 'Read status',
  },
  {
    title: '📓 Lab Log',
    description:
      'Chronological progress notes: experiments, debugging sessions, weekly updates, and failed attempts.',
    link: '/blog',
    cta: 'Open log',
  },
  {
    title: '🚀 Onboarding',
    description:
      'Setup instructions, repository map, common commands, and first tasks for new contributors.',
    link: '/docs/onboarding/setup',
    cta: 'Start here',
  },
  {
    title: '📚 Datasets',
    description:
      'Descriptions of datasets used in this project, including input distribution, target function, and drift support.',
    link: '/docs/datasets',
    cta: 'View datasets',
  },
  {
    title: '📦 Models',
    description:
      'Descriptions of models used in this project, including architecture, training procedure, and performance.',
    link: '/docs/models',
    cta: 'View models',
  },
  {
    title: '🔬 Experiments',
    description:
      'Experiment summaries, protocols, metrics, plots, and links to raw logs or notebooks.',
    link: '/docs/experiments',
    cta: 'View experiments',
  },
];

function Card({title, description, link, cta}) {
  return (
    <div className={styles.card}>
      <h3>{title}</h3>
      <p>{description}</p>
      <Link className="button button--primary button--sm" to={link}>
        {cta}
      </Link>
    </div>
  );
}

export default function Home() {
  return (
    <Layout
      title="Project Logbook"
      description="Living project documentation and lab log">
      <main>
        <section className={styles.hero}>
          <div className="container">
            <div className={styles.heroLayout}>
              <div className={styles.heroLogoWrap} aria-hidden="true">
                <img
                  className={styles.heroLogo}
                  src="/kappa_budgeting/img/logo.svg"
                  alt=""
                  loading="eager"
                />
              </div>
              <div className={styles.heroContent}>
                <Heading as="h1" className={styles.heroTitle}>
                  Logbook
                </Heading>
                <p className={styles.heroSubtitle}>
                  A living memory system for this project: current status, lab logs,
                  experiments, onboarding, and handoff notes.
                </p>
                <div className={styles.heroButtons}>
                  <Link className="button button--primary button--lg" to="/docs/current-status">
                    Current Status
                  </Link>
                  <Link className="button button--secondary button--lg" to="/blog">
                    Latest Lab Log
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="container">
          <div className={styles.statusBox}>
            <h2>Project snapshot</h2>
            <p>
              Use this section to summarize the project in one paragraph. Keep it
              brutally useful: what the project is, who owns it, what currently
              works, what is blocked, and what someone should read first.
            </p>
            <ul>
              <li><strong>Owner:</strong> NAME</li>
              <li><strong>Last updated:</strong> YYYY-MM-DD</li>
              <li><strong>Current milestone:</strong> MILESTONE</li>
              <li><strong>Main blocker:</strong> BLOCKER</li>
              <li><strong>Next action:</strong> NEXT ACTION</li>
            </ul>
          </div>

          <div className={styles.cardGrid}>
            {cards.map((card) => (
              <Card key={card.title} {...card} />
            ))}
          </div>
        </section>
      </main>
    </Layout>
  );
}