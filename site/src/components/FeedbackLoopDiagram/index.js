import React from 'react';
import clsx from 'clsx';
import katex from 'katex';

function renderLatex(value) {
  return {
    __html: katex.renderToString(value, {
      displayMode: false,
      throwOnError: false,
      strict: false,
    }),
  };
}

function MathText({value, className}) {
  return <span className={className} dangerouslySetInnerHTML={renderLatex(value)} />;
}

function normalizeCssLength(value, fallback) {
  if (typeof value === 'number') {
    return `${value}px`;
  }
  return value || fallback;
}

function Node({x, y, w, h, kind = 'box', label, sublabel}) {
  return (
    <foreignObject x={x} y={y} width={w} height={h}>
      <div className={clsx('feedback-loop__node', `feedback-loop__node--${kind}`)}>
        <MathText value={label} className="feedback-loop__label" />
        {sublabel && <MathText value={sublabel} className="feedback-loop__sublabel" />}
      </div>
    </foreignObject>
  );
}

function EdgeLabel({x, y, label, tone = 'default'}) {
  return (
    <foreignObject x={x} y={y} width="130" height="34">
      <div className={clsx('feedback-loop__edge-label', `feedback-loop__edge-label--${tone}`)}>
        <MathText value={label} />
      </div>
    </foreignObject>
  );
}

export default function FeedbackLoopDiagram({
  id,
  maxWidth = '920px',
  scale = 1,
  className,
}) {
  return (
    <>
      {id && <span id={id} className="doc-anchor" aria-hidden="true" />}
      <div
        className={clsx('feedback-loop', className)}
        style={{
          '--feedback-loop-max-width': normalizeCssLength(maxWidth, '920px'),
          '--feedback-loop-scale': String(scale || 1),
        }}
      >
        <svg className="feedback-loop__svg" viewBox="0 0 920 430" role="img" aria-label="Closed-loop training dynamics for a two-layer neural network">
          <defs>
            <marker id="feedback-loop-arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
              <path d="M 0 0 L 12 6 L 0 12 z" className="feedback-loop__arrow-head" />
            </marker>
            <marker id="feedback-loop-arrow-red" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
              <path d="M 0 0 L 12 6 L 0 12 z" className="feedback-loop__arrow-head-red" />
            </marker>
          </defs>

          <path className="feedback-loop__edge" d="M 45 125 L 145 125" />
          <path className="feedback-loop__edge" d="M 285 125 L 405 125" />
          <path className="feedback-loop__edge" d="M 545 125 C 610 125, 650 126, 704 148" />
          <path className="feedback-loop__edge" d="M 748 195 C 725 240, 655 275, 560 303" />
          <path className="feedback-loop__edge" d="M 405 310 L 295 310" />
          <path className="feedback-loop__edge" d="M 150 310 L 55 310" />

          <path className="feedback-loop__edge feedback-loop__edge--red" d="M 475 270 C 455 228, 455 195, 475 165" />
          <path className="feedback-loop__edge feedback-loop__edge--red" d="M 225 270 C 205 230, 205 198, 225 165" />
          <path className="feedback-loop__edge feedback-loop__edge--red feedback-loop__edge--soft" d="M 470 270 C 375 212, 315 190, 250 160" />
          <path className="feedback-loop__edge feedback-loop__edge--red feedback-loop__edge--soft" d="M 490 270 C 565 230, 565 188, 520 160" />

          <EdgeLabel x={54} y={84} label={String.raw`x`} />
          <EdgeLabel x={324} y={84} label={String.raw`a_1`} />
          <EdgeLabel x={584} y={92} label={String.raw`\hat{y}`} />
          <EdgeLabel x={640} y={244} label={String.raw`g_{\hat{y}}`} />
          <EdgeLabel x={328} y={272} label={String.raw`g_{a_1}`} />
          <EdgeLabel x={65} y={272} label={String.raw`g_x`} />
          <EdgeLabel x={490} y={200} label={String.raw`\dot{\theta}_2=-\eta g_{\theta_2}`} tone="red" />
          <EdgeLabel x={155} y={200} label={String.raw`\dot{\theta}_1=-\eta g_{\theta_1}`} tone="red" />

          <Node x={145} y={80} w={140} h={90} label={String.raw`\theta_1`} sublabel={String.raw`W_1,b_1`} />
          <Node x={405} y={80} w={140} h={90} label={String.raw`\theta_2`} sublabel={String.raw`W_2,b_2`} />
          <Node x={704} y={116} w={88} h={88} kind="loss" label={String.raw`\mathcal{L}`} />
          <Node x={405} y={265} w={150} h={90} kind="grad" label={String.raw`g_{\theta_2}`} sublabel={String.raw`\nabla_{\theta_2}\mathcal{L}`} />
          <Node x={145} y={265} w={150} h={90} kind="grad" label={String.raw`g_{\theta_1}`} sublabel={String.raw`\nabla_{\theta_1}\mathcal{L}`} />
        </svg>
      </div>
    </>
  );
}
