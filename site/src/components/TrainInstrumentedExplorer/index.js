import React, {useMemo, useState} from 'react';
import clsx from 'clsx';
import katex from 'katex';

const text = (value) => ({type: 'text', value});
const kw = (value) => ({type: 'kw', value});
const code = (value) => ({type: 'code', value});
const math = (value) => ({type: 'math', value});

const topAlgorithm = [
  {level: 0, parts: [kw('input'), text(' data '), math('X,Y'), text(', learning rate '), math('\\eta'), text(', controller settings, precision map '), math('P')]},
  {level: 0, parts: [math('P \\leftarrow \\operatorname{ensure\\_precision\\_dict}(\\texttt{precision\\_dict})')]},
  {level: 0, parts: [math('\\operatorname{validate\\_model}(P,\\operatorname{model})')]},
  {level: 0, parts: [math('D \\leftarrow \\operatorname{batch}(\\operatorname{Dataset}(X,Y))')]},
  {level: 0, parts: [kw('for'), text(' each epoch and batch '), kw('do')]},
  {level: 1, parts: [math('\\theta_t \\leftarrow \\operatorname{flatten}(\\operatorname{trainable\\ variables})')]},
  {level: 1, parts: [math('\\hat{y}_t,L_t,G_t \\leftarrow \\operatorname{ForwardGradientBlock}(\\operatorname{batch},P)')]},
  {level: 1, parts: [math('C_t,S_t \\leftarrow \\operatorname{CurvatureProxyBlock}(\\theta_t,G_t)')]},
  {level: 1, parts: [math('\\alpha_t,\\eta^{\\mathrm{eff}}_t \\leftarrow \\operatorname{ControllerBlock}(C_t,S_t,\\chi,\\eta)')]},
  {level: 1, parts: [math('\\theta_{t+1} \\leftarrow \\operatorname{UpdateStorageBlock}(\\theta_t,G_t,\\eta^{\\mathrm{eff}}_t,P)')]},
  {level: 1, parts: [math('\\operatorname{MetricsBlock}(\\ldots)')]},
  {level: 1, parts: [math('\\theta_{t-1},G_{t-1} \\leftarrow \\theta_t,G_t')]},
  {level: 0, parts: [kw('end for')]},
  {level: 0, parts: [kw('return'), text(' '), math('\\operatorname{FitHistory}(H)')]},
];

const blocks = {
  setup: {
    label: 'Setup',
    subtitle: 'precision, loss, dataset',
    algorithmTitle: 'Setup: train_instrumented(...)',
    algorithm: [
      {level: 0, parts: [math('P \\leftarrow \\operatorname{ensure\\_precision\\_dict}(\\texttt{precision\\_dict})')]},
      {level: 0, parts: [math('\\operatorname{validate\\_model}(P,\\operatorname{self.model})')]},
      {level: 0, parts: [math('X,Y \\leftarrow \\operatorname{float32}(X),\\operatorname{float32}(Y)')]},
      {level: 0, parts: [math('\\ell \\leftarrow \\operatorname{select\\_loss}(\\texttt{loss\\_mode})')]},
      {level: 0, parts: [math('D \\leftarrow \\operatorname{batch}(\\operatorname{Dataset}(X,Y),\\texttt{batch\\_size})')]},
      {level: 0, parts: [math('H \\leftarrow \\{\\}'), text(' history dictionary')]},
    ],
    notes: 'precision_dict=None preserves the floating-point EXP-000A path.',
    nodes: [
      {label: 'Precision map', detail: 'P'},
      {label: 'Validate names', detail: 'model layers'},
      {label: 'Data tensors', detail: 'X, Y'},
      {label: 'History store', detail: 'H'},
    ],
  },
  step: {
    label: 'Forward + Gradient',
    subtitle: 'loss and gradients',
    algorithmTitle: 'Step Block: Forward And Gradient',
    algorithm: [
      {level: 0, parts: [math('\\theta_t \\leftarrow \\operatorname{flatten}(\\texttt{model.trainable\\_variables})')]},
      {level: 0, parts: [kw('with'), text(' '), code('GradientTape'), text(' '), kw('do')]},
      {level: 1, parts: [math('\\hat{y}_b \\leftarrow \\operatorname{forward}(x_b,P)')]},
      {level: 1, parts: [math('L_t \\leftarrow \\ell(y_b,\\hat{y}_b)')]},
      {level: 0, parts: [kw('end with')]},
      {level: 0, parts: [math('\\{G_l\\}_l \\leftarrow \\nabla_{\\{\\theta_l\\}_l} L_t')]},
      {level: 0, parts: [math('G_t \\leftarrow \\operatorname{flatten}(\\{G_l\\}_l)')]},
    ],
    notes: 'When P is present, the forward path uses fake-quantized weights, activations, accumulators, and gradients.',
    nodes: [
      {label: 'Flatten variables', detail: 'theta_t'},
      {label: 'GradientTape', detail: 'record forward'},
      {label: 'Loss', detail: 'L_t'},
      {label: 'Gradients', detail: 'G_t'},
    ],
  },
  curvature: {
    label: 'Curvature Proxy',
    subtitle: 'C_t and EMA',
    algorithmTitle: 'Diagnostic: CurvatureProxy',
    algorithm: [
      {level: 0, parts: [math('\\Delta G_t \\leftarrow G_t-G_{t-1}')]},
      {level: 0, parts: [math('\\Delta\\theta_t \\leftarrow \\theta_t-\\theta_{t-1}')]},
      {level: 0, parts: [math('C_t \\leftarrow \\dfrac{\\lVert\\Delta G_t\\rVert_2}{\\lVert\\Delta\\theta_t\\rVert_2+\\varepsilon}')]},
      {level: 0, parts: [math('S_t \\leftarrow (1-\\rho)S_{t-1}+\\rho C_t')]},
      {level: 0, parts: [math('C^{\\mathrm{ctrl}}_t \\leftarrow \\max(C_t,S_t)')]},
    ],
    notes: 'This estimates local update-field sensitivity without computing the full Hessian.',
    nodes: [
      {label: 'Delta gradient', detail: 'dG_t'},
      {label: 'Delta theta', detail: 'dtheta_t'},
      {label: 'Proxy', detail: 'C_t'},
      {label: 'EMA', detail: 'S_t'},
    ],
  },
  controller: {
    label: 'Controller',
    subtitle: 'alpha and eta_eff',
    algorithmTitle: 'Controller: GlobalThrottle',
    algorithm: [
      {level: 0, parts: [math('\\alpha^{\\mathrm{would}}_t \\leftarrow \\min\\left(1,\\dfrac{\\chi}{\\eta(C^{\\mathrm{ctrl}}_t+\\varepsilon)}\\right)')]},
      {level: 0, parts: [kw('if'), text(' '), code('use_controller'), text(' '), kw('then')]},
      {level: 1, parts: [math('\\alpha_t \\leftarrow \\alpha^{\\mathrm{would}}_t')]},
      {level: 0, parts: [kw('else')]},
      {level: 1, parts: [math('\\alpha_t \\leftarrow 1')]},
      {level: 0, parts: [kw('end if')]},
      {level: 0, parts: [math('\\eta^{\\mathrm{eff}}_t \\leftarrow \\alpha_t\\eta')]},
    ],
    notes: 'Baseline runs still log alpha_would so intervention can be inspected offline.',
    nodes: [
      {label: 'Curvature input', detail: 'C_ctrl'},
      {label: 'Would throttle', detail: 'alpha_would'},
      {label: 'Controller switch', detail: 'use_controller?'},
      {label: 'Effective LR', detail: 'eta_eff'},
    ],
  },
  update: {
    label: 'Update + Storage',
    subtitle: 'apply update',
    algorithmTitle: 'Update: FloatOrQuantizedStorage',
    algorithm: [
      {level: 0, parts: [kw('for'), text(' each variable '), math('\\theta_l'), text(' and gradient '), math('G_l'), text(' '), kw('do')]},
      {level: 1, parts: [math('\\Delta_l \\leftarrow -\\eta^{\\mathrm{eff}}_t G_l')]},
      {level: 1, parts: [kw('if'), text(' update precision exists '), kw('then')]},
      {level: 2, parts: [math('\\Delta_l \\leftarrow Q_{\\mathrm{update},l}(\\Delta_l)')]},
      {level: 1, parts: [kw('end if')]},
      {level: 1, parts: [math('\\theta_l \\leftarrow \\theta_l+\\Delta_l')]},
      {level: 1, parts: [kw('if'), text(' storage precision exists '), kw('then')]},
      {level: 2, parts: [math('\\theta_l \\leftarrow Q_{\\mathrm{storage},l}(\\theta_l)')]},
      {level: 1, parts: [kw('end if')]},
      {level: 0, parts: [kw('end for')]},
    ],
    notes: 'This is where update quantization and stored-weight quantization are enforced.',
    nodes: [
      {label: 'Raw delta', detail: 'Delta_l'},
      {label: 'Update dtype?', detail: 'Q_update'},
      {label: 'Assign', detail: 'theta_l + Delta_l'},
      {label: 'Storage dtype?', detail: 'Q_storage'},
    ],
  },
  metrics: {
    label: 'Metrics',
    subtitle: 'logs and diagnostics',
    algorithmTitle: 'Metrics: LogStep',
    algorithm: [
      {level: 0, parts: [math('\\operatorname{log}(L_t,\\operatorname{rmse},\\lVert\\theta_t\\rVert_2,\\lVert G_t\\rVert_2)')]},
      {level: 0, parts: [math('\\operatorname{log}(\\lVert\\Delta\\theta_{\\mathrm{raw}}\\rVert_2,\\lVert\\Delta\\theta_{\\mathrm{actual}}\\rVert_2)')]},
      {level: 0, parts: [math('\\operatorname{log}(C_t,S_t,\\alpha_t,\\eta^{\\mathrm{eff}}_t)')]},
      {level: 0, parts: [kw('if'), text(' Hessian metrics are available '), kw('then')]},
      {level: 1, parts: [math('\\operatorname{log}(\\lambda_{\\max},\\operatorname{margins},\\rho(I-\\eta H))')]},
      {level: 0, parts: [kw('end if')]},
      {level: 0, parts: [kw('if'), text(' precision map is enabled '), kw('then')]},
      {level: 1, parts: [math('\\operatorname{log}(\\operatorname{rail\\ pressure},\\operatorname{underflow})')]},
      {level: 0, parts: [kw('end if')]},
    ],
    notes: 'The metrics table below defines the exact logged fields and interpretation.',
    nodes: [
      {label: 'Core logs', detail: 'loss, norms'},
      {label: 'Controller logs', detail: 'alpha, eta_eff'},
      {label: 'Stability logs', detail: 'lambda, rho'},
      {label: 'Rail logs', detail: 'saturation'},
    ],
  },
};

const topFlow = ['step', 'curvature', 'controller', 'update', 'metrics'];

function MathInline({latex}) {
  const html = katex.renderToString(latex, {
    displayMode: false,
    throwOnError: false,
    strict: false,
  });

  return <span className="pseudo-math" dangerouslySetInnerHTML={{__html: html}} />;
}

function AlgorithmPart({part}) {
  if (part.type === 'kw') {
    return <span className="pseudo-kw">{part.value}</span>;
  }
  if (part.type === 'code') {
    return <code>{part.value}</code>;
  }
  if (part.type === 'math') {
    return <MathInline latex={part.value} />;
  }
  return <>{part.value}</>;
}

function AlgorithmLine({line}) {
  return (
    <li>
      <span className={clsx(line.level > 0 && `pseudo-indent-${line.level}`)}>
        {line.parts.map((part, index) => (
          <AlgorithmPart key={`${part.type}-${index}-${part.value}`} part={part} />
        ))}
      </span>
    </li>
  );
}

function Algorithm({title, lines, caption}) {
  return (
    <div className="pseudo train-explorer__algorithm">
      <div className="pseudo-title">{title}</div>
      <div className="pseudo-code">
        <ol>
          {lines.map((line, index) => (
            <AlgorithmLine key={`${title}-${index}`} line={line} />
          ))}
        </ol>
      </div>
      {caption && <div className="pseudo-caption">{caption}</div>}
    </div>
  );
}

function FlowNode({block, top, active, onClick}) {
  return (
    <button
      type="button"
      className={clsx('train-flow__node', active && 'train-flow__node--active')}
      style={top === undefined ? undefined : {top}}
      onClick={onClick}
    >
      <strong>{block.label}</strong>
      <span>{block.subtitle}</span>
    </button>
  );
}

function FlowConnector({variant}) {
  return <div className={clsx('train-flow__connector', variant && `train-flow__connector--${variant}`)} />;
}

function StaticNode({node, top}) {
  return (
    <div className={clsx('train-flow__node', 'train-flow__node--static')} style={{top}}>
      <strong>{node.label}</strong>
      <span>{node.detail}</span>
    </div>
  );
}

function TopDiagram({selected, setSelected}) {
  return (
    <div className="train-flow train-flow--top" aria-label="train_instrumented top-level flow">
      <div className="train-flow__side-rail train-flow__side-rail--false">
        <span>false</span>
      </div>
      <div className="train-flow__side-rail train-flow__side-rail--loop">
        <span>next batch</span>
      </div>

      <div className="train-flow__stack">
        <div className="train-flow__terminal">Start</div>
        <FlowConnector />
        <FlowNode block={blocks.setup} active={selected === 'setup'} onClick={() => setSelected('setup')} />
        <FlowConnector />
        <div className="train-flow__diamond-wrap">
          <div className="train-flow__diamond">
            <strong>More batches?</strong>
            <span>epoch / batch loop</span>
          </div>
        </div>
        <FlowConnector variant="true" />
        <div className="train-flow__branch train-flow__branch--true">true</div>
        {topFlow.map((key, index) => (
          <React.Fragment key={key}>
            <FlowNode block={blocks[key]} active={selected === key} onClick={() => setSelected(key)} />
            {index < topFlow.length - 1 && <FlowConnector />}
          </React.Fragment>
        ))}
        <div className="train-flow__loop-spacer" />
        <div className="train-flow__terminal">Return FitHistory</div>
      </div>
    </div>
  );
}

function DetailDiagram({block}) {
  const height = 210 + block.nodes.length * 112;
  return (
    <div className="train-flow train-flow--detail" style={{height}}>
      <svg className="train-flow__edges" viewBox={`0 0 420 ${height}`} preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <marker id="train-arrow-detail" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M 0 0 L 6 3 L 0 6 z" />
          </marker>
        </defs>
        <path d="M210 90 L210 110" />
        {block.nodes.map((_, index) => (
          <path key={`edge-${index}`} d={`M210 ${190 + index * 112} L210 ${222 + index * 112}`} />
        ))}
      </svg>
      <div className="train-flow__terminal" style={{top: 20}}>Enter {block.label}</div>
      {block.nodes.map((node, index) => (
        <StaticNode key={`${node.label}-${node.detail}`} node={node} top={118 + index * 112} />
      ))}
      <div className="train-flow__terminal" style={{top: 118 + block.nodes.length * 112}}>
        Exit {block.label}
      </div>
    </div>
  );
}

export default function TrainInstrumentedExplorer() {
  const [selected, setSelected] = useState('top');
  const current = selected === 'top' ? null : blocks[selected];
  const algorithm = useMemo(() => {
    if (!current) {
      return {
        title: 'Algorithm: train_instrumented(...)',
        lines: topAlgorithm,
        caption: 'The top-level loop shows where each internal block is called. Click any block in the diagram to inspect it.',
      };
    }
    return {
      title: current.algorithmTitle,
      lines: current.algorithm,
      caption: current.notes,
    };
  }, [current]);

  return (
    <div className="train-explorer">
      <div className="train-explorer__header">
        <div>
          <strong>{current ? current.label : 'train_instrumented(...)'}</strong>
          <span>{current ? current.subtitle : 'top-level online training loop'}</span>
        </div>
        {current && (
          <button type="button" className="train-explorer__back" onClick={() => setSelected('top')}>
            ← Back to loop
          </button>
        )}
      </div>

      <div className="train-explorer__body">
        <div className="train-explorer__diagram">
          {current ? <DetailDiagram block={current} /> : <TopDiagram selected={selected} setSelected={setSelected} />}
        </div>
        <Algorithm title={algorithm.title} lines={algorithm.lines} caption={algorithm.caption} />
      </div>
    </div>
  );
}
