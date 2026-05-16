import React, {useMemo, useState} from 'react';
import clsx from 'clsx';

const topAlgorithm = [
  {level: 0, kind: 'kw', text: 'input', rest: 'X, Y, learning rate eta, controller settings, precision map P'},
  {level: 0, text: 'P <- ensure_precision_dict(precision_dict)'},
  {level: 0, text: 'validate_model(P, model)'},
  {level: 0, text: 'D <- batched Dataset(X, Y)'},
  {level: 0, kind: 'kw', text: 'for', rest: 'each epoch and batch do'},
  {level: 1, text: 'theta_t <- flatten(trainable variables)'},
  {level: 1, text: 'y_hat, L_t, G_t <- ForwardGradientBlock(batch, P)'},
  {level: 1, text: 'C_t, S_t <- CurvatureProxyBlock(theta_t, G_t)'},
  {level: 1, text: 'alpha_t, eta_eff <- ControllerBlock(C_t, S_t, chi, eta)'},
  {level: 1, text: 'theta_{t+1} <- UpdateStorageBlock(theta_t, G_t, eta_eff, P)'},
  {level: 1, text: 'log MetricsBlock(...)'},
  {level: 1, text: 'theta_{t-1}, G_{t-1} <- theta_t, G_t'},
  {level: 0, kind: 'kw', text: 'end for'},
  {level: 0, kind: 'kw', text: 'return', rest: 'FitHistory(H)'},
];

const blocks = {
  setup: {
    label: 'Setup',
    subtitle: 'precision, loss, dataset',
    algorithmTitle: 'Setup: train_instrumented(...)',
    algorithm: [
      {level: 0, text: 'P <- ensure_precision_dict(precision_dict)'},
      {level: 0, text: 'validate_model(P, self.model)'},
      {level: 0, text: 'X, Y <- float32(X), float32(Y)'},
      {level: 0, text: 'loss_fn <- select_loss(loss_mode)'},
      {level: 0, text: 'D <- batch(Dataset(X, Y), batch_size)'},
      {level: 0, text: 'H <- empty history dictionary'},
    ],
    notes: 'precision_dict=None preserves the floating-point EXP-000A path.',
    nodes: [
      ['Precision map', 'P'],
      ['Validate names', 'model layers'],
      ['Data tensors', 'X, Y'],
      ['History store', 'H'],
    ],
  },
  step: {
    label: 'Forward + Gradient',
    subtitle: 'loss and gradients',
    algorithmTitle: 'Step Block: Forward And Gradient',
    algorithm: [
      {level: 0, text: 'theta_t <- flatten(model.trainable_variables)'},
      {level: 0, kind: 'kw', text: 'with', rest: 'GradientTape do'},
      {level: 1, text: 'y_hat <- forward(x_b, P)'},
      {level: 1, text: 'L_t <- loss_fn(y_b, y_hat)'},
      {level: 0, kind: 'kw', text: 'end with'},
      {level: 0, text: '{G_l}_l <- grad(L_t, {theta_l}_l)'},
      {level: 0, text: 'G_t <- flatten({G_l}_l)'},
    ],
    notes: 'When P is present, the forward path uses fake-quantized weights, activations, accumulators, and gradients.',
    nodes: [
      ['theta_t', 'flatten variables'],
      ['GradientTape', 'record forward'],
      ['Loss', 'L_t'],
      ['Gradients', 'G_t'],
    ],
  },
  curvature: {
    label: 'Curvature Proxy',
    subtitle: 'C_t and EMA',
    algorithmTitle: 'Diagnostic: CurvatureProxy',
    algorithm: [
      {level: 0, text: 'dG_t <- G_t - G_{t-1}'},
      {level: 0, text: 'dtheta_t <- theta_t - theta_{t-1}'},
      {level: 0, text: 'C_t <- ||dG_t||_2 / (||dtheta_t||_2 + eps)'},
      {level: 0, text: 'S_t <- (1-rho) S_{t-1} + rho C_t'},
      {level: 0, text: 'C_ctrl <- max(C_t, S_t)'},
    ],
    notes: 'This estimates local update-field sensitivity without computing the full Hessian.',
    nodes: [
      ['Delta gradient', 'dG_t'],
      ['Delta theta', 'dtheta_t'],
      ['Proxy', 'C_t'],
      ['EMA', 'S_t'],
    ],
  },
  controller: {
    label: 'Controller',
    subtitle: 'alpha and eta_eff',
    algorithmTitle: 'Controller: GlobalThrottle',
    algorithm: [
      {level: 0, text: 'alpha_would <- min(1, chi / (eta (C_ctrl + eps)))'},
      {level: 0, kind: 'kw', text: 'if', rest: 'use_controller then'},
      {level: 1, text: 'alpha_t <- alpha_would'},
      {level: 0, kind: 'kw', text: 'else'},
      {level: 1, text: 'alpha_t <- 1'},
      {level: 0, kind: 'kw', text: 'end if'},
      {level: 0, text: 'eta_eff <- alpha_t eta'},
    ],
    notes: 'Baseline runs still log alpha_would so intervention can be inspected offline.',
    nodes: [
      ['Curvature input', 'C_ctrl'],
      ['Would throttle', 'alpha_would'],
      ['Switch', 'use_controller?'],
      ['Effective LR', 'eta_eff'],
    ],
  },
  update: {
    label: 'Update + Storage',
    subtitle: 'apply update',
    algorithmTitle: 'Update: FloatOrQuantizedStorage',
    algorithm: [
      {level: 0, kind: 'kw', text: 'for', rest: 'each variable theta_l and gradient G_l do'},
      {level: 1, text: 'Delta_l <- - eta_eff G_l'},
      {level: 1, kind: 'kw', text: 'if', rest: 'update precision exists then'},
      {level: 2, text: 'Delta_l <- Q_update,l(Delta_l)'},
      {level: 1, kind: 'kw', text: 'end if'},
      {level: 1, text: 'theta_l <- theta_l + Delta_l'},
      {level: 1, kind: 'kw', text: 'if', rest: 'storage precision exists then'},
      {level: 2, text: 'theta_l <- Q_storage,l(theta_l)'},
      {level: 1, kind: 'kw', text: 'end if'},
      {level: 0, kind: 'kw', text: 'end for'},
    ],
    notes: 'This is where update quantization and stored-weight quantization are enforced.',
    nodes: [
      ['Raw delta', 'Delta_l'],
      ['Update dtype?', 'Q_update'],
      ['Assign', 'theta_l + Delta_l'],
      ['Storage dtype?', 'Q_storage'],
    ],
  },
  metrics: {
    label: 'Metrics',
    subtitle: 'logs and diagnostics',
    algorithmTitle: 'Metrics: LogStep',
    algorithm: [
      {level: 0, text: 'log loss, rmse, theta_norm, grad_norm'},
      {level: 0, text: 'log raw_update_norm and actual_update_norm'},
      {level: 0, text: 'log curvature_proxy, curvature_ema, alpha, eta_eff'},
      {level: 0, kind: 'kw', text: 'if', rest: 'Hessian metrics are available then'},
      {level: 1, text: 'log lambda_max, margins, spectral radii'},
      {level: 0, kind: 'kw', text: 'end if'},
      {level: 0, kind: 'kw', text: 'if', rest: 'precision map is enabled then'},
      {level: 1, text: 'log rail pressure and underflow statistics'},
      {level: 0, kind: 'kw', text: 'end if'},
    ],
    notes: 'The metrics table below defines the exact logged fields and interpretation.',
    nodes: [
      ['Core logs', 'loss, norms'],
      ['Controller logs', 'alpha, eta_eff'],
      ['Stability logs', 'lambda, rho'],
      ['Rail logs', 'saturation'],
    ],
  },
};

const topFlow = ['setup', 'step', 'curvature', 'controller', 'update', 'metrics'];

function AlgorithmLine({line, index}) {
  return (
    <li>
      <span className={clsx(line.level > 0 && `pseudo-indent-${line.level}`)}>
        {line.kind === 'kw' ? <span className="pseudo-kw">{line.text}</span> : line.text}
        {line.rest ? ` ${line.rest}` : ''}
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
            <AlgorithmLine key={`${title}-${index}`} line={line} index={index} />
          ))}
        </ol>
      </div>
      {caption && <div className="pseudo-caption">{caption}</div>}
    </div>
  );
}

function NodeButton({block, active, onClick}) {
  return (
    <button
      type="button"
      className={clsx('train-flow__node', active && 'train-flow__node--active')}
      onClick={onClick}
    >
      <strong>{block.label}</strong>
      <span>{block.subtitle}</span>
    </button>
  );
}

function TopDiagram({selected, setSelected}) {
  return (
    <div className="train-flow train-flow--top" aria-label="train_instrumented top-level flow">
      <div className="train-flow__terminal">Start</div>
      <div className="train-flow__arrow">↓</div>
      <NodeButton block={blocks.setup} active={selected === 'setup'} onClick={() => setSelected('setup')} />
      <div className="train-flow__arrow">↓</div>
      <div className="train-flow__diamond">
        <strong>More batches?</strong>
        <span>epoch / batch loop</span>
      </div>
      <div className="train-flow__branch train-flow__branch--true">true</div>
      <div className="train-flow__pipeline">
        {topFlow.slice(1).map((key, index) => (
          <React.Fragment key={key}>
            <NodeButton block={blocks[key]} active={selected === key} onClick={() => setSelected(key)} />
            {index < topFlow.length - 2 && <div className="train-flow__arrow train-flow__arrow--inline">→</div>}
          </React.Fragment>
        ))}
      </div>
      <div className="train-flow__loop">loop back to next batch</div>
      <div className="train-flow__branch train-flow__branch--false">false</div>
      <div className="train-flow__terminal">Return FitHistory</div>
    </div>
  );
}

function DetailDiagram({block}) {
  return (
    <div className="train-flow train-flow--detail">
      <div className="train-flow__terminal">Enter {block.label}</div>
      <div className="train-flow__arrow">↓</div>
      <div className="train-flow__pipeline train-flow__pipeline--detail">
        {block.nodes.map(([label, detail], index) => (
          <React.Fragment key={`${label}-${detail}`}>
            <div className="train-flow__node train-flow__node--static">
              <strong>{label}</strong>
              <span>{detail}</span>
            </div>
            {index < block.nodes.length - 1 && <div className="train-flow__arrow train-flow__arrow--inline">→</div>}
          </React.Fragment>
        ))}
      </div>
      <div className="train-flow__arrow">↓</div>
      <div className="train-flow__terminal">Exit {block.label}</div>
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
