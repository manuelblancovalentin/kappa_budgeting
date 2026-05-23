import matplotlib.pyplot as plt
import numpy as np


class FitHistory(dict[str, np.ndarray]):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __repr__(self) -> str:
        s = 'FitHistory:\n'
        for k, v in self.items():
            s += f'  {k}: {v.shape}\n'
        return s

    def plot_results(self, title=None):
        fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

        if title:
            fig.suptitle(title)

        axs[0, 0].plot(self['loss'])
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].grid(True)

        axs[0, 1].plot(self['weight_error_fro'])
        axs[0, 1].set_title(r'Weight Error $\|W-A\|_F$')
        axs[0, 1].set_ylabel('Error')
        axs[0, 1].grid(True)

        axs[1, 0].plot(self['grad_norm'])
        axs[1, 0].set_title('Gradient Norm')
        axs[1, 0].set_ylabel(r'$\|G_t\|_2$')
        axs[1, 0].grid(True)

        axs[1, 1].plot(self['curvature_proxy'], label='C proxy', alpha=0.5)
        axs[1, 1].plot(self['curvature_ema'], label='EMA')
        axs[1, 1].plot(self['hessian_lambda_max'], label=r'$\lambda_{\max}(H)$', linestyle='--')
        axs[1, 1].set_title('Curvature Proxy vs True Hessian')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        axs[2, 0].plot(self['stability_margin_lambda_raw'], label=r'$\eta\lambda_{\max}$')
        axs[2, 0].axhline(2.0, linestyle='--', label='stability limit')
        axs[2, 0].set_title('Raw Stability Margin')
        axs[2, 0].set_xlabel('Step')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        axs[2, 1].plot(self['alpha_would'])
        axs[2, 1].set_title(r'Would-be $\alpha_t$')
        axs[2, 1].set_xlabel('Step')
        axs[2, 1].set_ylabel(r'$\alpha_t$')
        axs[2, 1].grid(True)

        plt.tight_layout()
        plt.show()
