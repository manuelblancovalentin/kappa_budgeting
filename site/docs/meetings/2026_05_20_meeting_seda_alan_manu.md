---
id: 2026-05-20-meeting-seda-alan-manu
title: "Alan, Seda, Manu sync"
sidebar_label: "🗞️ 2026-05-20 Seda/Alan/Manu"
slug: /meetings/2026-05-20-meeting-seda-alan-manu
status:
  - completed
tags:
  - meeting
  - project-log
last_modified: 2026-05-20
author: mbvalentin
meeting_id: MTG-2026-05-20-STATUS
date: 2026-05-20
attendees:
  - mbvalentin
  - alan-guo
  - seda-ogrenci
---
# Alan, Seda, Manu sync
<PageMeta />
---

## Updates on batchnorm and convolutional layers
<TBox type="summary" title="Batch-norm updates (Andrew and Darrel)">
    We discussed the current progress of Andrew and Darrel on training models with batchnorm. They basically ran the `MSWC` notebook but added batchnorm (no training, no propagation of gradients): just training dense layers. The main question was `does adding batchnorm layers to a model make it inherently more robust to drift?`. The results that they got is that apparently it does not, which is good news for <ENABOL />. In fact, the accuracy seems to be better with batchnorm overall and we can also recover much more accuracy on hardware. 
    **Conclusion**: batchnorm improves accuracy (this is why people use them), but <ENABOL /> is still useful and required for batch-norm-based models.
</TBox>


### Next steps for <Person id="andrew-tieu" /> and <Person id="darrel-zhao" />
- [ ] Re-run the `MSWC` notebook (with or without batchnorm) but now training the convolutional layers. This requires two things:
    * Making sure that the gradients still flow through the dense layers (even if they are frozen), to confirm the logic of the `apply_update` flag is consistent in `dense_backpass`. 
    * Making sure that the backpass for the `conv2d` layer is correct.
    * **This will force them to dig into the `cpp` code and understand how backprop works**
- [ ] Later, re-run the experiment but now adding batchnorm layers as well (no training of batchnorm layers yet, just training conv layers). <font color="red">**Important: this assumes that the backprop for batchnorm is correct, see Alan tasks below**</font>.

---

## Updates on column throttling, $\kappa$-budgeting

<TBox type="info" title="Column throttling and kappa budgeting updates (Alan)">
    Alan has been checking what the throttling of columns looked like on a per-layer basis, and for different budgets. He realized that the distribution of throttling was very uneven (meaning, some layer would get a lot of throttling, while others almost none). **This is a big problem because it means that while learning the direction of the updates is changing**. This is why when we had the full `RowScale` learning didn't work. We are fixing this with the `global throttling` (see notes on Manu updates below).
</TBox>

### Next steps for <Person id="alan-guo" />

- [ ] First, settle on a testing workflow that **always** logs three points for any test (basically the three points we have in our paper): 
    * baseline accuracy on HARDWARE without any drift (this is error introduced due to quantization, and the theoretical maximum we could recover on hardware).
    * accuracy on HARDWARE with drift at `epoch=0` (this is the first point of the training curve, it shows how bad the accuracy degrades without <ENABOL />).
    * accuracy on HARDWARE with drift at `epoch=END` (this is the last point of the training curve, it shows how much we recovered with <ENABOL />).
- [ ] Correct the backprop pass for batchnorm layers (even if we are not training them, we still need to make sure the backprop chain is correct).
    * First, I would do this manually, modifying the `cpp` code directly in `firmware`, before trying to do the `templates` and `vivado-writer` etc.
    * Later, we'll put it back onto `hls4ml` to make it automatic. 
- [ ] Consider using always `training/validation` split data to make our results more robust.

---

## Updates on global throttling
<TBox type="success" title="Global throttling updates (Manu)" >
    Manu has been working on a new version of the stability algorithm that uses a global throttle instead of a per-layer independent budgeting/throttle. It also created this documentation page and ran the initial ablation tests on a dummy model to test the hypothesis [exp-000-a](../experiments/exp-000a-global-throttle-float-lin1.md) and [exp-000-b](../experiments/exp-000b-global-throttle-qfx-lin1.md). The results show that indeed there is a divergence zone that will make the quantized model be unstable, and that global throttling can fix this [exp-001](../experiments/exp-001-global-throttle-parameter-sweep.md). It's left for us to decide how we can reconcile this with the old $\kappa$-budgeting idea: are we just saying that this is a `new implementation` of the idea? Or this is ruling out our previous method? 

</TBox>

### Next steps for <Person id="mbvalentin" />
- [ ] Re-run the ablation tests but now set $\alpha(0) = 0$ instead of $\alpha(0) = 1$ to confirm that the system is able to always keep training stable (in the first tests, because $\alpha(0) = 1$, a big enough learning rate $\eta$ would still be able to diverge in the first step, without the global throttle doing its job).
- [ ] Run the rest of ablation tests and check that the controller still works.
    - [ ] Distribution shift.
    - [ ] Transient spike in gradients, or inputs, or some path that would cause instabilities and divergence. 
    - [ ] Quantization ablations (rail saturation / underflow, dead-zone).
- [ ] Pull latest `hls4ml` code and start integrating the global throttle mechanism into it.
- [ ] Be able to do `csim` with the global throttle before leaving for AMD so I can hand this off.
