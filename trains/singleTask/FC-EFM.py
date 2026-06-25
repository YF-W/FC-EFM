from .EmotionFlow import EmotionFlowTrain

__all__ = ["EmotionFlowFFTrain"]


class EmotionFlowFFTrain(EmotionFlowTrain):
    """
    Reuse EmotionFlow trainer behavior with explicit output contract checks
    for EmotionFlow_FF.
    """

    REQUIRED_OUTPUT_KEYS = (
        "M",
        "mu0",
        "mu_T",
        "mu_A",
        "mu_V",
        "f_T",
        "f_A",
        "f_V",
        "omega",
        "w",
    )

    def _check_model_contract(self, model):
        real_model = model.Model if hasattr(model, "Model") else model
        model_name = type(real_model).__name__
        if model_name != "EmotionFlowFF":
            raise RuntimeError(
                f"[Contract] EmotionFlowFFTrain expects EmotionFlowFF, got {model_name}"
            )

    def _check_output_tensors(self, outputs, mode, epoch, step_idx, meta):
        missing = [k for k in self.REQUIRED_OUTPUT_KEYS if k not in outputs]
        if missing:
            raise RuntimeError(
                f"[Contract] missing required output keys={missing} "
                f"mode={mode} epoch={epoch} batch={step_idx}"
            )
        super()._check_output_tensors(outputs, mode, epoch, step_idx, meta)

    def do_train(self, model, dataloader, return_epoch_results=False):
        self._check_model_contract(model)
        return super().do_train(model, dataloader, return_epoch_results=return_epoch_results)
