from collections.abc import Callable
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners import learner_utils
from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
)
from supermimic.models import build_mlp


def _ids(params) -> list[int]:
  return [id(p) for p in params]


class _FakeCudaState:
  is_cuda = True
  device = torch.device("cuda", 0)

  def __init__(self, ptr: int = 1024):
    self._ptr = ptr

  def data_ptr(self) -> int:
    return self._ptr


class _FakeCudaStream:
  def __init__(self, harness: "_FakeCudaHarness"):
    self._harness = harness

  def wait_stream(self, other: "_FakeCudaStream") -> None:
    del other

  def synchronize(self) -> None:
    assert not self._harness.capturing, "host sync inside CUDA graph capture"
    self._harness.events.append("sync")


class _FakeCudaGraph:
  def __init__(self, harness: "_FakeCudaHarness"):
    self._harness = harness

  def replay(self) -> None:
    assert not self._harness.capturing, "replay inside CUDA graph capture"
    self._harness.events.append("replay")
    if self._harness.on_replay is not None:
      self._harness.on_replay(self)


class _FakeCudaHarness:
  """Minimal CPU-runnable mock of the torch.cuda surface CapturedUpdate uses."""

  def __init__(self) -> None:
    self.capturing = False
    self.events: list[str] = []
    self.pool = object()
    self.capture_stream = _FakeCudaStream(self)
    self.current = _FakeCudaStream(self)
    self.on_replay: Callable[[_FakeCudaGraph], None] | None = None

  def install(self, monkeypatch: pytest.MonkeyPatch) -> "_FakeCudaHarness":
    @contextmanager
    def _device(_dev):
      yield

    @contextmanager
    def _stream(_stream):
      yield

    @contextmanager
    def _graph(_graph, *, pool, stream):
      assert pool is self.pool
      assert stream is self.capture_stream
      self.capturing = True
      self.events.append("capture_enter")
      try:
        yield
      finally:
        self.capturing = False
        self.events.append("capture_exit")

    monkeypatch.setattr(torch.cuda, "device", _device)
    monkeypatch.setattr(torch.cuda, "stream", _stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: self.current)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: _FakeCudaGraph(self))
    monkeypatch.setattr(torch.cuda, "graph", _graph)
    monkeypatch.setattr(
      learner_utils,
      "_capture_stream_and_pool",
      lambda _dev: (self.capture_stream, self.pool),
    )
    return self


def test_muon_steps_the_hidden_matrices_and_adam_the_rest():
  torch.manual_seed(0)
  mlp = build_mlp(5, (8, 8, 8), 3)
  linears = [m for m in mlp if isinstance(m, nn.Linear)]
  hidden = [linears[1].weight, linears[2].weight]
  adam, muon = build_optimizers([mlp], 1e-3)
  assert isinstance(muon, torch.optim.Muon)
  assert _ids(muon.param_groups[0]["params"]) == _ids(hidden)
  rest = [p for p in mlp.parameters() if id(p) not in _ids(hidden)]
  assert _ids(adam.param_groups[0]["params"]) == _ids(rest)
  before = [p.detach().clone() for p in mlp.parameters()]
  _optimize_step([adam, muon], mlp(torch.randn(4, 5)).square().mean())
  for param, old in zip(mlp.parameters(), before, strict=True):
    assert not torch.equal(param, old)


def test_captured_update_outputs_do_not_alias_pool_on_capture_or_replay(
  monkeypatch: pytest.MonkeyPatch,
):
  harness = _FakeCudaHarness().install(monkeypatch)
  state = [_FakeCudaState(ptr=111)]

  def fn(td: TensorDict) -> dict[str, torch.Tensor]:
    # Include views into a 1D tensor (like WorldModelLearner's group_mse means).
    stats = torch.stack([td["x"].sum(), td["x"].mean()])
    return {"sum": stats[0], "mean": stats[1]}

  captured = CapturedUpdate(fn, lambda: state, warmup=1)

  def on_replay(_graph: _FakeCudaGraph) -> None:
    assert captured._inputs is not None
    fresh = fn(captured._inputs)
    for k, v in fresh.items():
      captured._outputs[k].copy_(v)

  harness.on_replay = on_replay

  # Warmup runs eagerly (no graph capture or replay).
  warmup_out = captured(TensorDict({"x": torch.tensor([1.0, 3.0])}, batch_size=[]))
  assert harness.events == []
  assert warmup_out["sum"].item() == pytest.approx(4.0)
  assert warmup_out["mean"].item() == pytest.approx(2.0)

  # Capture call records the graph, replays once, clones out of pool, and syncs.
  capture_out = captured(TensorDict({"x": torch.tensor([2.0, 6.0])}, batch_size=[]))
  assert harness.events == ["capture_enter", "capture_exit", "replay", "sync"]
  assert capture_out["sum"].item() == pytest.approx(8.0)
  assert capture_out["mean"].item() == pytest.approx(4.0)
  for k in capture_out:
    assert capture_out[k].data_ptr() != captured._outputs[k].data_ptr()

  # Simulate another CapturedUpdate overwriting the shared scratch pool in place.
  for v in captured._outputs.values():
    v.fill_(999.0)
  assert capture_out["sum"].item() == pytest.approx(8.0)
  assert capture_out["mean"].item() == pytest.approx(4.0)

  # Replay call computes fresh values into _outputs and returns an isolated clone.
  harness.events.clear()
  replay_out = captured(TensorDict({"x": torch.tensor([10.0, 20.0])}, batch_size=[]))
  assert harness.events == ["replay", "sync"]
  assert replay_out["sum"].item() == pytest.approx(30.0)
  assert replay_out["mean"].item() == pytest.approx(15.0)
  for k in replay_out:
    assert replay_out[k].data_ptr() != captured._outputs[k].data_ptr()

  # Overwriting _outputs again must not corrupt either prior return dict.
  for v in captured._outputs.values():
    v.fill_(-123.0)
  assert replay_out["sum"].item() == pytest.approx(30.0)
  assert replay_out["mean"].item() == pytest.approx(15.0)
  assert capture_out["sum"].item() == pytest.approx(8.0)
  assert capture_out["mean"].item() == pytest.approx(4.0)

  # A subsequent replay also must not overwrite replay_out.
  replay_out_2 = captured(TensorDict({"x": torch.tensor([5.0, 7.0])}, batch_size=[]))
  assert replay_out_2["sum"].item() == pytest.approx(12.0)
  assert replay_out["sum"].item() == pytest.approx(30.0)


def test_captured_update_shared_pool_interleaved_wm_after_policy(
  monkeypatch: pytest.MonkeyPatch,
):
  """Regression test for dmo_hold=1 where WM replays after Policy and clobbers
  Policy's shared-pool _outputs before logger.log() reads loss_dict."""
  harness = _FakeCudaHarness().install(monkeypatch)
  wm_state = [_FakeCudaState(ptr=10)]
  policy_state = [_FakeCudaState(ptr=20)]

  def wm_fn(td: TensorDict) -> dict[str, torch.Tensor]:
    return {"Loss/state": td["x"].square().mean()}

  def policy_fn(td: TensorDict) -> dict[str, torch.Tensor]:
    return {
      "Policy/replay_action_gap": td["x"].abs().mean(),
      "Policy/value_function": td["x"].mean(),
    }

  wm = CapturedUpdate(wm_fn, lambda: wm_state, warmup=0)
  policy = CapturedUpdate(policy_fn, lambda: policy_state, warmup=0)

  def on_replay(graph: _FakeCudaGraph) -> None:
    if graph is wm._graph:
      assert wm._inputs is not None
      for k, v in wm_fn(wm._inputs).items():
        wm._outputs[k].copy_(v)
      # WM scratch memory overlaps Policy's pool-allocated _outputs.
      for v in policy._outputs.values():
        v.fill_(2.9)
    elif graph is policy._graph:
      assert policy._inputs is not None
      for k, v in policy_fn(policy._inputs).items():
        policy._outputs[k].copy_(v)

  harness.on_replay = on_replay

  loss_dict: dict[str, torch.Tensor] = {}
  # WM captures first, Policy captures second (into WM's freed scratch pool).
  loss_dict.update(wm(TensorDict({"x": torch.tensor([2.0, 2.0])}, batch_size=[])))
  loss_dict.update(
    policy(TensorDict({"x": torch.tensor([0.001, 0.003])}, batch_size=[]))
  )

  # Next steps: WM replays after Policy before logger reads loss_dict.
  loss_dict.update(wm(TensorDict({"x": torch.tensor([3.0, 3.0])}, batch_size=[])))
  assert policy._outputs["Policy/replay_action_gap"].item() == pytest.approx(2.9)
  assert loss_dict["Policy/replay_action_gap"].item() == pytest.approx(0.002)
  assert loss_dict["Policy/value_function"].item() == pytest.approx(0.002)
  assert loss_dict["Loss/state"].item() == pytest.approx(9.0)


def test_captured_update_clones_nested_detaches_and_preserves_non_tensors(
  monkeypatch: pytest.MonkeyPatch,
):
  _FakeCudaHarness().install(monkeypatch)
  state = [_FakeCudaState(ptr=42)]

  def fn(td: TensorDict) -> dict:
    grad_tensor = td["x"].detach().requires_grad_(True) * 2.0
    return {
      "grad": grad_tensor.sum(),
      "nested": {"inner": td["x"].mean()},
      "td": TensorDict({"v": td["x"].sum()}, batch_size=[]),
      "pair": (td["x"][0], [td["x"][1]]),
      "meta": "ok",
      "count": 3,
    }

  captured = CapturedUpdate(fn, lambda: state, warmup=0)
  out = captured(TensorDict({"x": torch.tensor([1.5, 2.5])}, batch_size=[]))

  # Corrupt all underlying captured buffers in place.
  captured._outputs["grad"].detach().fill_(77.0)
  captured._outputs["nested"]["inner"].fill_(77.0)
  captured._outputs["td"]["v"].fill_(77.0)
  captured._outputs["pair"][0].fill_(77.0)
  captured._outputs["pair"][1][0].fill_(77.0)

  assert not out["grad"].requires_grad
  assert out["grad"].item() == pytest.approx(8.0)
  assert out["nested"]["inner"].item() == pytest.approx(2.0)
  assert out["td"]["v"].item() == pytest.approx(4.0)
  assert out["pair"][0].item() == pytest.approx(1.5)
  assert out["pair"][1][0].item() == pytest.approx(2.5)
  assert out["meta"] == "ok"
  assert out["count"] == 3


def test_captured_update_cpu_eager_and_moved_state_error(
  monkeypatch: pytest.MonkeyPatch,
):
  # CPU path runs eagerly without touching CUDA graph capture.
  cpu_state = [torch.zeros(1)]
  sentinel = {"loss": torch.tensor(1.25)}
  cpu_captured = CapturedUpdate(lambda _td: sentinel, lambda: cpu_state, warmup=0)
  cpu_out = cpu_captured(TensorDict({"x": torch.zeros(1)}, batch_size=[]))
  assert cpu_out is sentinel

  # Moving a captured state tensor raises on replay.
  _FakeCudaHarness().install(monkeypatch)
  cuda_state = [_FakeCudaState(ptr=100)]
  cuda_captured = CapturedUpdate(
    lambda td: {"loss": td["x"].sum()}, lambda: cuda_state, warmup=0
  )
  cuda_captured(TensorDict({"x": torch.tensor([1.0])}, batch_size=[]))
  cuda_state[0] = _FakeCudaState(ptr=200)
  with pytest.raises(RuntimeError, match="a captured tensor moved"):
    cuda_captured(TensorDict({"x": torch.tensor([2.0])}, batch_size=[]))
