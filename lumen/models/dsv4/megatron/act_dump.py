"""Dump per-TransformerLayer hidden states for Lumen vs Miles forward compare.

Enable with DSV4_DUMP_LAYER_ACT=1.  Only TP=0 and EP=0 write files (SP is
gathered first when the tensor looks like a hidden state).  Default is one
forward per hook to keep dumps small.

Layer-1 inner probes (default on): DSV4_DUMP_LAYER1_INNER=1.
Files: {tag}_ppXX_l01_{stage}_fwd0.pt
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _enabled() -> bool:
    return os.environ.get("DSV4_DUMP_LAYER_ACT", "0") == "1"


def _inner_enabled() -> bool:
    return os.environ.get("DSV4_DUMP_LAYER1_INNER", "1") == "1"


def _stop_after_layer() -> int:
    """Skip compute after this layer_number. 0 = run all layers.

    When layer-act dump is on, default to 1 so a layer-1 compare does not
    run the remaining MLA/MoE stack. Set DSV4_DUMP_STOP_AFTER_LAYER=0 to dump
    every layer again.
    """
    raw = os.environ.get("DSV4_DUMP_STOP_AFTER_LAYER")
    if raw is None or str(raw).strip() == "":
        return 1 if _enabled() else 0
    return int(raw)


def _patch_forward_only() -> None:
    """Skip backward after the dump forward."""
    try:
        import megatron.core.pipeline_parallel as pp

        if getattr(pp, "_dsv4_dump_forward_only", False):
            return
        orig = pp.get_forward_backward_func

        def _get():
            fn = orig()

            def _fn(*args, **kwargs):
                kwargs["forward_only"] = True
                return fn(*args, **kwargs)

            return _fn

        pp.get_forward_backward_func = _get
        pp._dsv4_dump_forward_only = True
        try:
            import megatron.training.training as meg_training

            meg_training.get_forward_backward_func = _get
        except Exception:
            pass
        print("[act-dump] patched pipeline engine forward_only=True", flush=True)
    except Exception as exc:
        print(f"[act-dump] WARN could not patch forward_only: {exc}", flush=True)


def _passthrough_layer_forward(mod):
    orig = mod.forward

    def _fwd(*args, **kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        context = kwargs.get("context")
        if context is None and len(args) >= 3:
            context = args[2]
        return hidden, context

    _fwd._dsv4_orig_forward = orig  # noqa: SLF001
    mod.forward = _fwd


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _as_hidden(output) -> torch.Tensor | None:
    if output is None:
        return None
    if isinstance(output, dict):
        for key in ("hidden_states", "hidden", "x"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        return None
    if isinstance(output, (tuple, list)):
        if not output:
            return None
        output = output[0]
    if not torch.is_tensor(output):
        return None
    return output


def _parallel_ranks() -> tuple[int, int, int]:
    try:
        from megatron.core import parallel_state as mpu

        tp = mpu.get_tensor_model_parallel_rank()
        pp = mpu.get_pipeline_model_parallel_rank()
        ep = (
            mpu.get_expert_model_parallel_rank()
            if hasattr(mpu, "get_expert_model_parallel_rank")
            else 0
        )
        return int(tp), int(pp), int(ep)
    except Exception:
        return 0, 0, 0


def _maybe_gather_sp(hidden: torch.Tensor) -> torch.Tensor:
    try:
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
        from megatron.training import get_args

        args = get_args()
        if not getattr(args, "sequence_parallel", False):
            return hidden
        if mpu.get_tensor_model_parallel_world_size() <= 1:
            return hidden
        if hidden.size(-1) != int(getattr(args, "hidden_size", 4096)):
            return hidden
        return gather_from_sequence_parallel_region(hidden)
    except Exception:
        return hidden


def _iter_layers(root):
    for name, mod in root.named_modules():
        if not hasattr(mod, "layer_number"):
            continue
        if not (hasattr(mod, "self_attention") or hasattr(mod, "mlp")):
            continue
        yield name, mod


def _module_tree(mod, prefix: str = "", depth: int = 0, max_depth: int = 2) -> list[str]:
    lines = []
    for name, child in mod.named_children():
        extra = ""
        if hasattr(child, "layer_number"):
            extra += f" layer_number={child.layer_number}"
        if hasattr(child, "compress_ratio"):
            extra += f" compress_ratio={child.compress_ratio}"
        if hasattr(child, "layer_id"):
            extra += f" layer_id={child.layer_id}"
        lines.append(f"{prefix}{name}: {type(child).__name__}{extra}")
        if depth < max_depth:
            lines.extend(_module_tree(child, prefix + "  ", depth + 1, max_depth))
    return lines


def _find_embeddings(root):
    found = []
    for name, mod in root.named_modules():
        if hasattr(mod, "word_embeddings"):
            found.append((name or "embedding", mod))
    return found


def install_layer_act_dump(model, tag: str | None = None) -> int:
    """Register forward hooks. Returns number of layer hooks (not inner)."""
    if not _enabled():
        return 0

    root = unwrap_model(model)
    dump_dir = Path(os.environ.get("DSV4_DUMP_DIR", "/root/models/dsv4_layer_act"))
    tag = tag or os.environ.get("DSV4_DUMP_TAG", "lumen")
    max_fwd = int(os.environ.get("DSV4_DUMP_LAYER_ACT_MAX", "1"))
    stop_after = _stop_after_layer()
    dump_dir.mkdir(parents=True, exist_ok=True)
    if stop_after > 0:
        _patch_forward_only()
        root.eval()

    hooked = 0
    skipped = 0
    counts: dict[str, int] = {}

    def _save(
        key: str,
        layer_id: int,
        tensor: torch.Tensor,
        stage: str | None,
        *,
        gather: bool = True,
    ) -> None:
        n = counts.get(key, 0)
        if n >= max_fwd:
            return
        counts[key] = n + 1
        gathered = _maybe_gather_sp(tensor.detach()) if gather else tensor.detach()
        tp, pp, ep = _parallel_ranks()
        if tp != 0 or ep != 0:
            return
        cpu = gathered.float().contiguous().cpu()
        if stage:
            path = dump_dir / f"{tag}_pp{pp:02d}_l{int(layer_id):02d}_{stage}_fwd{n}.pt"
        else:
            path = dump_dir / f"{tag}_pp{pp:02d}_l{int(layer_id):02d}_fwd{n}.pt"
        torch.save(
            {
                "tag": tag,
                "layer": int(layer_id),
                "stage": stage or "layer_out",
                "pp": pp,
                "tp": tp,
                "ep": ep,
                "fwd": n,
                "shape": tuple(cpu.shape),
                "hidden": cpu,
            },
            path,
        )
        print(f"[act-dump] {path} shape={tuple(cpu.shape)}", flush=True)

    def _make_out_hook(layer_id: int, stage: str | None, key: str):
        def _hook(_mod, _inp, output):
            hidden = _as_hidden(output)
            if hidden is None:
                return
            _save(key, layer_id, hidden, stage)

        return _hook

    def _make_pre_hook(layer_id: int, stage: str, key: str):
        def _hook(_mod, args, kwargs=None):
            hidden = _as_hidden(args)
            if hidden is None:
                hidden = _as_hidden(kwargs)
            if hidden is None:
                return
            _save(key, layer_id, hidden, stage)

        return _hook

    for _name, mod in _iter_layers(root):
        lid = int(getattr(mod, "layer_number", 0))
        if stop_after > 0 and lid > stop_after:
            _passthrough_layer_forward(mod)
            skipped += 1
            continue
        mod.register_forward_hook(_make_out_hook(lid, None, f"layer{lid}"))
        hooked += 1
        if _inner_enabled():
            if lid == 1:
                tree = _module_tree(mod, max_depth=2)
                msg = "[act-dump] layer 1 module tree:\n  " + "\n  ".join(tree)
                print(msg, flush=True)
                (dump_dir / f"{tag}_layer1_modules.txt").write_text(msg + "\n")
            mod.register_forward_pre_hook(
                _make_pre_hook(lid, "layer_in", f"l{lid}_layer_in"),
                with_kwargs=True,
            )
            for stage, attr in (
                ("input_layernorm", "input_layernorm"),
                ("self_attention", "self_attention"),
                ("self_attn_bda", "self_attn_bda"),
                ("pre_mlp_layernorm", "pre_mlp_layernorm"),
                ("mlp", "mlp"),
                ("mlp_bda", "mlp_bda"),
            ):
                child = getattr(mod, attr, None)
                if child is None or not isinstance(child, torch.nn.Module):
                    continue
                if type(child).__name__ in ("IdentityOp", "IdentityFuncOp"):
                    continue
                child.register_forward_hook(_make_out_hook(lid, stage, f"l{lid}_{stage}"))
            mlp = getattr(mod, "mlp", None)
            if mlp is not None:
                router = getattr(mlp, "router", None)
                if router is not None:

                    def _router_hook(mod, inp, output, *, _lid=lid):
                        hidden_in = _as_hidden(inp)
                        if hidden_in is not None:
                            _save(f"l{_lid}_router_in", _lid, hidden_in, "router_in")
                        weight = getattr(mod, "weight", None)
                        if torch.is_tensor(weight):
                            _save(
                                f"l{_lid}_router_weight",
                                _lid,
                                weight.detach(),
                                "router_weight",
                                gather=False,
                            )
                        bias = getattr(mod, "expert_bias", None)
                        if torch.is_tensor(bias):
                            _save(
                                f"l{_lid}_router_expert_bias",
                                _lid,
                                bias.detach(),
                                "router_expert_bias",
                                gather=False,
                            )
                        tid = getattr(mod, "tid2eid", None)
                        if torch.is_tensor(tid):
                            _save(
                                f"l{_lid}_router_tid2eid",
                                _lid,
                                tid.detach(),
                                "router_tid2eid",
                                gather=False,
                            )
                        if hidden_in is not None and torch.is_tensor(weight):
                            x = hidden_in.reshape(-1, hidden_in.size(-1)).to(torch.float32)
                            w = weight.detach().to(torch.float32)
                            b = getattr(mod, "bias", None)
                            b = b.detach().to(torch.float32) if torch.is_tensor(b) else None
                            logits = torch.nn.functional.linear(x, w, b)
                            _save(
                                f"l{_lid}_router_logits",
                                _lid,
                                logits,
                                "router_logits",
                                gather=False,
                            )
                        if not isinstance(output, (tuple, list)) or len(output) < 2:
                            return
                        probs, routing_map = output[0], output[1]
                        if torch.is_tensor(probs):
                            _save(
                                f"l{_lid}_router_probs",
                                _lid,
                                probs.detach(),
                                "router_probs",
                                gather=False,
                            )
                        if torch.is_tensor(routing_map):
                            _save(
                                f"l{_lid}_router_map",
                                _lid,
                                routing_map.detach().to(torch.float32),
                                "router_map",
                                gather=False,
                            )

                    router.register_forward_hook(_router_hook)
                    print(
                        f"[act-dump] router hook on layer {lid} {type(router).__name__}",
                        flush=True,
                    )
                shared = getattr(mlp, "shared_experts", None)
                if shared is not None and isinstance(shared, torch.nn.Module):
                    if type(shared).__name__ not in ("IdentityOp", "IdentityFuncOp"):
                        shared.register_forward_hook(
                            _make_out_hook(lid, "shared_experts", f"l{lid}_shared_experts")
                        )
            attn = getattr(mod, "self_attention", None)
            if attn is not None:
                for stage, attr in (
                    ("attn_wq_a", "wq_a"),
                    ("attn_q_norm", "q_norm"),
                    ("attn_wq_b", "wq_b"),
                    ("attn_wkv", "wkv"),
                    ("attn_kv_norm", "kv_norm"),
                    ("attn_wo_b", "wo_b"),
                ):
                    child = getattr(attn, attr, None)
                    if child is None or not isinstance(child, torch.nn.Module):
                        continue
                    child.register_forward_hook(
                        _make_out_hook(lid, stage, f"l{lid}_{stage}")
                    )

    if _inner_enabled():
        for name, emb in _find_embeddings(root):
            emb.register_forward_hook(_make_out_hook(1, "embed", f"embed:{name}"))
            print(f"[act-dump] embedding hook on {name} {type(emb).__name__}", flush=True)

    print(
        f"[act-dump] installed {hooked} layer hooks tag={tag} dir={dump_dir} "
        f"max_fwd={max_fwd} layer1_inner={int(_inner_enabled())} "
        f"stop_after={stop_after} skipped={skipped}",
        flush=True,
    )
    return hooked


def install_mhc_dx_ln_hooks(model) -> int:
    """Hook input_ln / pre_mlp_ln outputs for DSV4_DUMP_MHC_DX=1."""
    if os.environ.get("DSV4_DUMP_MHC_DX", "0") != "1":
        return 0
    from lumen.models.dsv4.ops.hyper_connection import _dsv4_hook_dh

    root = unwrap_model(model)
    n = 0
    for _name, mod in _iter_layers(root):
        lid = int(getattr(mod, "layer_number", 0))
        for attr, site, sub in (
            ("input_layernorm", "after_input_ln", "attn"),
            ("pre_mlp_layernorm", "after_pre_mlp_ln", "ffn"),
        ):
            child = getattr(mod, attr, None)
            if child is None or not isinstance(child, torch.nn.Module):
                continue
            if type(child).__name__ in ("IdentityOp", "IdentityFuncOp"):
                continue

            def _fh(_m, _inp, output, *, _lid=lid, _site=site, _sub=sub):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if torch.is_tensor(hidden):
                    _dsv4_hook_dh(hidden, _site, _lid, _sub)

            child.register_forward_hook(_fh)
            n += 1
    print(f"[mhc-dx] installed {n} LN grad hooks", flush=True)
    return n
