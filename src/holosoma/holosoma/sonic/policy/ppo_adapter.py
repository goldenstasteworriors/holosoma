"""PPO adapter for SONIC universal policy.

This adapter lets us reproduce SONIC Sec. 3.2 within Holosoma's existing PPO
training loop by keeping the public PPO interface unchanged:

- Input: a single `actor_obs` tensor
- Output: action mean tensor (used to define the PPO Gaussian policy)

We additionally compute SONIC's auxiliary losses (reconstruction/token/cycle)
and expose them via `get_aux_losses()` so PPO can add them to the actor loss.

Expected actor_obs layout for SONIC experiments:
- The first observation term is `motion_command_bundle` (see
    `holosoma.managers.observation.terms.sonic`). It returns:
        [dr, dh, dm, onehot(3), g_r(dr), g_h(dh), g_m(dm)]
- The remaining features are treated as proprio (unused by this minimal
    reproduction but kept to match paper's state definition).
"""

from __future__ import annotations

from dataclasses import dataclass

from holosoma.sonic.commands.multi_command import SonicCommandType
from holosoma.sonic.policy.decoders import ControlDecoderMLP, MotionDecoderMLP
from holosoma.sonic.policy.encoders import HybridCommandEncoderMLP, HumanCommandEncoderMLP, RobotCommandEncoderMLP
from holosoma.sonic.policy.fsq import FiniteScalarQuantizer, IdentityQuantizer
from holosoma.sonic.policy.universal_policy import UniversalPolicyModules, UniversalSonicPolicy
from holosoma.utils.safe_torch_import import torch


@dataclass(frozen=True)
class SonicUniversalActorCfg:
    """Configuration for SONIC universal PPO actor adapter."""

    latent_dim: int = 128

    # Quantizer
    fsq_enabled: bool = True
    fsq_levels: int | list[int] = 8

    # Aux loss weights
    recon_coef: float = 1.0
    token_coef: float = 1.0
    cycle_coef: float = 1.0


class UniversalSonicPolicyActorAdapter(torch.nn.Module):
    """Adapter: tensor actor_obs -> tensor action_mean."""

    def __init__(
        self,
        *,
        action_dim: int,
        decoder_hidden_dims: list[int],
        control_decoder_hidden_dims: list[int] | None = None,
        motion_decoder_hidden_dims: list[int] | None = None,
        latent_dim: int,
        cfg: SonicUniversalActorCfg | None = None,
        robot_encoder_hidden_dims: list[int] | None = None,
        human_encoder_hidden_dims: list[int] | None = None,
        hybrid_encoder_hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        # NOTE: This adapter builds the internal SONIC policy lazily on the first
        # forward pass (since dr/dh/dm are only known then). If the adapter gets
        # moved to a device (e.g., CUDA) before the first forward, lazily-built
        # modules would otherwise remain on CPU and cause device-mismatch errors.
        # This buffer moves with `.to(device)` and lets us place lazy modules on
        # the correct device when they are created.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

        self.action_dim = int(action_dim)
        self.decoder_hidden_dims = list(decoder_hidden_dims)
        self.control_decoder_hidden_dims = (
            list(control_decoder_hidden_dims)
            if control_decoder_hidden_dims is not None
            else list(self.decoder_hidden_dims)
        )
        self.motion_decoder_hidden_dims = (
            list(motion_decoder_hidden_dims)
            if motion_decoder_hidden_dims is not None
            else list(self.decoder_hidden_dims)
        )
        self.latent_dim = int(latent_dim)
        self.cfg = cfg or SonicUniversalActorCfg(latent_dim=self.latent_dim)

        # SONIC-specific defaults:
        # - must be distinct per modality
        # - must not silently reuse LayerConfig.hidden_dims
        if robot_encoder_hidden_dims is None:
            robot_encoder_hidden_dims = [512, 256]
        if human_encoder_hidden_dims is None:
            human_encoder_hidden_dims = [512, 512]
        if hybrid_encoder_hidden_dims is None:
            hybrid_encoder_hidden_dims = [256, 256]

        self.robot_encoder_hidden_dims = list(robot_encoder_hidden_dims)
        self.human_encoder_hidden_dims = list(human_encoder_hidden_dims)
        self.hybrid_encoder_hidden_dims = list(hybrid_encoder_hidden_dims)

        self._built = False
        self.policy: UniversalSonicPolicy | None = None
        self._bundle_header_dim = 6  # [dr,dh,dm] + onehot(3)
        self._last_aux_losses: dict[str, torch.Tensor] = {}

    def _build_once(self, *, dr: int, dh: int, dm: int) -> None:
        if self._built:
            return

        if self.cfg.fsq_enabled:
            levels = self.cfg.fsq_levels
            if isinstance(levels, int):
                # Default to a small codebook-dim (4) as commonly used in FSQ.
                levels_list = [int(levels)] * 4
            else:
                levels_list = [int(x) for x in levels]
            quantizer = FiniteScalarQuantizer(levels=levels_list, dim=self.latent_dim)
        else:
            quantizer = IdentityQuantizer()

        modules = UniversalPolicyModules(
            robot_encoder=RobotCommandEncoderMLP(
                command_dim=dr,
                latent_dim=self.latent_dim,
                hidden_dims=self.robot_encoder_hidden_dims,
            ),
            human_encoder=HumanCommandEncoderMLP(
                command_dim=dh,
                latent_dim=self.latent_dim,
                hidden_dims=self.human_encoder_hidden_dims,
            ),
            hybrid_encoder=HybridCommandEncoderMLP(
                command_dim=dm,
                latent_dim=self.latent_dim,
                hidden_dims=self.hybrid_encoder_hidden_dims,
            ),
            quantizer=quantizer,
            control_decoder=ControlDecoderMLP(
                token_dim=self.latent_dim,
                action_dim=self.action_dim,
                hidden_dims=self.control_decoder_hidden_dims,
            ),
            motion_decoder=MotionDecoderMLP(
                token_dim=self.latent_dim,
                recon_dim=dr,
                hidden_dims=self.motion_decoder_hidden_dims,
            ),
        )
        self.policy = UniversalSonicPolicy(modules)
        self.policy.to(self._device_anchor.device)
        self._built = True

    def _maybe_build_from_state_dict(self, state_dict: dict[str, torch.Tensor], *, prefix: str = "") -> None:
        if self._built:
            return

        # Infer modality dims from the first Linear layer weights.
        try:
            w_r = state_dict.get(prefix + "policy.robot_encoder.net.0.weight")
            w_h = state_dict.get(prefix + "policy.human_encoder.net.0.weight")
            w_m = state_dict.get(prefix + "policy.hybrid_encoder.net.0.weight")
            if w_r is not None and w_h is not None and w_m is not None:
                dr = int(w_r.shape[1])
                dh = int(w_h.shape[1])
                dm = int(w_m.shape[1])
                if dr > 0 and dh > 0 and dm > 0:
                    self._build_once(dr=dr, dh=dh, dm=dm)
        except Exception:
            # Fall back to default behavior if inference fails.
            return

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Called during recursive parent `load_state_dict`. Build lazy SONIC
        # modules before default loading so `policy.*` keys become expected.
        if not self._built:
            self._maybe_build_from_state_dict(state_dict, prefix=prefix)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """Load state dict for lazily-built SONIC modules.

        The SONIC universal policy modules are created on the first forward pass
        based on (dr, dh, dm) parsed from the observation bundle header.

        When resuming from checkpoint, the adapter is constructed in an
        un-built state, but the checkpoint contains `policy.*` parameters.
        If we call the default `load_state_dict` directly, these keys appear as
        unexpected. To support resume/eval/export, infer (dr, dh, dm) from the
        checkpoint weights and build the modules before loading.
        """

        if not self._built:
            self._maybe_build_from_state_dict(state_dict, prefix="")

        return super().load_state_dict(state_dict, strict=strict)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return dict(self._last_aux_losses)

    def _parse_bundle(
        self, actor_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        # Bundle is the first block of actor_obs.
        if actor_obs.shape[1] < self._bundle_header_dim:
            raise ValueError(f"actor_obs too small for SONIC bundle: {actor_obs.shape[1]}")

        header = actor_obs[:, : self._bundle_header_dim]
        dr = int(torch.round(header[0, 0]).item())
        dh = int(torch.round(header[0, 1]).item())
        dm = int(torch.round(header[0, 2]).item())
        onehot = header[:, 3:6]
        cmd_type_id = torch.argmax(onehot, dim=-1)

        start = self._bundle_header_dim
        end_r = start + dr
        end_h = end_r + dh
        end_m = end_h + dm
        if end_m > actor_obs.shape[1]:
            raise ValueError(
                f"SONIC bundle dims exceed actor_obs: dr={dr}, dh={dh}, dm={dm}, obs_dim={actor_obs.shape[1]}"
            )

        g_r = actor_obs[:, start:end_r]
        g_h = actor_obs[:, end_r:end_h]
        g_m = actor_obs[:, end_h:end_m]
        proprio = actor_obs[:, end_m:]
        return cmd_type_id, g_r, g_h, g_m, proprio, dr, dh, dm

    @staticmethod
    def _id_to_type(i: int) -> SonicCommandType:
        if i == 0:
            return "robot"
        if i == 1:
            return "human"
        if i == 2:
            return "hybrid"
        raise ValueError(f"Invalid command type id: {i}")

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        cmd_type_id, g_r, g_h, g_m, _proprio, dr, dh, dm = self._parse_bundle(actor_obs)
        self._build_once(dr=dr, dh=dh, dm=dm)
        assert self.policy is not None
        assert self.policy.motion_decoder is not None

        # Encode+quantize all modalities.
        z_r = self.policy.robot_encoder(g_r).z
        z_h = self.policy.human_encoder(g_h).z
        z_m = self.policy.hybrid_encoder(g_m).z

        q_r = self.policy.quantizer(z_r)
        q_h = self.policy.quantizer(z_h)
        q_m = self.policy.quantizer(z_m)

        # Action path uses the selected modality token.
        tokens = q_r.tokens
        mask_h = cmd_type_id == 1
        mask_m = cmd_type_id == 2
        if mask_h.any():
            tokens = torch.where(mask_h.unsqueeze(-1), q_h.tokens, tokens)
        if mask_m.any():
            tokens = torch.where(mask_m.unsqueeze(-1), q_m.tokens, tokens)
        action = self.policy.control_decoder(tokens=tokens).action

        # Aux losses (Sec. 3.2).
        recon_r = self.policy.motion_decoder(tokens=q_r.tokens).recon
        recon_h = self.policy.motion_decoder(tokens=q_h.tokens).recon
        recon_m = self.policy.motion_decoder(tokens=q_m.tokens).recon

        recon_loss = (
            torch.nn.functional.mse_loss(recon_r, g_r)
            + torch.nn.functional.mse_loss(recon_h, g_r)
            + torch.nn.functional.mse_loss(recon_m, g_r)
        )
        token_loss = torch.nn.functional.mse_loss(q_r.tokens, q_h.tokens)

        z_cycle = self.policy.robot_encoder(recon_h).z
        q_cycle = self.policy.quantizer(z_cycle)
        cycle_loss = torch.nn.functional.mse_loss(q_cycle.tokens, q_r.tokens)

        self._last_aux_losses = {
            "sonic_recon_loss": self.cfg.recon_coef * recon_loss,
            "sonic_token_loss": self.cfg.token_coef * token_loss,
            "sonic_cycle_loss": self.cfg.cycle_coef * cycle_loss,
        }

        return action


def build_sonic_universal_actor_mean(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    latent_dim: int | None = None,
    robot_encoder_hidden_dims: list[int] | None = None,
    human_encoder_hidden_dims: list[int] | None = None,
    hybrid_encoder_hidden_dims: list[int] | None = None,
    control_decoder_hidden_dims: list[int] | None = None,
    motion_decoder_hidden_dims: list[int] | None = None,
    fsq_enabled: bool = True,
    fsq_levels: int | list[int] = 8,
    recon_coef: float = 1.0,
    token_coef: float = 1.0,
    cycle_coef: float = 1.0,
) -> torch.nn.Module:
    """Factory used by BaseModule for `type=="SonicUniversal"`."""

    if latent_dim is None:
        latent_dim = hidden_dims[-1] if len(hidden_dims) > 0 else 128

    # Note: command dims (dr/dh/dm) are inferred lazily from the SONIC bundle
    # header, and modules are built on the first forward pass.
    cfg = SonicUniversalActorCfg(
        latent_dim=latent_dim,
        fsq_enabled=bool(fsq_enabled),
        fsq_levels=fsq_levels,
        recon_coef=float(recon_coef),
        token_coef=float(token_coef),
        cycle_coef=float(cycle_coef),
    )
    return UniversalSonicPolicyActorAdapter(
        action_dim=output_dim,
        decoder_hidden_dims=hidden_dims,
        control_decoder_hidden_dims=control_decoder_hidden_dims,
        motion_decoder_hidden_dims=motion_decoder_hidden_dims,
        latent_dim=latent_dim,
        cfg=cfg,
        robot_encoder_hidden_dims=robot_encoder_hidden_dims,
        human_encoder_hidden_dims=human_encoder_hidden_dims,
        hybrid_encoder_hidden_dims=hybrid_encoder_hidden_dims,
    )
