import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        self.heads["mask"] = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.mask_head["layers"] if "mask_head" in config else 2,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.mask_head["outscale"] if "mask_head" in config else 1.0,
            device=config.device,
            name="Mask",
        )
        self.heads["current_node"] = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.decoder["mlp_layers"],
            config.units,
            config.act,
            config.norm,
            dist="onehot",
            device=config.device,
            name="CurrentNode",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
            mask=config.mask_head["loss_scale"] if "mask_head" in config else 1.0,
            current_node=1.0,
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    target = data[name]
                    if name == 'reward' and len(target.shape) == 2:
                        target = target.unsqueeze(-1)
                    loss = -pred.log_prob(target)
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }

        if 'image' in obs:
            obs["image"] = obs["image"] / 255.0
            
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        if self._config.actor.get("type", "mlp") == "tsp_pointer":
            self.actor = networks.TSPPointerActor(
                feat_size,
                config.num_actions,
                embed_dim=config.units,
                unimix_ratio=config.actor["unimix_ratio"],
            )
        else:
            self.actor = networks.MLP(
                feat_size,
                (config.num_actions,),
                config.actor["layers"],
                config.units,
                config.act,
                config.norm,
                config.actor["dist"],
                config.actor["std"],
                config.actor["min_std"],
                config.actor["max_std"],
                absmax=1.0,
                temp=config.actor["temp"],
                unimix_ratio=config.actor["unimix_ratio"],
                outscale=config.actor["outscale"],
                name="Actor",
            )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # 1. Obtain Rollout Data
                imag_feat, imag_state, imag_action, imag_mask, sim_reward, imag_position, imag_mask_misses = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )

                # 2. Extract Coords for Masked Actor Loss
                coords = start.get('coords')
                actor_coords = None
                if coords is not None:
                    # coords: (B, T, N*2) -> (B*T, N, 2)
                    actor_coords = coords.reshape(-1, self._config.num_actions, 2)
                    # Expand for horizon: (H, B*T, N, 2)
                    actor_coords = actor_coords.unsqueeze(0).expand(imag_feat.shape[0], -1, -1, -1)

                # 3. Compute Value Target
                # FIX: Use world model reward head instead of simulated distance
                # This aligns with original DreamerV3 (objective callback)
                reward_head = self._world_model.heads["reward"]
                reward = reward_head(imag_feat).mode()
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )

                # 4. Compute Actor Loss
                actor_ent = self.actor(imag_feat, mask=imag_mask, coords=actor_coords, current_pos=imag_position).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                    mask=imag_mask,
                    coords=actor_coords,
                    current_pos=imag_position,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        # Add Debug Metrics for Imagination
        with torch.no_grad():
            # imag_mask is (horizon, batch_size*time, num_nodes)
            mask_sums = imag_mask.float().sum(dim=-1).mean(dim=1) # (horizon,)
            for h in range(len(mask_sums)):
                metrics[f"imag_mask_sum_step_{h}"] = to_np(mask_sums[h])
            
            # imag_reward is (horizon, batch_size*time, 1)
            reward_means = reward.mean(dim=1).squeeze(-1) # (horizon,)
            for h in range(len(reward_means)):
                 metrics[f"imag_reward_step_{h}"] = to_np(reward_means[h])
            
            # Mask miss tracking - how many invalid actions per step
            if imag_mask_misses is not None:
                miss_rate_per_step = imag_mask_misses.squeeze(-1).mean(dim=1)  # (horizon,)
                total_misses = imag_mask_misses.sum()
                metrics["imag_mask_miss_total"] = to_np(total_misses)
                metrics["imag_mask_miss_rate"] = to_np(imag_mask_misses.mean())
                for h in range(len(miss_rate_per_step)):
                    metrics[f"imag_mask_miss_step_{h}"] = to_np(miss_rate_per_step[h])

        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        """
        Imagination rollout with deterministic mask simulation.
        PROPER IMPLEMENTATION: Pre-calculates initial mask and uses state passing.
        """
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        
        # Check if mask simulation is enabled
        use_mask_sim = getattr(self._config, 'mask_simulation', False)
        
        # Pre-calculate initial mask for the first step
        # This prevents NoneType issues and ensures correct starting state
        feat = dynamics.get_feat(start)
        inp = feat.detach()
        if "mask" in self._world_model.heads:
            # FIX: Use ground truth mask from start state if available
            # start['mask'] should be available if injected in dreamer.py
            # Note: start is flattened (B*T, ...). We need to check if 'mask' is in start.
            if 'mask' in start:
                 # Flattened mask: (B*T, N)
                 init_mask = start['mask']
                 # Ensure float/bool compatibility if needed, though boolean is fine for logic
                 if init_mask.dtype != torch.bool:
                     init_mask = init_mask > 0.5
            else:
                 init_mask = self._world_model.heads["mask"](inp).mode()
        else:
            init_mask = None # This will likely fail if sim is required, but consistent with legacy

        # Pre-calculate initial position
        init_pos = start.get('current_pos') # (B*T, 2)
        if init_pos is None:
             # Fallback: find it in coords using current_node if available
             c_node = start.get('current_node') # (B*T, N) one-hot
             coords = start.get('coords') # (B*T, N*2)
             if c_node is not None and coords is not None:
                  idx = c_node.argmax(dim=-1, keepdim=True)
                  c = coords.reshape(coords.shape[0], -1, 2)
                  init_pos = torch.gather(c, 1, idx.unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)
             else:
                  init_pos = torch.zeros((start['stoch'].shape[0], 2), device=dynamics._device)

        def step(prev, _):
            # prev: (state, feat, action, current_mask, prev_pos, next_mask_state, next_pos_state, reward, mask_miss)
            state, _, _, _, prev_pos, prev_next_mask, _, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            
            if use_mask_sim:
                current_mask = prev_next_mask
            else:
                if "mask" in self._world_model.heads:
                    current_mask = self._world_model.heads["mask"](inp).mode()
                else:
                    current_mask = None
            
            coords = start.get('coords') # (B*T, N*2)
            if coords is not None:
                coords = coords.reshape(coords.shape[0], -1, 2)
            
            if isinstance(policy, networks.TSPPointerActor):
                action = policy(inp, coords=coords, mask=current_mask, current_pos=prev_pos).sample()
            else:
                action = policy(inp, mask=current_mask).sample()

            succ = dynamics.img_step(state, action)
            
            # Simulated reward and state transition
            sim_reward = torch.zeros((action.shape[0], 1), device=action.device)
            next_pos = prev_pos
            next_mask_state = current_mask
            
            if coords is not None:
                action_idx = action.argmax(dim=-1, keepdim=True)
                next_pos = torch.gather(coords, 1, action_idx.unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)
                # reward = negative distance
                dist = torch.norm(next_pos - prev_pos, dim=-1, keepdim=True)
                sim_reward = -dist
                
                # Penalty for invalid action - reduced from -2.0 to -0.5 to prevent value explosion
                if current_mask is not None:
                    valid_check = torch.gather(current_mask, -1, action_idx)
                    penalty = torch.where(valid_check, torch.zeros_like(sim_reward), torch.tensor(-0.5, device=sim_reward.device))
                    sim_reward += penalty
                    # Track mask misses (invalid action selections)
                    mask_miss = (~valid_check).float()  # 1 if miss, 0 if valid
                else:
                    mask_miss = torch.zeros_like(sim_reward)

                if use_mask_sim and current_mask is not None:
                    next_mask = current_mask.scatter(-1, action_idx, False)
                    non_depot_mask = next_mask[..., 1:]
                    all_visited = ~non_depot_mask.any(dim=-1, keepdim=True)
                    depot_valid = all_visited.expand_as(next_mask[..., :1])
                    next_mask_state = torch.cat([depot_valid, next_mask[..., 1:]], dim=-1)
            
            return succ, feat, action, current_mask, prev_pos, next_mask_state, next_pos, sim_reward, mask_miss

        # scan: (state, feat, action, mask, pos, next_mask, next_pos, reward, mask_miss)
        succ, feats, actions, masks, positions, _, _, rewards, mask_misses = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None, None, init_pos, init_mask, None, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions, masks, rewards, positions, mask_misses

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        # FIX: Clamp target to prevent value explosion (-30 is safe bound for TSP-20)
        # lambda_return returns a tensor, clamp each element
        if isinstance(target, (list, tuple)):
            target = [torch.clamp(t, min=-30.0, max=0.0) for t in target]
        else:
            target = torch.clamp(target, min=-30.0, max=0.0)
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
        mask=None,
        coords=None,
        current_pos=None,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp, mask=mask, coords=coords, current_pos=current_pos)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            # Advantage scaling to prevent gradient vanishing
            adv_scale = torch.clamp(adv.abs().mean(), min=0.1)
            adv = adv / adv_scale
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["adv_scale"] = to_np(adv_scale)
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            # FIX: Use raw advantage like original DreamerV3, not EMA normalized
            raw_adv = target - self.value(imag_feat[:-1]).mode()
            if self._config.reward_EMA:
                # Still apply advantage scaling to prevent vanishing gradients
                adv_scale = torch.clamp(raw_adv.abs().mean(), min=0.1)
                raw_adv = raw_adv / adv_scale
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None] * raw_adv.detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
