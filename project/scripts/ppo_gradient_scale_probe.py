"""Does the value loss swamp the policy gradient in PPO's shared trunk? (untrained net, one rollout, training-code math)"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eligibility import eligibility_from_features
from ppo.actor_critic import ActorCritic, RolloutBuffer, mask_switch_logits
from ppo.config import PPOConfig
NET = os.environ.get("TLS_NET_FILE", os.path.join(_ROOT, "sumo_4x4_network", "grid4x4.net.xml")); ROUTE = os.environ.get("TLS_ROUTE_FILE", os.path.join(_ROOT, "sumo_4x4_network", "routes.rou.xml"))
torch.manual_seed(0); cfg = PPOConfig(); H = 200
env = TrafficGridEnv(GridEnvConfig(net_file=NET, route_file=ROUTE, episode_seconds=1200)); net = ActorCritic(11, cfg.hidden_sizes, 2)
buf = RolloutBuffer(H, env.n_agents, 11); obs, _ = env.reset(seed=1000); rews = []; raw = []
for _ in range(H):
    f = obs["node_features"]; el = eligibility_from_features(f)
    a, lp, v = net.act(torch.as_tensor(f, dtype=torch.float32), torch.as_tensor(el))
    obs, r, te, tr, info = env.step(a.numpy()); raw.append(info["agent_rewards"].copy()); ar = np.clip(info["agent_rewards"], -cfg.reward_clip, cfg.reward_clip); rews.append(ar)
    buf.add(f, a.numpy(), lp.numpy(), v.numpy(), ar, float(te or tr), el)
with torch.no_grad(): _, lastv = net(torch.as_tensor(obs["node_features"], dtype=torch.float32))
adv, ret = buf.compute_gae(lastv.numpy(), cfg.gamma, cfg.gae_lambda)
bo, ba, blp, badv, bret, bel = buf.get_flat(adv, ret)
ae = badv[bel]; badv = (badv - ae.mean()) / (ae.std() + 1e-8)
T = lambda x: torch.as_tensor(x)
rews = np.array(rews); raw = np.array(raw); print(f"fraction of per-junction rewards at the +-{cfg.reward_clip:.0f} clip: {(np.abs(raw) >= cfg.reward_clip).mean():.4f}"); print(f"per-junction reward per step: mean {rews.mean():.2f}, std {rews.std():.2f}, min {rews.min():.1f}, max {rews.max():.1f}")
print(f"value targets (returns): mean {bret.mean():.1f}, std {bret.std():.1f}, max |.| {np.abs(bret).max():.1f}")
params = list(net.parameters()); rng = np.random.default_rng(0); ratios = []; n_pol = []; n_val = []; tot = []
for _ in range(20):
    idx = rng.choice(len(bo), 64, replace=False)
    logits, vp = net(T(bo[idx])); logits = mask_switch_logits(logits, T(bel[idx]))
    dist = torch.distributions.Categorical(logits=logits); newlp = dist.log_prob(T(ba[idx])); ent = dist.entropy().mean()
    ratio = torch.exp(newlp - T(blp[idx])); adv_t = T(badv[idx])
    pol = -torch.min(ratio * adv_t, torch.clamp(ratio, 0.8, 1.2) * adv_t).mean(); val = (vp - T(bret[idx])).pow(2).mean()
    gp = torch.autograd.grad(pol - cfg.entropy_coeff * ent, params, retain_graph=True, allow_unused=True)
    gv = torch.autograd.grad(cfg.value_coeff * val, params, retain_graph=True, allow_unused=True)
    npol = torch.sqrt(sum((g ** 2).sum() for g in gp if g is not None)).item(); nval = torch.sqrt(sum((g ** 2).sum() for g in gv if g is not None)).item()
    n_pol.append(npol); n_val.append(nval); tot.append((npol ** 2 + nval ** 2) ** 0.5)
    last = (pol.item(), val.item())
print(f"value_loss (minibatch) ~ {last[1]:.1f} vs |policy_loss| ~ {abs(last[0]):.3f}")
print(f"grad norm: policy+entropy {np.mean(n_pol):.3f} | 0.5*value {np.mean(n_val):.3f} | ratio value/policy {np.mean(n_val) / np.mean(n_pol):.1f}x")
print(f"max_grad_norm={cfg.max_grad_norm}: clip factor on the total gradient = {min(1, cfg.max_grad_norm / np.mean(tot)):.3f} -> "
      f"policy gradient after clipping ~ {np.mean(n_pol) * min(1, cfg.max_grad_norm / np.mean(tot)):.4f}")
# --- the same gradients WITH cfg.scale_value_loss: loss/sigma^2 scales the value gradient by exactly 1/sigma^2 ---
ret_scale = max(float(bret.std()), 1.0); nv_s = np.mean(n_val) / ret_scale ** 2; tot_s = (np.mean(n_pol) ** 2 + nv_s ** 2) ** 0.5
clip_s = min(1, cfg.max_grad_norm / tot_s)
print(f"WITH scale_value_loss (ret_scale={ret_scale:.1f}): value grad {nv_s:.4f} | ratio value/policy {nv_s / np.mean(n_pol):.2f}x | "
      f"clip factor {clip_s:.3f} -> policy gradient after clipping ~ {np.mean(n_pol) * clip_s:.4f}   [config flag is currently {cfg.scale_value_loss}]")
env.close()
