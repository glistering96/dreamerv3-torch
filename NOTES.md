# DreamerV3 TSP Implementation Notes

> **목적**: AI assistant가 대화 시작 시 이 파일을 읽고 프로젝트 컨텍스트를 빠르게 파악하도록 함.
> **Repo**: NM512/dreamerv3-torch 기반, TSP/CVRP용으로 수정됨.

---

## Quick Reference

| Item | Value |
|------|-------|
| Best Return | -4.17 (optimal ~-3.8) |
| Mean Return | -7.7 |
| Key Fix | Advantage scaling (`adv / adv_scale`) |
| Main Issue | imag_mask accuracy ~50% |
| Verdict | 🟡 연구용 OK, Production 비추천 |

---

## 1. Architecture Overview

### Original vs Current

| Component | Original DreamerV3 | Current (TSP) |
|-----------|-------------------|---------------|
| Actor | MLP | TSPPointerActor (attention) |
| Heads | decoder, reward, cont | + mask, current_node, current_pos, coords |
| Imag reward | objective callback | Simulated distance |
| Mask | N/A | mask_simulation |

### Key Files

| File | Content |
|------|---------|
| `models.py` | WorldModel, ImagBehavior |
| `networks.py` | RSSM, TSPPointerActor |
| `envs/routing.py` | TSPEnv, CVRPEnv |
| `configs.yaml` | tsp_attn config |

---

## 2. Applied Fixes

### 2.1 Advantage Scaling (Critical)

**문제**: actor_grad_norm 0.03→0.009 vanishing  
**해결**: `models.py`

```python
adv_scale = torch.clamp(adv.abs().mean(), min=0.1)
adv = adv / adv_scale
```

### 2.2 Hyperparameters

| Param | Before | After |
|-------|--------|-------|
| discount | 0.997 | 0.99 |
| imag_horizon | 15 | 10 |
| actor.entropy | 1e-4 | 1e-3 |
| actor.lr | 1e-4 | 3e-4 |

### 2.3 WM Reward Head (2026-01-08)

**문제**: Simulated distance reward 사용 → WM reward head 미활용  
**해결**: `models.py` Line 352

```python
# 이전: reward = sim_reward
reward = reward_head(imag_feat).mode()
```

### 2.4 Raw Advantage in REINFORCE (2026-01-08)

**문제**: EMA normalized advantage → Original과 다름  
**해결**: `models.py` Line 574-580

```python
raw_adv = target - self.value(imag_feat[:-1]).mode()
adv_scale = torch.clamp(raw_adv.abs().mean(), min=0.1)
raw_adv = raw_adv / adv_scale  # scaling은 유지
```

**새 실험**: `logdir_wmreward` (2026-01-08 23:00 시작)

---

## 3. Known Issues

### ⚠️ Issue 1: Value Loss Double Target

```python
value_loss -= value.log_prob(slow_target.mode().detach())
```

→ Original에도 동일, 의도적 설계로 추정

### ⚠️ Issue 2: Imagination Mask Accuracy

`imag_mask_step_0 ≈ 9.5/19` (50% 정확도)  
→ World model이 방문 상태를 잘 예측 못함

### ⚠️ Issue 3: REINFORCE Advantage

- Original: `target - value.mode()` (raw)
- Current: EMA normalized + scaled

---

## 4. Loss Audit Summary

| Loss | Distribution | Status |
|------|-------------|--------|
| coords | Normal | ❓ |
| mask | Binary | ✅ |
| current_node | Categorical | ❓ |
| current_pos | Normal | ✅ |
| reward | SymlogDisc | ✅ |
| KL | Dual KL | ✅ |
| Value | SymlogDisc | ⚠️ |
| Actor | REINFORCE | ✅ |

### Loss 필요성 분석

| Loss | 필수? | 이유 |
|------|-------|------|
| coords | ❓ | 에피소드 내 고정값, 매번 예측 불필요 |
| current_node | ❓ | current_pos와 중복 (둘 다 현재 위치) |
| mask | ✅ | imagination에서 방문 추적에 필수 |

---

## 5. Future Experiments

| ID | Experiment | Priority |
|----|-----------|----------|
| F1 | current_pos를 노드 인덱스로 변경 | Low |
| F2 | imag_gradient: dynamics 테스트 | Medium |
| F3 | Curriculum (TSP-5→10→20) | Medium |
| F4 | objective callback 복원 | High |
| F5 | PPO + PointerNet baseline | High |
| F6 | coords_loss 제거 실험 | Medium |
| F7 | current_node_loss 제거 실험 | Medium |

---

## 6. Current Experiment Status (2026-01-11)

### Step 54,000 🔴 COLLAPSED

| Metric | 초기 | 현재 |
|--------|------|------|
| Best | -4.31 | (변화없음) |
| **Last 50 Mean** | -7.6 | **-38.0** 💥 |
| actor_grad_norm | 0.07 | **NaN** 💥 |
| entropy | 0.1 | 3.0 (random) |
| value_min | -3.6 | **-178** 💥 |
| target_min | -12 | **-179** 💥 |

**Quartiles**: Q1=-8.2 → Q4=**-38.0** (완전 붕괴)

### 결론

학습 **완전 붕괴**. Value explosion 수정 없이 진행 불가.

---

## 7. Improvement Plan

### 🔴 우선순위 1: Imagination Mask 정확도

**문제**: mask_step_0이 50%만 정확 → imagination rollout이 잘못된 마스크로 시작

**해결책 옵션**:

- A) Ground truth mask 강제 주입 (현재도 시도 중이지만 불완전)
- B) Mask head의 loss weight 증가
- C) Mask를 one-hot이 아닌 continuous로 변경

### 🟡 우선순위 2: Actor Exploration

**문제**: entropy 0.11로 빠르게 수렴 → local optimum에 갇힘

**해결책**:

- actor.entropy coefficient 증가 (1e-3 → 5e-3)
- Initial random exploration steps 추가

### 🟢 우선순위 3: Architecture 단순화

**문제**: 불필요한 prediction head가 학습 방해 가능

**해결책**:

- coords_loss 제거 (에피소드 내 고정값)
- current_node_loss 제거 (current_pos와 중복)

---

## 8. Next Actions

1. [ ] value explosion 수정 (아래 옵션 중 선택)
2. [ ] mask ground truth 주입 로직 검증

---

## 9. 🚨 Value Explosion Issue (2026-01-11)

### 현상

| Metric | 초기 | 현재 |
|--------|------|------|
| value_min | -3.6 | **-130** |
| target_min | -12 | **-140** |
| value_std | 0.01 | **36** |

### 원인 분석 (악순환)

```
Mask 50% 부정확 → invalid action 선택 → penalty -2.0 누적
     ↓
Value target -100+ → Value network -130 학습
     ↓
Bootstrap (value[-1] = -130) → Lambda-return 전파
     ↓
target_min = -140 → Value 더 나빠짐 (악순환)
```

### 수정 옵션

| 옵션 | 수정 파일 | 수정 내용 |
|------|----------|----------|
| **A** | `models.py:498-504` | imagination에서 penalty 제거 |
| **B** | `models.py:530-535` | value target clipping 추가 |
| **C** | `models.py:437-444` | mask ground truth 강제 주입 수정 |

### 옵션 A: Imagination Penalty 제거

```python
# models.py:498-504
# 변경 전: sim_reward += penalty
# 변경 후: penalty 조건 삭제
```

### 옵션 B: Value Target Clipping

```python
# models.py:530-535 근처
target = tools.lambda_return(...)
target = torch.clamp(target, min=-20, max=0)  # 추가
```

### 옵션 C: Mask Ground Truth 강제

```python
# models.py:437-444: start['mask'] 사용 확인
# 현재 불완전할 수 있음 - 디버깅 필요
```

---

## 10. Root Cause Analysis (2026-01-11)

### 원본 DreamerV3와 비교 결과

| 항목 | 원본 | 현재 (문제) |
|------|------|------------|
| imag_gradient | `dynamics` | `reinforce` |
| REINFORCE | raw advantage | adv_scale 나눗셈 → NaN |

**결론**: 원본은 `dynamics` 모드가 기본값. Actor가 world model을 통해 backprop.

### 적용된 수정

- `configs.yaml`: `imag_gradient: dynamics` 로 변경
- penalty: -2.0 → -0.5
- target clipping 유지

---

## 11. Mask Miss 제거 전략

### 현재 상황

- miss_rate: 0.7%
- 원인: imagination 중 actor가 invalid action 선택

### 왜 0%가 필수인가?

- TSP에서 이미 방문한 노드 재방문은 **논리적으로 불가능**
- 0.7%도 누적되면 학습 신호 왜곡
- 실제 환경에서는 mask가 강제되므로 imagination과 괴리 발생

### 해결 옵션

#### A. Actor 출력에 Hard Masking 강제 (권장)

```python
# networks.py: TSPPointerActor
def forward(self, feat, mask=None, ...):
    logits = self.compute_logits(...)
    if mask is not None:
        logits = torch.where(mask, logits, torch.tensor(-1e9))  # ← 강제
    return OneHotDist(logits)
```

- **효과**: 완전히 0% miss 보장
- **주의**: gradient flow에 영향 없음 (logits만 조정)

#### B. Masked Sampling 구현

```python
# action 샘플 후 강제 보정
action = policy(...).sample()
if mask is not None:
    action = action * mask  # invalid action 0으로
    action = action / action.sum()  # renormalize
```

- **효과**: 샘플링 레벨에서 보정
- **주의**: gradient disconnection 가능

#### C. Penalty-Free Imagination

- Penalty 제거하고 mask miss를 허용하되 무시
- 학습에 영향 없이 진행
- **비권장**: 근본 해결 아님

### 권장 순서

1. ~~**옵션 A 구현**~~ - 이미 구현됨!
2. **실제 문제 발견** (아래 참조)

---

## 12. 🔴 Root Cause: unimix_ratio (2026-01-11)

### 발견

`TSPPointerActor.forward()`에는 이미 hard masking 있음:

```python
logits = logits.masked_fill(~mask, float('-inf'))  # ✅ 적용됨
```

**하지만!** `OneHotDist.__init__`에서:

```python
probs = probs * (1 - unimix_ratio) + unimix_ratio / num_actions
#                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 이게 모든 action에 1% uniform 확률 추가 → masked도 포함!
```

### 해결책 (한 줄 수정)

```python
# networks.py Line 1069
# Before:
return tools.OneHotDist(logits, unimix_ratio=self._unimix_ratio)

# After:
return tools.OneHotDist(logits, unimix_ratio=self._unimix_ratio, mask=mask)
```

`OneHotDist`에 이미 `mask` 파라미터가 있고, unimix 후 재적용 로직도 있음!

---

*Last updated: 2026-01-11*
