# Spec Summary

Source: `/Users/sraising/Downloads/CartScout- RL Context Manager for Real Browser Shopping Agents.pdf`

CartScout should be a real-browser shopping context manager, not an autonomous purchasing bot. The
agent browses public product pages, verifies shopping constraints, cites visible evidence, emits a
compact purchase packet, and stops before checkout for human or frontier-model approval.

## RL Objective

Reward uncertainty reduction under a token/action budget:

- Verify category, variant, seller, price, availability, and delivery/pickup.
- Verify must-have and must-not-have constraints.
- Collect evidence quotes that support claims.
- Compress the result into a JSON packet.
- Avoid unsafe actions.

Do not reward raw click sequences.

## MVP

- HUD browser environment with CDP and RFB capabilities.
- Safe shopping task specs.
- Deterministic reward function.
- Grouped rollouts with reward spread.
- Fireworks single-turn RFT path for snippets -> packet JSON.

## Stretch

- Optional cart-prep tasks.
- Snapshot-backed quote verification for reproducible training.
- Fireworks remote multi-turn browser RFT.
