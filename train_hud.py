from __future__ import annotations

import asyncio


async def main() -> None:
    """Skeleton for HUD-native grouped rollouts.

    The exact trainer API can evolve with HUD. Use this file as the place to wire:
    tasks -> grouped eval job -> trainer.step(batch, group_size=N).
    """
    raise SystemExit(
        "Wire this to your HUD TrainingClient after `hud deploy .` and a baseline `hud eval` succeed."
    )


if __name__ == "__main__":
    asyncio.run(main())
