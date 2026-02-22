import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import QuadrickTradingBot

DURATION_SECONDS = 120 * 60


async def run_window() -> int:
    bot = QuadrickTradingBot()
    run_task = None
    try:
        logger.info('LIVE 2h runner: initializing bot')
        await bot.initialize()
        logger.info('LIVE 2h runner: bot initialized, starting main loop')
        run_task = asyncio.create_task(bot.run())
        await asyncio.sleep(DURATION_SECONDS)
        logger.info('LIVE 2h runner: duration reached, shutting down bot')
        await bot.shutdown()
        if run_task:
            try:
                await asyncio.wait_for(run_task, timeout=max(90, bot.decision_interval * 3))
            except asyncio.TimeoutError:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
        logger.info('LIVE 2h runner: completed cleanly')
        return 0
    except Exception as exc:
        logger.exception(f'LIVE 2h runner failed: {exc}')
        try:
            await bot.shutdown()
        except Exception:
            pass
        if run_task:
            run_task.cancel()
            try:
                await run_task
            except Exception:
                pass
        return 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(run_window()))
