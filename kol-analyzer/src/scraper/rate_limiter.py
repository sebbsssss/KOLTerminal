"""Human-like rate limiting for web scraping."""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HumanLikeRateLimiter:
    """
    Rate limiter that mimics human browsing patterns.

    Features:
    - Random delays using beta distribution (clusters toward middle)
    - Burst mode (occasional rapid scrolling like humans)
    - Fatigue simulation (gets slower over time)
    - Periodic breaks every 50-100 requests
    - Micro-jitter on all delays
    """

    base_delay: tuple = (3.0, 8.0)  # Min/max seconds between requests
    burst_probability: float = 0.15  # Chance of rapid requests
    fatigue_factor: float = 0.1  # Slowdown per hour of activity

    # Internal state
    _request_count: int = field(default=0, init=False)
    _session_start: float = field(default_factory=time.time, init=False)
    _last_request: float = field(default=0.0, init=False)
    _in_burst: bool = field(default=False, init=False)
    _burst_remaining: int = field(default=0, init=False)
    _break_interval: int = field(default=75, init=False)  # Requests before break

    def __post_init__(self):
        """Initialize break interval with randomness."""
        self._break_interval = random.randint(50, 100)
        self._session_start = time.time()

    def _get_fatigue_multiplier(self) -> float:
        """Calculate slowdown based on session duration."""
        hours_active = (time.time() - self._session_start) / 3600
        return 1.0 + (self.fatigue_factor * hours_active)

    def _beta_delay(self) -> float:
        """
        Generate delay using beta distribution.
        This creates more human-like delays that cluster in the middle.
        """
        min_delay, max_delay = self.base_delay
        # Beta(2, 2) creates a bell curve centered at 0.5
        beta_value = random.betavariate(2, 2)
        return min_delay + (max_delay - min_delay) * beta_value

    def _add_micro_jitter(self, delay: float) -> float:
        """Add small random variations to any delay."""
        jitter = random.uniform(-0.3, 0.3)
        return max(0.5, delay + jitter)

    async def wait(self) -> float:
        """
        Wait for an appropriate amount of time before the next request.

        Returns:
            The actual delay in seconds.
        """
        self._request_count += 1

        # Check if we need a break
        if self._request_count >= self._break_interval:
            await self._take_break()
            self._break_interval = random.randint(50, 100)
            self._request_count = 0

        # Determine if we're in burst mode
        if self._in_burst and self._burst_remaining > 0:
            # Fast scroll during burst
            delay = random.uniform(0.5, 1.5)
            self._burst_remaining -= 1
            if self._burst_remaining == 0:
                self._in_burst = False
        elif random.random() < self.burst_probability:
            # Start a burst
            self._in_burst = True
            self._burst_remaining = random.randint(3, 7)
            delay = random.uniform(0.5, 1.5)
        else:
            # Normal delay with beta distribution
            delay = self._beta_delay()

        # Apply fatigue
        delay *= self._get_fatigue_multiplier()

        # Add micro-jitter
        delay = self._add_micro_jitter(delay)

        # Actually wait
        await asyncio.sleep(delay)
        self._last_request = time.time()

        return delay

    async def _take_break(self):
        """Simulate a human taking a short break."""
        # Breaks are 15-45 seconds
        break_duration = random.uniform(15.0, 45.0)
        await asyncio.sleep(break_duration)

    def reset(self):
        """Reset the rate limiter state."""
        self._request_count = 0
        self._session_start = time.time()
        self._last_request = 0.0
        self._in_burst = False
        self._burst_remaining = 0
        self._break_interval = random.randint(50, 100)

    @property
    def stats(self) -> dict:
        """Get current rate limiter statistics."""
        return {
            'request_count': self._request_count,
            'session_duration_minutes': (time.time() - self._session_start) / 60,
            'fatigue_multiplier': self._get_fatigue_multiplier(),
            'in_burst': self._in_burst,
            'requests_until_break': self._break_interval - self._request_count
        }


class SimpleRateLimiter:
    """Simple rate limiter for demo mode - minimal delays."""

    def __init__(self, delay: float = 0.1):
        self.delay = delay

    async def wait(self) -> float:
        """Wait a short fixed time."""
        await asyncio.sleep(self.delay)
        return self.delay

    def reset(self):
        """Reset is a no-op for simple limiter."""
        pass

    @property
    def stats(self) -> dict:
        return {'mode': 'simple', 'delay': self.delay}
