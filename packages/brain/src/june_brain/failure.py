"""Two ways to fail, named, so the choice is visible at the call site.

`except Exception: pass` was used 29 times in this codebase for three different
situations, and the reader could not tell them apart:

1. A decorative feature failed and nobody needs to know.
2. A functional feature failed and the user should be told June is doing less.
3. A safety check could not be evaluated.

The third is the dangerous one. It looked exactly like the first two, and that
is how the privacy dial came to fail open in two of the three places it was
enforced (D.2): the handler was written in the shape everything else used.

So the shapes are now different:

- :func:`degrade_quietly` — cases 1 and 2. Never re-raises, always leaves a log
  line. Silence is the thing being fixed here; a swallowed failure is a turn
  whose Glass Box trace is quietly incomplete, and the Glass Box is the product.
- :func:`fail_closed` — case 3. Use it in any handler guarding a privacy, guard,
  ledger, or consent decision, where "we could not check" must not read as
  "permitted". See :mod:`june_brain.privacy` for the worked example.

For case 2 specifically, pass ``user_visible=True``. It does not itself surface
anything to the user — the caller still has to degrade and say so — but it logs
at warning rather than debug, so the difference is greppable.
"""

from __future__ import annotations

import logging
from typing import NoReturn

logger = logging.getLogger(__name__)


def degrade_quietly(what: str, *, user_visible: bool = False) -> None:
    """Record that ``what`` failed and that June is carrying on without it.

    Call from inside an ``except`` block. ``what`` names the thing that did not
    happen, in the user's terms where possible ("temporal context", "trace
    cleanup"), because these lines are read when someone asks why a turn looked
    thin.

    ``user_visible=True`` marks a functional degradation the caller is expected
    to surface — it raises the log level, it does not do the surfacing.
    """
    logger.log(
        logging.WARNING if user_visible else logging.DEBUG,
        "degraded: %s",
        what,
        exc_info=True,
    )


def fail_closed(what: str, exc: BaseException | None = None) -> NoReturn:
    """Record that a safety check could not be evaluated, and refuse.

    Raises. A check that cannot evaluate itself has not established that the
    action is permitted, and the two failure directions are not symmetric: a
    false refusal costs a turn that degrades and explains itself, while a false
    permission costs data leaving a machine whose owner set a dial to stop
    exactly that.

    Prefer a predicate that returns the safe value (as ``privacy.local_only``
    does) where one fits; reach for this when the caller has no safe value to
    return and must not continue.
    """
    logger.warning("failing closed: %s", what, exc_info=True)
    raise RuntimeError(f"safety check could not be evaluated: {what}") from exc
