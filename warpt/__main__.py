# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Package entry point, so that ``python3 -m warpt ...`` runs the orchestrator CLI.

Imports :func:`warpt.orchestrator.main` and calls it when executed as ``__main__``.
Reads and writes nothing itself; defines no public functions.
"""

from .orchestrator import main

if __name__ == "__main__":
    main()
