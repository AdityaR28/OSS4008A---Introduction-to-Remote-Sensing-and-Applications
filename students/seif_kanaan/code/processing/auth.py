"""Authentication is explicit and stays outside published data.

Official guide: https://developers.google.com/earth-engine/guides/auth
"""

import re


def initialize(project: str, *, authenticate: bool = False):
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project or ""):
        raise ValueError("Enter your own enabled Google Cloud Project ID (not a number or display name).")
    import ee

    if authenticate:
        ee.Authenticate()
    try:
        ee.Initialize(project=project)
        # A tiny real computation verifies API access before constructing a large graph.
        if ee.Number(1).getInfo() != 1:
            raise RuntimeError("Earth Engine API probe returned an unexpected response.")
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine access failed. In your Google account, select an Earth Engine-registered "
            "Cloud Project with the API enabled and your IAM access granted, then authenticate again. "
            "Do not paste tokens or service-account keys into chat or source files."
        ) from exc
    return ee
