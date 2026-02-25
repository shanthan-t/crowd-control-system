"""
jwt_utils.py — JWT encode / decode for Sentinel auth.
Uses PyJWT. Token contains {sub (username), role, exp}.
"""

import jwt
import os
import datetime

SECRET_KEY = os.environ.get("SENTINEL_JWT_SECRET", "sentinel-secret-key-change-in-prod")
ALGORITHM = "HS256"
EXPIRY_HOURS = 24


def encode_token(username: str, role: str) -> str:
    """Create a signed JWT token."""
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=EXPIRY_HOURS),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Returns:
        {"sub": username, "role": role}

    Raises:
        jwt.ExpiredSignatureError — token expired
        jwt.InvalidTokenError — bad token
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return {"sub": payload["sub"], "role": payload["role"]}
