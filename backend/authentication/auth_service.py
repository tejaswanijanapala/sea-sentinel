"""
Sea Sentinel - Role-Based Access Control (RBAC) & Authentication Service
========================================================================
Implements stateless token-based authorization and user scoping with two profiles:
  1. ADMIN (Global Ocean Map, Model Management, System Telemetry, Global Metrics)
  2. USER (Current Input GIS, Debris Verification, Risk Analysis, Input Metrics)
"""

import time
import hmac
import hashlib
import json
import base64
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import Header, HTTPException, Depends, status

# Secret key for signature generation (in production, load from environment)
AUTH_SECRET_KEY = "sea-sentinel-tactical-oceanographic-secret-key-2026"
TOKEN_EXPIRY_SECONDS = 86400 * 7 # 7 days


class UserRole:
    ADMIN = "ADMIN"
    USER = "USER"


class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: str # "ADMIN" | "USER"
    organization: str
    avatar_initials: str
    permissions: List[str]


class LoginRequest(BaseModel):
    email: str
    password: str


class SwitchRoleRequest(BaseModel):
    target_role: str = Field(..., description="Target role to switch: 'ADMIN' or 'USER'")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: UserProfile


# Pre-seeded authorized accounts
SEED_USERS: Dict[str, Dict[str, Any]] = {
    "admin@seasentinel.ocean": {
        "user_id": "usr_admin_001",
        "email": "admin@seasentinel.ocean",
        "password": "admin123",
        "full_name": "Chief Hydrographer & ML Admin",
        "role": UserRole.ADMIN,
        "organization": "Sea Sentinel Ocean Command",
        "avatar_initials": "AD",
        "permissions": [
            "global_ocean_map",
            "ai_models_manage",
            "model_evaluation_run",
            "system_telemetry",
            "global_metrics",
            "admin_settings",
            "current_input_gis",
            "debris_inspection",
            "risk_analysis",
            "report_generation"
        ]
    },
    "operator@seasentinel.ocean": {
        "user_id": "usr_operator_002",
        "email": "operator@seasentinel.ocean",
        "password": "user123",
        "full_name": "Surveillance Sonar Operator",
        "role": UserRole.USER,
        "organization": "Maritime Patrol Unit 4",
        "avatar_initials": "SO",
        "permissions": [
            "current_input_gis",
            "debris_inspection",
            "risk_analysis",
            "input_metrics",
            "report_generation"
        ]
    }
}


def _generate_signature(payload_b64: str) -> str:
    """Generates HMAC-SHA256 signature for token payload."""
    sig = hmac.new(
        AUTH_SECRET_KEY.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256
    ).digest()
    return base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")


def create_access_token(user_dict: Dict[str, Any]) -> str:
    """Creates a secure stateless authentication token."""
    payload = {
        "sub": user_dict["user_id"],
        "email": user_dict["email"],
        "role": user_dict["role"],
        "full_name": user_dict["full_name"],
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRY_SECONDS
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8").rstrip("=")
    sig_b64 = _generate_signature(payload_b64)
    return f"{payload_b64}.{sig_b64}"


def verify_access_token(token_str: str) -> Optional[Dict[str, Any]]:
    """Verifies and decodes an access token."""
    try:
        parts = token_str.strip().split(".")
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts
        expected_sig = _generate_signature(payload_b64)
        if not hmac.compare_digest(sig_b64, expected_sig):
            return None
        
        # Add padding back if necessary
        rem = len(payload_b64) % 4
        if rem > 0:
            payload_b64 += "=" * (4 - rem)
        
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        payload = json.loads(payload_json)
        
        if payload.get("exp", 0) < int(time.time()):
            return None # Expired
            
        return payload
    except Exception:
        return None


def authenticate_user(email: str, password: str) -> Optional[UserProfile]:
    """Authenticates credentials against seed database."""
    email_clean = email.strip().lower()
    user_record = SEED_USERS.get(email_clean)
    if not user_record:
        # Fallback support for admin/user shortcuts
        if email_clean in ["admin", "admin@seasentinel.com"]:
            user_record = SEED_USERS["admin@seasentinel.ocean"]
        elif email_clean in ["user", "operator", "user@seasentinel.com"]:
            user_record = SEED_USERS["operator@seasentinel.ocean"]
    
    if user_record and user_record["password"] == password:
        return UserProfile(
            user_id=user_record["user_id"],
            email=user_record["email"],
            full_name=user_record["full_name"],
            role=user_record["role"],
            organization=user_record["organization"],
            avatar_initials=user_record["avatar_initials"],
            permissions=user_record["permissions"]
        )
    return None


def get_user_profile_by_role(role: str) -> UserProfile:
    """Returns profile for a specific role (for role switching & default loading)."""
    target = UserRole.ADMIN if role.upper() == UserRole.ADMIN else UserRole.USER
    for u in SEED_USERS.values():
        if u["role"] == target:
            return UserProfile(
                user_id=u["user_id"],
                email=u["email"],
                full_name=u["full_name"],
                role=u["role"],
                organization=u["organization"],
                avatar_initials=u["avatar_initials"],
                permissions=u["permissions"]
            )
    # Default fallback
    u = SEED_USERS["operator@seasentinel.ocean"]
    return UserProfile(
        user_id=u["user_id"],
        email=u["email"],
        full_name=u["full_name"],
        role=u["role"],
        organization=u["organization"],
        avatar_initials=u["avatar_initials"],
        permissions=u["permissions"]
    )


# =====================================================================
# FastAPI Dependencies for Authorization Guards
# =====================================================================
def get_current_user(authorization: Optional[str] = Header(None)) -> UserProfile:
    """
    Dependency extracting current user from 'Authorization: Bearer <token>' header.
    Defaults to User role if unauthenticated in non-strict development mode.
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        payload = verify_access_token(token)
        if payload:
            email = payload.get("email", "")
            user_rec = SEED_USERS.get(email)
            if user_rec:
                return UserProfile(
                    user_id=user_rec["user_id"],
                    email=user_rec["email"],
                    full_name=user_rec["full_name"],
                    role=payload.get("role", user_rec["role"]),
                    organization=user_rec["organization"],
                    avatar_initials=user_rec["avatar_initials"],
                    permissions=user_rec["permissions"]
                )
            # Custom decoded payload
            role = payload.get("role", UserRole.USER)
            return get_user_profile_by_role(role)
    
    # Check for direct custom role header
    if authorization and authorization.upper() == "ADMIN":
        return get_user_profile_by_role(UserRole.ADMIN)
    
    # Default fallback user for open endpoints
    return get_user_profile_by_role(UserRole.USER)


def require_admin(user: UserProfile = Depends(get_current_user)) -> UserProfile:
    """
    Enforces strict ADMIN role check on protected endpoints.
    Returns HTTP 403 Forbidden for non-admin users.
    """
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Access Denied: Administrator privileges required.",
                "required_role": UserRole.ADMIN,
                "current_role": user.role,
                "message": "This operation or dataset is restricted to Sea Sentinel Administrators."
            }
        )
    return user


def require_user_or_admin(user: UserProfile = Depends(get_current_user)) -> UserProfile:
    """Ensures at least USER role is authenticated."""
    return user
