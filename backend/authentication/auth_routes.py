"""
Sea Sentinel - RBAC Authentication Endpoints
===========================================
Provides login, identity verification, and role switching endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from backend.authentication.auth_service import (
    UserProfile,
    LoginRequest,
    SwitchRoleRequest,
    TokenResponse,
    UserRole,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_user_profile_by_role,
    SEED_USERS,
    TOKEN_EXPIRY_SECONDS
)

auth_router = APIRouter(prefix="/api/auth", tags=["Authentication & RBAC"])


@auth_router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """
    Authenticates user credentials and returns JWT Bearer token with profile and permissions.
    """
    profile = authenticate_user(req.email, req.password)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials. Use 'admin@seasentinel.ocean' / 'admin123' or 'operator@seasentinel.ocean' / 'user123'."
        )
    
    token = create_access_token({
        "user_id": profile.user_id,
        "email": profile.email,
        "role": profile.role,
        "full_name": profile.full_name
    })
    
    return TokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_in=TOKEN_EXPIRY_SECONDS,
        user=profile
    )


@auth_router.get("/me", response_model=UserProfile)
async def get_current_user_profile(user: UserProfile = Depends(get_current_user)):
    """
    Returns the profile and permissions of the currently authenticated user.
    """
    return user


@auth_router.post("/switch-role", response_model=TokenResponse)
async def switch_role(req: SwitchRoleRequest, current_user: UserProfile = Depends(get_current_user)):
    """
    Convenience method for rapid role switching between ADMIN and USER profiles.
    Generates a valid token and user profile for the target role.
    """
    target_role = req.target_role.upper()
    if target_role not in [UserRole.ADMIN, UserRole.USER]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target role '{req.target_role}'. Must be 'ADMIN' or 'USER'."
        )
    
    profile = get_user_profile_by_role(target_role)
    token = create_access_token({
        "user_id": profile.user_id,
        "email": profile.email,
        "role": profile.role,
        "full_name": profile.full_name
    })
    
    return TokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_in=TOKEN_EXPIRY_SECONDS,
        user=profile
    )


@auth_router.get("/roles")
async def get_role_definitions():
    """
    Returns metadata about the available roles and their operational scopes.
    """
    return {
        "roles": [
            {
                "role": UserRole.ADMIN,
                "title": "Administrator / Hydrographer",
                "scope": "Entire Ocean Map & Model Operations",
                "default_account": "admin@seasentinel.ocean",
                "features": [
                    "Entire Ocean Map (Global Dataset)",
                    "AI/ML Model Management & Weights",
                    "Real-Time Model Evaluation & Tuning",
                    "Global Debris Telemetry & Heatmap",
                    "Full System & Pipeline Analytics",
                    "Admin Configuration & Security Settings"
                ]
            },
            {
                "role": UserRole.USER,
                "title": "Operator / Surveillance Analyst",
                "scope": "Current Input Survey Isolation",
                "default_account": "operator@seasentinel.ocean",
                "features": [
                    "Current Input Survey GIS (Strict Isolation)",
                    "Target & Debris Visual Verification",
                    "Local Threat & Risk Classification",
                    "Input-Scoped Tactical Metrics",
                    "Formal Oceanographic Survey PDF Report"
                ]
            }
        ]
    }
